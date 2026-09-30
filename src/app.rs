//! Application state and the pure update function.
//!
//! [`App`] is only ever changed through [`App::update`], which returns the side effects to run
//! as [`Effect`]s. Rendering (`crate::ui`) only reads it.
//!
//! Derived display data (the [`Transcript`] line cache) is refreshed at the end of
//! `update`, except for streamed tokens: those only invalidate the message and the next
//! [`Action::Tick`] re-renders it, which caps markdown work at the tick rate.
//!
//! Persistence: the user message is saved when sent, the assistant reply when it ends
//! (complete, cancelled or failed). A conversation gets its id — and is stored — with its
//! first message.

use std::collections::HashMap;

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType},
};
use ratatui_textarea::{TextArea, WrapMode};

use crate::{
    action::{Action, Effect},
    commands::{self, Arg, CommandId, CommandSpec, Parsed},
    config::Config,
    context::{Context, ContextChunk},
    files::{self, Attachment},
    keymap::KeyContext,
    layout,
    llm::{
        ChatMessage, LlmEvent, RequestId, Usage,
        stream_task::{CompletionJob, JobKind},
    },
    prompt,
    rag::{
        RagConfig,
        indexer::{IndexEvent, IndexReport, Staleness},
        store::CollectionSummary,
    },
    state::{
        Citation, Conversation, MessageId, MessageStatus, ModelPicker, Overlay, Palette, Role,
        ScrollState, Sidebar, Status,
    },
    storage::{ConversationId, ConversationRecord, StoreEvent, StoreRequest, StoredConversation},
    tokens,
    transcript::Transcript,
};

/// The command and path being typed when the input is `/add <path>` or `/index <path>`
/// (only the first argument of `/index`, before any name).
fn path_argument(input: &str) -> Option<(&'static str, &str)> {
    if let Some(rest) = input.strip_prefix("/add ") {
        return Some(("/add ", rest));
    }
    input
        .strip_prefix("/index ")
        .filter(|rest| !rest.contains(' '))
        .map(|rest| ("/index ", rest))
}

/// Splits `/index` arguments into the positional ones and the `--types` option
/// (`Some(vec![])` for `--types all`).
fn index_arguments(arg: &str) -> Result<(Vec<String>, Option<Vec<String>>), String> {
    let mut positional = Vec::new();
    let mut types = None;
    let mut words = commands::split_args(arg).into_iter();
    while let Some(word) = words.next() {
        let value = if let Some(value) = word.strip_prefix("--types=") {
            value.to_owned()
        } else if word == "--types" {
            words
                .next()
                .ok_or_else(|| "--types attend une liste : --types pdf,md,docx".to_owned())?
        } else if word.starts_with("--") {
            return Err(format!("option inconnue : {word} (seule --types existe)"));
        } else {
            positional.push(word);
            continue;
        };
        types = Some(parse_types(&value)?);
    }
    Ok((positional, types))
}

/// `pdf, .MD,word` → `["pdf", "md", "docx"]`; `all` → every type.
fn parse_types(value: &str) -> Result<Vec<String>, String> {
    let mut types = Vec::new();
    for raw in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let lower = raw.trim_start_matches('.').to_lowercase();
        let normalized = match lower.as_str() {
            "all" | "tout" | "tous" => return Ok(Vec::new()),
            "markdown" => "md".to_owned(),
            "word" => "docx".to_owned(),
            "libreoffice" | "writer" => "odt".to_owned(),
            "texte" | "text" => "txt".to_owned(),
            other => other.to_owned(),
        };
        let supported = normalized == "code"
            || crate::rag::extract::FileKind::of(std::path::Path::new(&format!("f.{normalized}")))
                .is_some();
        if !supported {
            return Err(format!(
                "type non pris en charge : {raw} (ex. : pdf, md, txt, docx, odt, code, rs)"
            ));
        }
        if !types.contains(&normalized) {
            types.push(normalized);
        }
    }
    if types.is_empty() {
        return Err("--types attend une liste : --types pdf,md,docx".into());
    }
    Ok(types)
}

/// `2 fichiers modifiés, 1 nouveau` (or that the folder is gone).
pub fn staleness_summary(staleness: &Staleness) -> String {
    if staleness.missing_root {
        return "dossier introuvable".into();
    }
    let mut parts = Vec::new();
    for (count, one, many) in [
        (staleness.added, "nouveau fichier", "nouveaux fichiers"),
        (staleness.modified, "fichier modifié", "fichiers modifiés"),
        (staleness.removed, "fichier supprimé", "fichiers supprimés"),
    ] {
        match count {
            0 => {}
            1 => parts.push(format!("1 {one}")),
            n => parts.push(format!("{n} {many}")),
        }
    }
    parts.join(", ")
}

/// One-line outcome of an indexing run.
pub fn index_summary(report: &IndexReport) -> String {
    let mut parts = vec![format!("{} fichiers", report.files)];
    for (count, label) in [
        (report.added, "ajoutés"),
        (report.updated, "modifiés"),
        (report.removed, "retirés"),
        (report.unchanged, "inchangés"),
        (report.skipped.len(), "ignorés"),
    ] {
        if count > 0 {
            parts.push(format!("{count} {label}"));
        }
    }
    let reset = if report.reset {
        " (modèle d'embedding changé : réindexé)"
    } else {
        ""
    };
    format!(
        "« {} » indexée : {}, {} passages écrits{reset}",
        report.collection,
        parts.join(", "),
        tokens::format_count(u64::try_from(report.passages).unwrap_or(u64::MAX))
    )
}

/// Maximum length of a conversation title, in characters.
const TITLE_MAX_CHARS: usize = 60;

/// The reply currently being streamed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generation {
    pub request_id: RequestId,
    /// Message receiving the tokens (an assistant reply, or a summary).
    pub message_id: MessageId,
    pub kind: JobKind,
}

/// What the app needs to know about a provider (never its API key).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderInfo {
    pub id: String,
    pub label: String,
    /// Runs on this machine.
    pub local: bool,
    /// Model for new conversations.
    pub default_model: Option<String>,
    /// Context window set in the configuration (applies to all its models).
    pub context_window: Option<u64>,
}

/// Where the context window size comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSource {
    /// `context_window` in the configuration.
    Config,
    /// Reported by the server.
    Server,
}

/// Token counts of the last completed request, as measured by the server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Measured {
    pub input_tokens: u64,
    pub output_tokens: Option<u64>,
    /// Number of messages in the conversation when it was measured.
    pub messages: usize,
}

impl Measured {
    /// Tokens occupied in the context after the request: prompt plus reply.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens.unwrap_or(0)
    }
}

/// How full the context is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextUsage {
    pub tokens: u64,
    /// Exact server count (`false`: estimate, or measure plus estimated new messages).
    pub measured: bool,
}

/// Progress of the running `/index`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexProgress {
    pub collection: String,
    pub done: usize,
    /// Files to process; 0 while scanning.
    pub total: usize,
    /// File being processed.
    pub current: String,
}

/// A deletion waiting for confirmation.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Confirm {
    Forget(String),
    DeleteConversation(ConversationId),
}

/// Passages given to the model for the last reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Retrieved {
    /// Prompt number of the first passage (attached files come first).
    pub first_number: usize,
    pub chunks: Vec<ContextChunk>,
}

/// The whole application state.
#[derive(Debug)]
pub struct App {
    /// `false` once the user asked to quit.
    pub running: bool,
    /// Configured providers, in display order.
    pub providers: Vec<ProviderInfo>,
    /// Provider id used for the next request.
    pub provider: String,
    /// Model used for the next request (empty until one is chosen).
    pub model: String,
    /// System prompt sent with every request.
    pub system_prompt: String,
    /// Whether the terminal distinguishes `Shift+Enter` (affects the hints shown).
    pub keyboard_enhanced: bool,
    /// The conversation being displayed.
    pub conversation: Conversation,
    /// Storage id of the conversation; `None` until its first message is sent.
    pub conversation_id: Option<ConversationId>,
    /// Title of the conversation (its first message, shortened).
    pub conversation_title: Option<String>,
    /// What the app is doing right now.
    pub status: Status,
    /// Multi-line message editor.
    pub input: TextArea<'static>,
    /// The running generation, if any.
    pub generation: Option<Generation>,
    /// The conversation list, when open.
    pub sidebar: Option<Sidebar>,
    /// The open popup (model picker, palette, help), if any.
    pub overlay: Option<Overlay>,
    /// Highlighted slash-command suggestion.
    pub suggestion: usize,
    /// Suggestions hidden with Esc until the input changes.
    suggestions_dismissed: bool,
    /// Terminal size.
    pub viewport: Rect,
    /// Scroll position of the conversation.
    pub scroll: ScrollState,
    /// Display lines of the conversation (derived from `conversation`).
    pub transcript: Transcript,
    next_request_id: u64,
    /// Context windows reported by servers, by (provider, model).
    pub known_windows: HashMap<(String, String), u64>,
    /// Document collections and the time they were listed at, once listed (RAG).
    pub collections: Option<(Vec<CollectionSummary>, i64)>,
    /// The running `/index`, if any.
    pub indexing: Option<IndexProgress>,
    /// Report of the last completed `/index` of this session.
    pub last_index: Option<IndexReport>,
    /// Indexing settings (shown in `/collections`).
    pub rag: RagConfig,
    /// Collection searched for each reply of this conversation (`/rag`).
    pub rag_collection: Option<String>,
    /// Collection asked for with `/rag`, applied once the collection list confirms it.
    pending_rag: Option<String>,
    /// Passages retrieved for the last reply of this conversation.
    pub retrieved: Option<Retrieved>,
    /// Collections whose folder changed since they were indexed.
    pub stale: Vec<Staleness>,
    /// `true` once the collections were checked (the first check is announced).
    collections_checked: bool,
    /// Context fill (percent) from which `/compact` is suggested (0: never).
    pub compact_threshold: u8,
    /// Summarize the history before sending once past `compact_threshold`.
    pub auto_compact: bool,
    /// Message to send once the automatic `/compact` running for it is done.
    queued: Option<String>,
    /// Deletion waiting for its command to be run again (`/forget`, `/delete`).
    pending_confirm: Option<Confirm>,
    /// Token counts of the last completed request of this conversation.
    pub measured: Option<Measured>,
    /// Token counts received for the running request.
    request_usage: Usage,
    /// Makes conversation ids unique across sessions (see [`App::with_session_seed`]).
    session_seed: u64,
    next_conversation: u64,
}

impl App {
    /// Creates the initial state from the configuration.
    pub fn new(config: &Config, keyboard_enhanced: bool) -> Self {
        // Keys are irrelevant here: only the runtime builds clients.
        let resolved = config.resolve_providers(|_| None);
        let default = config.default_provider(&resolved);
        let provider = default.map(|p| p.id.clone()).unwrap_or_default();
        let model = default.and_then(|p| p.model.clone()).unwrap_or_default();
        let providers = resolved
            .iter()
            .map(|p| ProviderInfo {
                id: p.id.clone(),
                label: p.label.clone(),
                local: p.local,
                default_model: p.model.clone(),
                context_window: p.context_window,
            })
            .collect();
        Self {
            running: true,
            providers,
            provider,
            model,
            system_prompt: config.system_prompt.clone(),
            keyboard_enhanced,
            conversation: Conversation::new(),
            conversation_id: None,
            conversation_title: None,
            status: Status::Ready,
            input: new_input(),
            generation: None,
            sidebar: None,
            overlay: None,
            suggestion: 0,
            suggestions_dismissed: false,
            viewport: Rect::default(),
            scroll: ScrollState::default(),
            transcript: Transcript::default(),
            next_request_id: 0,
            known_windows: HashMap::new(),
            measured: None,
            collections: None,
            indexing: None,
            last_index: None,
            rag: config.rag.clone(),
            rag_collection: None,
            pending_rag: None,
            retrieved: None,
            stale: Vec::new(),
            collections_checked: false,
            pending_confirm: None,
            compact_threshold: config.compact_threshold,
            auto_compact: config.auto_compact,
            queued: None,
            request_usage: Usage::default(),
            session_seed: 0,
            next_conversation: 0,
        }
    }

    /// Sets the value mixed into new conversation ids. The runtime passes something unique
    /// per session (time and process id) so that ids never collide in the database.
    pub fn with_session_seed(mut self, seed: u64) -> Self {
        self.session_seed = seed;
        self
    }

    /// The current provider.
    pub fn provider_info(&self) -> Option<&ProviderInfo> {
        self.providers.iter().find(|p| p.id == self.provider)
    }

    /// Display name of a provider id (the id itself when unknown).
    pub fn provider_label(&self, id: &str) -> String {
        self.providers
            .iter()
            .find(|p| p.id == id)
            .map_or_else(|| id.to_owned(), |p| p.label.clone())
    }

    /// `true` when the current provider runs on this machine.
    pub fn is_local(&self) -> bool {
        self.provider_info().is_some_and(|p| p.local)
    }

    /// Context window of the current model, and where the number comes from.
    pub fn context_window(&self) -> Option<(u64, WindowSource)> {
        if let Some(tokens) = self.provider_info().and_then(|p| p.context_window) {
            return Some((tokens, WindowSource::Config));
        }
        self.known_windows
            .get(&(self.provider.clone(), self.model.clone()))
            .map(|tokens| (*tokens, WindowSource::Server))
    }

    /// The messages the next request would send (same builder as the real request). The
    /// passages retrieved for the next question are only known when it is sent: those of
    /// the last reply stand in for them.
    pub fn prompt(&self) -> Vec<ChatMessage> {
        let context = Context {
            chunks: self
                .retrieved
                .as_ref()
                .map(|r| r.chunks.clone())
                .unwrap_or_default(),
        };
        prompt::build_messages(
            &self.system_prompt,
            &context,
            self.conversation.context_messages(),
        )
    }

    /// How many tokens the conversation occupies.
    ///
    /// Uses the server's count of the last request when there is one, plus an estimate of
    /// the messages written since; otherwise an estimate of the whole prompt.
    pub fn context_usage(&self) -> ContextUsage {
        let messages = self.conversation.messages();
        match self.measured {
            Some(measured) if measured.messages <= messages.len() => {
                let newer: u64 = messages[measured.messages..]
                    .iter()
                    .map(|m| tokens::estimate_message(&m.content))
                    .sum();
                ContextUsage {
                    tokens: measured.total() + newer,
                    measured: measured.messages == messages.len(),
                }
            }
            _ => ContextUsage {
                tokens: tokens::estimate_prompt(&self.prompt()),
                measured: false,
            },
        }
    }

    /// How full the context is, in percent of the window (`None` when the window is
    /// unknown).
    pub fn context_percent(&self) -> Option<u64> {
        let (window, _) = self.context_window()?;
        Some(tokens::percent(self.context_usage().tokens, window))
    }

    /// `true` when the context passed the `/compact` threshold.
    fn context_nearly_full(&self) -> bool {
        self.compact_threshold > 0
            && self
                .context_percent()
                .is_some_and(|p| p >= u64::from(self.compact_threshold))
    }

    /// Messages `/compact` would summarize (at least a question and its answer).
    fn compactable(&self) -> bool {
        self.conversation
            .context_messages()
            .iter()
            .filter(|m| matches!(m.role, Role::User | Role::Assistant))
            .count()
            >= 2
    }

    /// Asks the server for the current model's window, unless already known.
    fn detect_window(&self) -> Vec<Effect> {
        if self.model.is_empty() || self.context_window().is_some() {
            return Vec::new();
        }
        vec![Effect::DetectContextWindow {
            provider: self.provider.clone(),
            model: self.model.clone(),
        }]
    }

    /// `true` while a reply is being streamed.
    pub fn is_generating(&self) -> bool {
        self.generation.is_some()
    }

    /// Context the keymap needs to interpret keys.
    pub fn key_context(&self) -> KeyContext {
        KeyContext {
            input_empty: self.input.is_empty(),
            completing_path: self.input.lines().len() == 1
                && path_argument(&self.input_text()).is_some(),
            sidebar_open: self.sidebar.is_some(),
            overlay: self.overlay.as_ref().map(Overlay::kind),
            suggestions_open: !self.suggestions().is_empty(),
        }
    }

    /// The model popup, when open.
    pub fn model_picker(&self) -> Option<&ModelPicker> {
        match &self.overlay {
            Some(Overlay::ModelPicker(picker)) => Some(picker),
            _ => None,
        }
    }

    /// Slash commands matching what is being typed (empty when not typing a command).
    pub fn suggestions(&self) -> Vec<&'static CommandSpec> {
        if self.suggestions_dismissed || self.overlay.is_some() || self.sidebar.is_some() {
            return Vec::new();
        }
        commands::suggestions(&self.input_text())
    }

    /// The highlighted suggestion.
    pub fn selected_suggestion(&self) -> Option<&'static CommandSpec> {
        let suggestions = self.suggestions();
        suggestions
            .get(self.suggestion.min(suggestions.len().saturating_sub(1)))
            .copied()
    }

    /// Screen layout for the current terminal size, input and panels.
    pub fn layout(&self) -> layout::AppLayout {
        layout::compute(
            self.viewport,
            self.input.lines().len(),
            self.sidebar.is_some(),
        )
    }

    /// Area where conversation lines are drawn.
    pub fn chat_area(&self) -> Rect {
        layout::chat_content(self.layout().chat)
    }

    /// First visible transcript line.
    pub fn scroll_offset(&self) -> usize {
        let height = usize::from(self.chat_area().height);
        self.scroll.offset(self.transcript.total_lines(), height)
    }

    /// Current input text, lines joined with `\n`.
    pub fn input_text(&self) -> String {
        self.input.lines().join("\n")
    }

    /// Applies an action and returns the side effects to execute.
    pub fn update(&mut self, action: Action) -> Vec<Effect> {
        let is_token = matches!(
            action,
            Action::Llm {
                event: LlmEvent::Token(_),
                ..
            }
        );
        let effects = self.apply(action);
        if !is_token {
            self.refresh_view();
        }
        effects
    }

    fn refresh_view(&mut self) {
        let width = usize::from(self.chat_area().width);
        self.transcript.refresh(
            self.conversation.messages(),
            width,
            self.conversation.context_start(),
        );
    }

    fn apply(&mut self, action: Action) -> Vec<Effect> {
        let total = self.transcript.total_lines();
        let height = usize::from(self.chat_area().height);
        let page = height.saturating_sub(2).max(1);
        match action {
            Action::Quit => {
                self.running = false;
                self.cancel_generation()
            }
            Action::Submit => self.submit(),
            Action::InsertNewline => {
                self.input.insert_newline();
                self.input_changed();
                Vec::new()
            }
            // Esc closes the topmost popup or panel first.
            Action::Cancel => {
                self.pending_confirm = None;
                if self.overlay.take().is_some() {
                    Vec::new()
                } else if let Some(sidebar) = &mut self.sidebar {
                    // Esc undoes the panel's current step before closing it.
                    if sidebar.rename.take().is_some() {
                        self.status = Status::Ready;
                        Vec::new()
                    } else if !sidebar.filter.is_empty() {
                        sidebar.filter.clear();
                        vec![Effect::Store(StoreRequest::List)]
                    } else {
                        self.sidebar = None;
                        Vec::new()
                    }
                } else if self.is_generating() {
                    self.cancel_generation()
                } else if self.indexing.is_some() {
                    vec![Effect::CancelIndex]
                } else {
                    Vec::new()
                }
            }
            Action::OpenModelPicker => self.toggle_model_picker(),
            Action::OpenPalette => {
                self.toggle_overlay(
                    |o| matches!(o, Overlay::Palette(_)),
                    || Overlay::Palette(Palette::default()),
                );
                Vec::new()
            }
            Action::OpenHelp => {
                self.toggle_overlay(
                    |o| matches!(o, Overlay::Help { .. }),
                    || Overlay::Help { scroll: 0 },
                );
                Vec::new()
            }
            Action::OverlayUp => {
                match &mut self.overlay {
                    Some(Overlay::ModelPicker(picker)) => picker.select_previous(),
                    Some(Overlay::Palette(palette)) => palette.select_previous(),
                    Some(_) => self.scroll_popup(-1),
                    None => {}
                }
                Vec::new()
            }
            Action::OverlayDown => {
                match &mut self.overlay {
                    Some(Overlay::ModelPicker(picker)) => picker.select_next(),
                    Some(Overlay::Palette(palette)) => palette.select_next(),
                    Some(_) => self.scroll_popup(1),
                    None => {}
                }
                Vec::new()
            }
            Action::OverlayPageUp => {
                self.scroll_popup(-i32::from(self.popup_page()));
                Vec::new()
            }
            Action::OverlayPageDown => {
                self.scroll_popup(i32::from(self.popup_page()));
                Vec::new()
            }
            Action::OverlayFilter(c) => {
                match &mut self.overlay {
                    Some(Overlay::ModelPicker(picker)) => picker.push_filter(c),
                    Some(Overlay::Palette(palette)) => palette.push_filter(c),
                    _ => {}
                }
                Vec::new()
            }
            Action::OverlayBackspace => {
                match &mut self.overlay {
                    Some(Overlay::ModelPicker(picker)) => picker.pop_filter(),
                    Some(Overlay::Palette(palette)) => palette.pop_filter(),
                    _ => {}
                }
                Vec::new()
            }
            Action::OverlaySelect => match &self.overlay {
                Some(Overlay::ModelPicker(_)) => self.select_model(),
                Some(Overlay::Palette(palette)) => {
                    let command = palette.selected_command();
                    self.overlay = None;
                    match command {
                        Some(spec) if matches!(spec.arg, Arg::Required(_)) => {
                            // The argument is typed in the input box.
                            self.set_input(&format!("/{} ", spec.name));
                            Vec::new()
                        }
                        Some(spec) => self.run_command(spec.id, ""),
                        None => Vec::new(),
                    }
                }
                Some(
                    Overlay::Help { .. }
                    | Overlay::Context { .. }
                    | Overlay::Prompt { .. }
                    | Overlay::Collections { .. },
                ) => {
                    self.overlay = None;
                    Vec::new()
                }
                None => Vec::new(),
            },
            Action::ModelsListed(results) => {
                let providers = self.providers.clone();
                let label_of = |id: &str| {
                    providers
                        .iter()
                        .find(|p| p.id == id)
                        .map_or_else(|| id.to_owned(), |p| p.label.clone())
                };
                for result in &results {
                    for model in result.result.iter().flatten() {
                        if let Some(tokens) = model.context_window {
                            self.known_windows
                                .insert((result.provider.clone(), model.id.clone()), tokens);
                        }
                    }
                }
                if let Some(Overlay::ModelPicker(picker)) = &mut self.overlay {
                    picker.set_models(results, label_of, (&self.provider, &self.model));
                }
                Vec::new()
            }
            Action::SuggestionUp => {
                let len = self.suggestions().len();
                if len > 0 {
                    self.suggestion = (self.suggestion + len - 1) % len;
                }
                Vec::new()
            }
            Action::SuggestionDown => {
                let len = self.suggestions().len();
                if len > 0 {
                    self.suggestion = (self.suggestion + 1) % len;
                }
                Vec::new()
            }
            Action::CompleteSuggestion => {
                if let Some(spec) = self.selected_suggestion() {
                    let completed = match spec.arg {
                        Arg::None => format!("/{}", spec.name),
                        Arg::Optional(_) | Arg::Required(_) => format!("/{} ", spec.name),
                    };
                    self.set_input(&completed);
                }
                Vec::new()
            }
            Action::DismissSuggestions => {
                self.suggestions_dismissed = true;
                Vec::new()
            }
            Action::CompletePath => match path_argument(&self.input_text()) {
                Some((_, partial)) => vec![Effect::CompletePath(partial.to_owned())],
                None => Vec::new(),
            },
            Action::PathCompleted {
                partial,
                candidates,
            } => {
                self.on_path_completed(&partial, &candidates);
                Vec::new()
            }
            Action::FileRead(result) => self.on_file_read(result),
            Action::CopyLastReply => self.run_command(CommandId::Copy, ""),
            Action::Copied { what, chars, how } => {
                let chars = tokens::format_count(u64::try_from(chars).unwrap_or(u64::MAX));
                self.status = Status::Info(match how {
                    crate::clipboard::Copied::Tool(tool) => {
                        format!("copié : {what} ({chars} caractères, via {tool})")
                    }
                    crate::clipboard::Copied::TerminalOnly => format!(
                        "copie ({what}) envoyée au terminal (OSC 52) ; si rien n'est \
                         copié, installez wl-clipboard ou xclip"
                    ),
                });
                Vec::new()
            }
            Action::NewConversation => self.new_conversation(),
            Action::ToggleSidebar => {
                if self.sidebar.take().is_some() {
                    Vec::new()
                } else {
                    self.sidebar = Some(Sidebar::default());
                    vec![Effect::Store(StoreRequest::List)]
                }
            }
            Action::SidebarUp => {
                if let Some(sidebar) = &mut self.sidebar {
                    sidebar.select_previous();
                }
                Vec::new()
            }
            Action::SidebarDown => {
                if let Some(sidebar) = &mut self.sidebar {
                    sidebar.select_next();
                }
                Vec::new()
            }
            Action::SidebarOpen => match self.sidebar.as_ref().and_then(|s| s.rename.clone()) {
                Some(title) => self.finish_rename(&title),
                None => self.open_selected(),
            },
            Action::SidebarType(c) => self.sidebar_edit(|text| text.push(c)),
            Action::SidebarBackspace => self.sidebar_edit(|text| {
                text.pop();
            }),
            Action::SidebarRename => {
                if let Some(sidebar) = &mut self.sidebar
                    && let Some(title) = sidebar.selected_item().map(|i| i.title.clone())
                {
                    sidebar.confirm_delete = None;
                    sidebar.rename = Some(title);
                    self.status =
                        Status::Info("nouveau titre, puis Entrée (Échap pour annuler)".into());
                }
                Vec::new()
            }
            Action::SidebarDelete => self.sidebar_delete(),
            Action::Edit(key) => {
                if self.input.input(key) {
                    self.input_changed();
                }
                Vec::new()
            }
            Action::Paste(text) => {
                // Normalize CRLF / CR line endings so pasted text keeps its line breaks.
                let text = text.replace("\r\n", "\n").replace('\r', "\n");
                self.input.insert_str(text);
                self.input_changed();
                Vec::new()
            }
            Action::ScrollUp(lines) => {
                self.scroll.scroll_up(lines, total, height);
                Vec::new()
            }
            Action::ScrollDown(lines) => {
                self.scroll.scroll_down(lines, total, height);
                Vec::new()
            }
            Action::PageUp => {
                self.scroll.scroll_up(page, total, height);
                Vec::new()
            }
            Action::PageDown => {
                self.scroll.scroll_down(page, total, height);
                Vec::new()
            }
            Action::ScrollToTop => {
                self.scroll.to_top(total, height);
                Vec::new()
            }
            Action::ScrollToBottom => {
                self.scroll.to_bottom();
                Vec::new()
            }
            Action::Resize { width, height } => {
                self.viewport = Rect::new(0, 0, width, height);
                Vec::new()
            }
            Action::Tick => Vec::new(),
            Action::Init => {
                let mut effects = self.detect_window();
                effects.push(Effect::CheckCollections);
                effects
            }
            Action::CollectionsChecked(result) => {
                self.on_collections_checked(result);
                Vec::new()
            }
            Action::ContextWindowDetected {
                provider,
                model,
                tokens,
            } => {
                if let Some(tokens) = tokens.filter(|n| *n > 0) {
                    self.known_windows.insert((provider, model), tokens);
                }
                Vec::new()
            }
            Action::Llm { request_id, event } => self.on_llm_event(request_id, event),
            Action::Storage(event) => self.on_storage_event(event),
            Action::Index(event) => self.on_index_event(event),
        }
    }

    /// Enter: runs a slash command or sends the input as a message.
    fn submit(&mut self) -> Vec<Effect> {
        let text = self.input_text();
        match commands::parse(&text) {
            Parsed::Command { spec, arg } => {
                let arg = arg.to_owned();
                self.set_input("");
                self.run_command(spec.id, &arg)
            }
            Parsed::Unknown(name) => match self.selected_suggestion() {
                // "/hi" + Enter runs the highlighted suggestion ("/history").
                Some(spec) if matches!(spec.arg, Arg::Required(_)) => {
                    self.set_input(&format!("/{} ", spec.name));
                    Vec::new()
                }
                Some(spec) => {
                    self.set_input("");
                    self.run_command(spec.id, "")
                }
                None => {
                    self.status =
                        Status::Error(format!("commande inconnue : /{name} (tapez /help)"));
                    Vec::new()
                }
            },
            Parsed::Escaped(rest) => {
                let rest = rest.to_owned();
                self.send_message(&rest)
            }
            Parsed::Message => self.send_message(&text),
        }
    }

    /// Executes a user command.
    pub fn run_command(&mut self, id: CommandId, arg: &str) -> Vec<Effect> {
        // A deletion is confirmed only by running the command again right away.
        if !matches!(id, CommandId::Forget | CommandId::Delete) {
            self.pending_confirm = None;
        }
        match id {
            CommandId::New => self.new_conversation(),
            CommandId::History => {
                if self.sidebar.is_some() {
                    Vec::new()
                } else {
                    self.sidebar = Some(Sidebar::default());
                    vec![Effect::Store(StoreRequest::List)]
                }
            }
            CommandId::Model if arg.is_empty() => {
                self.overlay = Some(Overlay::ModelPicker(ModelPicker::default()));
                vec![Effect::ListModels]
            }
            CommandId::Model => self.set_model_from_arg(arg),
            CommandId::Help => {
                self.overlay = Some(Overlay::Help { scroll: 0 });
                Vec::new()
            }
            CommandId::Context => {
                self.overlay = Some(Overlay::Context { scroll: 0 });
                self.detect_window()
            }
            CommandId::Prompt => {
                self.overlay = Some(Overlay::Prompt { scroll: 0 });
                Vec::new()
            }
            CommandId::Add => self.attach(arg),
            CommandId::Clear => self.clear_context(),
            CommandId::Compact => self.compact(),
            CommandId::Index => self.start_index(arg),
            CommandId::Rag => self.choose_rag(arg.trim()),
            CommandId::Forget => self.forget(arg.trim()),
            CommandId::Rename => self.rename_current(arg.trim()),
            CommandId::Copy => self.copy(arg.trim()),
            CommandId::Delete => self.delete_current(),
            CommandId::Collections => {
                self.overlay = Some(Overlay::Collections { scroll: 0 });
                let mut effects = self.refresh_collections();
                effects.push(Effect::CheckCollections);
                effects
            }
            CommandId::Quit => self.apply(Action::Quit),
        }
    }

    /// Scrolls the open text popup by `delta` lines, within its content.
    fn scroll_popup(&mut self, delta: i32) {
        let max = crate::ui::popup_max_scroll(self);
        if let Some(scroll) = self.overlay.as_mut().and_then(Overlay::scroll_mut) {
            let next = i32::from(*scroll)
                .saturating_add(delta)
                .clamp(0, i32::from(max));
            *scroll = u16::try_from(next).unwrap_or(0);
        }
    }

    /// Lines moved by PgUp / PgDn in a text popup.
    fn popup_page(&self) -> u16 {
        self.viewport.height.saturating_sub(8).max(1)
    }

    /// Applies path completions if the input did not change meanwhile.
    fn on_path_completed(&mut self, partial: &str, candidates: &[String]) {
        let text = self.input_text();
        let Some((command, typed)) = path_argument(&text) else {
            return;
        };
        if typed != partial {
            return; // the user kept typing
        }
        let command = command.to_owned();
        match candidates {
            [] => {
                self.status = Status::Error(format!("aucun fichier ne correspond à « {partial} »"))
            }
            [only] => self.set_input(&format!("{command}{only}")),
            many => {
                let prefix = files::common_prefix(many);
                if prefix.chars().count() > partial.chars().count() {
                    self.set_input(&format!("{command}{prefix}"));
                }
                let names: Vec<String> = many
                    .iter()
                    .take(6)
                    .map(|c| {
                        let trimmed = c.trim_end_matches('/');
                        let name = files::file_name(trimmed);
                        if c.ends_with('/') {
                            format!("{name}/")
                        } else {
                            name
                        }
                    })
                    .collect();
                let more = if many.len() > 6 { ", …" } else { "" };
                self.status = Status::Info(format!(
                    "{} possibilités : {}{more}",
                    many.len(),
                    names.join(", ")
                ));
            }
        }
    }

    /// Replaces the input text (cursor at the end).
    fn set_input(&mut self, text: &str) {
        self.input = new_input();
        self.input.insert_str(text);
        self.input_changed();
    }

    fn input_changed(&mut self) {
        self.suggestion = 0;
        self.suggestions_dismissed = false;
    }

    fn toggle_model_picker(&mut self) -> Vec<Effect> {
        if matches!(self.overlay, Some(Overlay::ModelPicker(_))) {
            self.overlay = None;
            Vec::new()
        } else {
            self.run_command(CommandId::Model, "")
        }
    }

    /// Closes the popup if it is the one `is_it` recognizes, otherwise opens `make()`.
    fn toggle_overlay(&mut self, is_it: impl Fn(&Overlay) -> bool, make: impl Fn() -> Overlay) {
        if self.overlay.as_ref().is_some_and(is_it) {
            self.overlay = None;
        } else {
            self.overlay = Some(make());
        }
    }

    /// Sends `text` as a user message and starts the reply.
    fn send_message(&mut self, text: &str) -> Vec<Effect> {
        self.pending_confirm = None;
        if self.is_generating() {
            return Vec::new();
        }
        if text.trim().is_empty() {
            return Vec::new();
        }
        if self.model.is_empty() {
            self.status = Status::Error(format!(
                "{} : aucun modèle choisi (F2 ou /model)",
                self.provider_label(&self.provider)
            ));
            return Vec::new();
        }
        if self.auto_compact && self.context_nearly_full() && self.compactable() {
            // Summarize first; the message goes out once the summary is in.
            self.queued = Some(text.trim_end().to_owned());
            self.input = new_input();
            let effect = self.start_job(JobKind::Summary, Role::Summary);
            self.status = Status::Info(
                "contexte presque plein : résumé de l'historique avant l'envoi…".into(),
            );
            return vec![effect];
        }
        self.send_now(text)
    }

    /// Sends `text` as the next user message.
    fn send_now(&mut self, text: &str) -> Vec<Effect> {
        let text = text.trim_end();
        if self.conversation_title.is_none() {
            self.conversation_title = Some(title_from(text));
        }
        let user_id = self
            .conversation
            .push(Role::User, text, MessageStatus::Complete);
        self.input = new_input();
        let mut effects: Vec<Effect> = self.save(user_id).into_iter().collect();
        effects.push(self.start_job(JobKind::Reply, Role::Assistant));
        effects
    }

    /// Starts streaming into a new `role` message, from the messages in the context.
    fn start_job(&mut self, kind: JobKind, role: Role) -> Effect {
        let request_id = RequestId(self.next_request_id);
        self.next_request_id += 1;
        self.request_usage = Usage::default();
        let job = CompletionJob {
            kind,
            request_id,
            provider: self.provider.clone(),
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            history: self.conversation.context_messages().to_vec(),
            rag_collection: match kind {
                JobKind::Reply => self.rag_collection.clone(),
                JobKind::Summary => None,
            },
        };
        let message_id = self.conversation.push(role, "", MessageStatus::Streaming);
        // Sending a message brings the view back to the latest content.
        self.scroll.to_bottom();
        self.generation = Some(Generation {
            request_id,
            message_id,
            kind,
        });
        self.status = Status::Generating;
        Effect::StartCompletion(job)
    }

    /// `/add <file>`: asks the runtime to read the file.
    fn attach(&mut self, path: &str) -> Vec<Effect> {
        if path.is_empty() {
            self.status = Status::Error("usage : /add <fichier>".into());
            return Vec::new();
        }
        if self.is_generating() {
            self.set_input(&format!("/add {path}"));
            self.status =
                Status::Error("attendez la fin de la réponse pour joindre un fichier".into());
            return Vec::new();
        }
        vec![Effect::ReadFile(path.to_owned())]
    }

    /// `/index <folder> [name]`: starts indexing a folder into a collection.
    fn start_index(&mut self, arg: &str) -> Vec<Effect> {
        if let Some(progress) = &self.indexing {
            self.status = Status::Error(format!(
                "indexation de « {} » déjà en cours (Échap pour l'arrêter)",
                progress.collection
            ));
            return Vec::new();
        }
        let (args, types) = match index_arguments(arg) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.status = Status::Error(error);
                return Vec::new();
            }
        };
        let (root, name) = match args.as_slice() {
            [root] => {
                let trimmed = root.trim_end_matches('/');
                (
                    root.clone(),
                    files::file_name(if trimmed.is_empty() { root } else { trimmed }),
                )
            }
            [root, name] => (root.clone(), name.clone()),
            _ => {
                self.status = Status::Error(
                    "usage : /index <dossier|collection> [nom] [--types pdf,md,…]".into(),
                );
                return Vec::new();
            }
        };
        if name.is_empty() || name.contains('/') || matches!(name.as_str(), "~" | "." | "..") {
            self.status = Status::Error("donnez un nom : /index <dossier> <nom>".into());
            return Vec::new();
        }
        self.indexing = Some(IndexProgress {
            collection: name.clone(),
            ..IndexProgress::default()
        });
        // The status bar shows the progress (and Esc to stop it).
        self.status = Status::Ready;
        vec![Effect::StartIndex {
            collection: name,
            root,
            types,
        }]
    }

    fn on_index_event(&mut self, event: IndexEvent) -> Vec<Effect> {
        match event {
            IndexEvent::Scanning { collection } => {
                self.indexing = Some(IndexProgress {
                    collection,
                    ..IndexProgress::default()
                });
                Vec::new()
            }
            IndexEvent::Progress {
                collection,
                done,
                total,
                current,
            } => {
                self.indexing = Some(IndexProgress {
                    collection,
                    done,
                    total,
                    current,
                });
                Vec::new()
            }
            IndexEvent::Finished(report) => {
                self.indexing = None;
                self.stale.retain(|s| s.collection != report.collection);
                self.status = Status::Info(index_summary(&report));
                self.last_index = Some(report);
                self.refresh_collections()
            }
            IndexEvent::Failed { collection, error } => {
                self.indexing = None;
                self.status = Status::Error(format!("indexation de « {collection} » : {error}"));
                self.refresh_collections()
            }
            IndexEvent::Cancelled { collection } => {
                self.indexing = None;
                self.status = Status::Info(format!("indexation de « {collection} » arrêtée"));
                self.refresh_collections()
            }
        }
    }

    /// `/rag [collection|off]`: shows, changes or turns off the collection searched for
    /// each reply. A new name is checked against the collection list first.
    fn choose_rag(&mut self, arg: &str) -> Vec<Effect> {
        match arg {
            "" => {
                self.status = Status::Info(match &self.rag_collection {
                    Some(name) => {
                        format!("réponses à partir de « {name} » (/rag off pour arrêter)")
                    }
                    None => "pas de collection : /rag <collection> (voir /collections)".into(),
                });
                Vec::new()
            }
            "off" | "non" | "aucune" => {
                if self.rag_collection.is_none() {
                    self.status = Status::Info("aucune collection utilisée".into());
                    return Vec::new();
                }
                self.status = Status::Info("réponses sans documents".into());
                self.set_rag(None)
            }
            name => {
                self.pending_rag = Some(name.to_owned());
                vec![Effect::Store(StoreRequest::ListCollections)]
            }
        }
    }

    /// Uses `name` if it is one of `collections`.
    fn apply_rag(&mut self, name: &str, collections: &[CollectionSummary]) -> Vec<Effect> {
        let Some(found) = collections.iter().find(|c| c.name == name) else {
            let known: Vec<&str> = collections.iter().map(|c| c.name.as_str()).collect();
            self.status = Status::Error(if known.is_empty() {
                format!(
                    "collection « {name} » introuvable : indexez d'abord un dossier avec /index"
                )
            } else {
                format!(
                    "collection « {name} » introuvable (collections : {})",
                    known.join(", ")
                )
            });
            return Vec::new();
        };
        let cloud = if self.is_local() {
            ""
        } else {
            " ☁ les extraits seront envoyés au fournisseur"
        };
        self.status = Status::Info(format!(
            "réponses à partir de « {name} » ({} documents){cloud}",
            found.documents
        ));
        self.set_rag(Some(name.to_owned()))
    }

    fn set_rag(&mut self, collection: Option<String>) -> Vec<Effect> {
        if self.rag_collection == collection {
            return Vec::new();
        }
        self.rag_collection = collection.clone();
        self.retrieved = None;
        match &self.conversation_id {
            Some(id) => vec![Effect::Store(StoreRequest::SetRag {
                id: id.clone(),
                collection,
            })],
            None => Vec::new(),
        }
    }

    /// `/forget <collection>`: deletes a collection's index, once confirmed by running the
    /// same command again.
    fn forget(&mut self, name: &str) -> Vec<Effect> {
        if name.is_empty() {
            self.status = Status::Error("usage : /forget <collection>".into());
            return Vec::new();
        }
        if self.indexing.as_ref().is_some_and(|p| p.collection == name) {
            self.status = Status::Error(format!(
                "« {name} » est en cours d'indexation (Échap pour l'arrêter)"
            ));
            return Vec::new();
        }
        let confirm = Confirm::Forget(name.to_owned());
        if self.pending_confirm.as_ref() == Some(&confirm) {
            self.pending_confirm = None;
            return vec![Effect::Store(StoreRequest::DeleteCollection(
                name.to_owned(),
            ))];
        }
        self.pending_confirm = Some(confirm);
        self.set_input(&format!("/forget {name}"));
        self.status = Status::Info(format!(
            "Entrée pour confirmer : l'index de « {name} » sera supprimé (pas vos fichiers)"
        ));
        Vec::new()
    }

    /// Types into the panel: the new title when renaming, otherwise the search.
    fn sidebar_edit(&mut self, edit: impl FnOnce(&mut String)) -> Vec<Effect> {
        let Some(sidebar) = &mut self.sidebar else {
            return Vec::new();
        };
        sidebar.confirm_delete = None;
        if let Some(title) = &mut sidebar.rename {
            edit(title);
            return Vec::new();
        }
        edit(&mut sidebar.filter);
        self.refresh_list()
    }

    /// Asks for the conversation list, searched with the panel's filter.
    fn refresh_list(&self) -> Vec<Effect> {
        match &self.sidebar {
            Some(sidebar) if !sidebar.filter.trim().is_empty() => {
                vec![Effect::Store(StoreRequest::Search(sidebar.filter.clone()))]
            }
            Some(_) => vec![Effect::Store(StoreRequest::List)],
            None => Vec::new(),
        }
    }

    /// Saves the title typed in the panel for the highlighted conversation.
    fn finish_rename(&mut self, title: &str) -> Vec<Effect> {
        let Some(sidebar) = &mut self.sidebar else {
            return Vec::new();
        };
        sidebar.rename = None;
        let title = title.trim();
        let Some(id) = sidebar.selected_item().map(|i| i.id.clone()) else {
            return Vec::new();
        };
        if title.is_empty() {
            self.status = Status::Error("titre vide : conversation non renommée".into());
            return Vec::new();
        }
        sidebar.retitle(&id, title);
        if self.conversation_id.as_ref() == Some(&id) {
            self.conversation_title = Some(title.to_owned());
        }
        self.status = Status::Info(format!("renommée : {title}"));
        vec![Effect::Store(StoreRequest::Rename {
            id,
            title: title.to_owned(),
        })]
    }

    /// `Suppr` in the panel: asks to press it again, then deletes.
    fn sidebar_delete(&mut self) -> Vec<Effect> {
        let Some(sidebar) = &mut self.sidebar else {
            return Vec::new();
        };
        if sidebar.rename.is_some() {
            return Vec::new();
        }
        let Some(item) = sidebar.selected_item().cloned() else {
            return Vec::new();
        };
        if sidebar.confirm_delete.as_ref() == Some(&item.id) {
            sidebar.confirm_delete = None;
            if self.conversation_id.as_ref() == Some(&item.id) {
                // Nothing may be saved into it any more.
                self.generation.take();
            }
            return vec![Effect::Store(StoreRequest::DeleteConversation(item.id))];
        }
        sidebar.confirm_delete = Some(item.id);
        self.status = Status::Info(format!(
            "Suppr à nouveau pour supprimer « {} » (définitif)",
            item.title
        ));
        Vec::new()
    }

    /// `/copy [code [n]]`: copies the last reply, or its code block `n` (default: the last).
    fn copy(&mut self, arg: &str) -> Vec<Effect> {
        let reply = self.conversation.messages().iter().rev().find(|m| {
            m.role == Role::Assistant
                && !m.content.trim().is_empty()
                && m.status != MessageStatus::Streaming
        });
        let Some(reply) = reply else {
            self.status = Status::Error("aucune réponse à copier".into());
            return Vec::new();
        };
        let words: Vec<&str> = arg.split_whitespace().collect();
        let (text, what) = match words.as_slice() {
            [] => (reply.content.clone(), "dernière réponse".to_owned()),
            ["code" | "bloc", rest @ ..] => {
                let blocks = crate::markdown::code_blocks(&reply.content);
                let number = match rest {
                    [] => blocks.len(),
                    [n] => match n.parse::<usize>() {
                        Ok(n) => n,
                        Err(_) => {
                            self.status = Status::Error("usage : /copy code [numéro]".into());
                            return Vec::new();
                        }
                    },
                    _ => {
                        self.status = Status::Error("usage : /copy code [numéro]".into());
                        return Vec::new();
                    }
                };
                match number.checked_sub(1).and_then(|i| blocks.get(i)) {
                    Some(block) => (block.clone(), format!("bloc de code #{number}")),
                    None if blocks.is_empty() => {
                        self.status =
                            Status::Error("la dernière réponse n'a pas de bloc de code".into());
                        return Vec::new();
                    }
                    None => {
                        self.status = Status::Error(format!(
                            "pas de bloc #{number} : la dernière réponse en a {}",
                            blocks.len()
                        ));
                        return Vec::new();
                    }
                }
            }
            _ => {
                self.status = Status::Error("usage : /copy ou /copy code [numéro]".into());
                return Vec::new();
            }
        };
        vec![Effect::Copy { text, what }]
    }

    /// `/rename <title>`: renames the current conversation.
    fn rename_current(&mut self, title: &str) -> Vec<Effect> {
        if title.is_empty() {
            self.status = Status::Error("usage : /rename <titre>".into());
            return Vec::new();
        }
        self.conversation_title = Some(title.to_owned());
        self.status = Status::Info(format!("renommée : {title}"));
        match &self.conversation_id {
            Some(id) => vec![Effect::Store(StoreRequest::Rename {
                id: id.clone(),
                title: title.to_owned(),
            })],
            // Used as the title when the conversation is first saved.
            None => Vec::new(),
        }
    }

    /// `/delete`: deletes the current conversation, once confirmed by running it again.
    fn delete_current(&mut self) -> Vec<Effect> {
        let Some(id) = self.conversation_id.clone() else {
            self.status =
                Status::Info("conversation pas encore enregistrée : rien à supprimer".into());
            return Vec::new();
        };
        let confirm = Confirm::DeleteConversation(id.clone());
        if self.pending_confirm.as_ref() == Some(&confirm) {
            self.pending_confirm = None;
            let mut effects = self.cancel_generation();
            effects.push(Effect::Store(StoreRequest::DeleteConversation(id)));
            return effects;
        }
        self.pending_confirm = Some(confirm);
        self.set_input("/delete");
        self.status = Status::Info(format!(
            "Entrée pour confirmer : « {} » sera supprimée définitivement",
            self.conversation_title
                .as_deref()
                .unwrap_or("cette conversation")
        ));
        Vec::new()
    }

    fn on_collections_checked(&mut self, result: Result<Vec<Staleness>, String>) {
        // A failed check is not worth an error: indexing reports real problems.
        let Ok(checked) = result else {
            return;
        };
        self.stale = checked.into_iter().filter(Staleness::is_stale).collect();
        let first = !self.collections_checked;
        self.collections_checked = true;
        if first && matches!(self.status, Status::Ready) {
            if let [only] = self.stale.as_slice() {
                self.status = Status::Info(format!(
                    "« {} » : {} (/index {} pour mettre à jour)",
                    only.collection,
                    staleness_summary(only),
                    only.collection
                ));
            } else if !self.stale.is_empty() {
                self.status = Status::Info(format!(
                    "{} collections ont changé depuis leur indexation (/collections)",
                    self.stale.len()
                ));
            }
        }
    }

    /// Asks for the collection list when the `/collections` popup is open.
    fn refresh_collections(&self) -> Vec<Effect> {
        if matches!(self.overlay, Some(Overlay::Collections { .. })) {
            vec![Effect::Store(StoreRequest::ListCollections)]
        } else {
            Vec::new()
        }
    }

    /// Adds a file that was read to the conversation.
    fn on_file_read(&mut self, result: Result<Attachment, String>) -> Vec<Effect> {
        let attachment = match result {
            Ok(attachment) => attachment,
            Err(error) => {
                self.status = Status::Error(error);
                return Vec::new();
            }
        };
        if self.is_generating() {
            self.status =
                Status::Error("attendez la fin de la réponse pour joindre un fichier".into());
            return Vec::new();
        }
        if self.conversation_title.is_none() {
            self.conversation_title = Some(title_from(&format!(
                "📎 {}",
                files::file_name(&attachment.source)
            )));
        }
        let estimate = tokens::estimate(&attachment.content);
        let name = files::file_name(&attachment.source);
        let id = self
            .conversation
            .push_attachment(attachment.source, attachment.content);
        self.scroll.to_bottom();
        let full = self
            .context_window()
            .is_some_and(|(window, _)| self.context_usage().tokens > window);
        self.status = if full {
            Status::Error(format!(
                "{name} joint, mais le contexte déborde : /compact ou /clear"
            ))
        } else {
            Status::Info(format!(
                "fichier joint : {name} (≈ {} tokens)",
                tokens::format_count(estimate)
            ))
        };
        self.save(id).into_iter().collect()
    }

    /// `/clear`: keeps the messages on screen but stops sending them.
    fn clear_context(&mut self) -> Vec<Effect> {
        let mut effects = self.cancel_generation();
        let count = self.conversation.context_messages().len();
        if count == 0 {
            self.status = Status::Info("le contexte est déjà vide".into());
            return effects;
        }
        let start = self.conversation.next_id();
        effects.extend(self.move_context_start(start));
        self.status = Status::Info(format!(
            "contexte vidé : {count} message(s) ne sont plus envoyés"
        ));
        effects
    }

    /// `/compact`: asks the model for a summary that will replace the context.
    fn compact(&mut self) -> Vec<Effect> {
        if self.is_generating() {
            self.status = Status::Error("attendez la fin de la réponse pour résumer".into());
            return Vec::new();
        }
        if self.model.is_empty() {
            self.status = Status::Error("aucun modèle choisi (F2 ou /model)".into());
            return Vec::new();
        }
        let turns = self
            .conversation
            .context_messages()
            .iter()
            .filter(|m| matches!(m.role, Role::User | Role::Assistant))
            .count();
        if turns < 2 {
            self.status = Status::Info("rien à résumer".into());
            return Vec::new();
        }
        vec![self.start_job(JobKind::Summary, Role::Summary)]
    }

    /// Moves the context start and records it.
    fn move_context_start(&mut self, start: u64) -> Vec<Effect> {
        self.conversation.set_context_start(start);
        self.measured = None;
        self.retrieved = None;
        match &self.conversation_id {
            Some(id) => vec![Effect::Store(StoreRequest::SetContextStart {
                id: id.clone(),
                start,
            })],
            None => Vec::new(),
        }
    }

    /// Builds the effect that stores a message, allocating the conversation id if needed.
    fn save(&mut self, message_id: MessageId) -> Option<Effect> {
        let message = self
            .conversation
            .messages()
            .iter()
            .find(|m| m.id == message_id)?
            .clone();
        let id = match &self.conversation_id {
            Some(id) => id.clone(),
            None => {
                let id = ConversationId(format!(
                    "{:012x}-{:04}",
                    self.session_seed, self.next_conversation
                ));
                self.next_conversation += 1;
                self.conversation_id = Some(id.clone());
                id
            }
        };
        let conversation = ConversationRecord {
            id,
            title: self.conversation_title.clone().unwrap_or_default(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            rag_collection: self.rag_collection.clone(),
        };
        Some(Effect::Store(StoreRequest::SaveMessage {
            conversation,
            message,
        }))
    }

    /// Stops the running generation, keeping (and saving) its partial reply.
    fn cancel_generation(&mut self) -> Vec<Effect> {
        let Some(generation) = self.generation else {
            return Vec::new();
        };
        let mut effects = vec![Effect::CancelCompletion(generation.request_id)];
        effects.extend(self.finish_generation(MessageStatus::Cancelled));
        self.status = Status::Ready;
        self.restore_queued("résumé annulé");
        effects
    }

    /// Puts back in the input the message an automatic `/compact` was waiting to send.
    fn restore_queued(&mut self, reason: &str) {
        if let Some(text) = self.queued.take() {
            self.set_input(&text);
            self.status = Status::Error(format!(
                "{reason} : message non envoyé (il est dans la zone de saisie)"
            ));
        }
    }

    /// Clears the running generation, sets the final status of its message and saves it.
    fn finish_generation(&mut self, status: MessageStatus) -> Option<Effect> {
        let generation = self.generation.take()?;
        if let Some(message) = self.conversation.get_mut(generation.message_id) {
            message.status = status;
        }
        self.transcript.invalidate(generation.message_id);
        self.save(generation.message_id)
    }

    fn on_llm_event(&mut self, request_id: RequestId, event: LlmEvent) -> Vec<Effect> {
        // Events of a cancelled or older request are stale.
        let Some(generation) = self.generation else {
            return Vec::new();
        };
        if generation.request_id != request_id {
            return Vec::new();
        }
        match event {
            LlmEvent::Token(token) => {
                if let Some(message) = self.conversation.get_mut(generation.message_id) {
                    message.content.push_str(&token);
                }
                self.transcript.invalidate(generation.message_id);
                Vec::new()
            }
            LlmEvent::Retrieved {
                first_number,
                chunks,
            } => {
                if let Some(message) = self.conversation.get_mut(generation.message_id) {
                    message.citations = chunks
                        .iter()
                        .enumerate()
                        .map(|(i, chunk)| Citation {
                            number: first_number + i,
                            path: chunk.source.clone(),
                            location: chunk.location.clone(),
                        })
                        .collect();
                }
                self.transcript.invalidate(generation.message_id);
                self.retrieved = Some(Retrieved {
                    first_number,
                    chunks,
                });
                Vec::new()
            }
            LlmEvent::Usage(usage) => {
                self.request_usage = Usage {
                    input_tokens: usage.input_tokens.or(self.request_usage.input_tokens),
                    output_tokens: usage.output_tokens.or(self.request_usage.output_tokens),
                };
                Vec::new()
            }
            LlmEvent::Done if generation.kind == JobKind::Summary => {
                self.status =
                    Status::Info("historique résumé : il remplace les messages précédents".into());
                let mut effects: Vec<Effect> = self
                    .finish_generation(MessageStatus::Complete)
                    .into_iter()
                    .collect();
                effects.extend(self.move_context_start(generation.message_id.0));
                if let Some(text) = self.queued.take() {
                    effects.extend(self.send_now(&text));
                }
                effects
            }
            LlmEvent::Done => {
                if let Some(input_tokens) = self.request_usage.input_tokens {
                    self.measured = Some(Measured {
                        input_tokens,
                        output_tokens: self.request_usage.output_tokens,
                        messages: self.conversation.messages().len(),
                    });
                }
                self.status = Status::Ready;
                if let Some(message) = self.conversation.get_mut(generation.message_id) {
                    keep_cited(message);
                }
                if self.context_nearly_full() {
                    let percent = self.context_percent().unwrap_or_default();
                    self.status = Status::Info(if self.auto_compact {
                        format!(
                            "contexte rempli à {percent} % : il sera résumé avant votre \
                             prochain message"
                        )
                    } else {
                        format!(
                            "contexte rempli à {percent} % : /compact le résume, /clear \
                             repart de zéro"
                        )
                    });
                }
                let mut effects: Vec<Effect> = self
                    .finish_generation(MessageStatus::Complete)
                    .into_iter()
                    .collect();
                // Local servers often report the window only once the model is loaded.
                effects.extend(self.detect_window());
                effects
            }
            LlmEvent::Error(error) => {
                self.status = Status::Error(error.clone());
                self.restore_queued(&error);
                self.finish_generation(MessageStatus::Failed(error))
                    .into_iter()
                    .collect()
            }
        }
    }

    fn on_storage_event(&mut self, event: StoreEvent) -> Vec<Effect> {
        match event {
            StoreEvent::Listed { conversations, now } => {
                // While searching, only the search results are shown.
                if let Some(sidebar) = &mut self.sidebar
                    && sidebar.filter.trim().is_empty()
                {
                    sidebar.set_items(conversations, now, self.conversation_id.as_ref());
                }
                Vec::new()
            }
            StoreEvent::Loaded(stored) => self.load(stored),
            StoreEvent::Searched {
                query,
                results,
                now,
            } => {
                if let Some(sidebar) = &mut self.sidebar
                    && sidebar.filter == query
                {
                    sidebar.set_results(results, now, self.conversation_id.as_ref());
                }
                Vec::new()
            }
            StoreEvent::ConversationDeleted(id) => {
                self.status = Status::Info("conversation supprimée".into());
                let mut effects = Vec::new();
                if self.conversation_id.as_ref() == Some(&id) {
                    // The panel stays open on the refreshed list.
                    let sidebar = self.sidebar.take();
                    effects.extend(self.new_conversation());
                    self.sidebar = sidebar;
                    self.status = Status::Info("conversation supprimée".into());
                }
                effects.extend(self.refresh_list());
                effects
            }
            StoreEvent::CollectionDeleted { name, found } => {
                if !found {
                    self.status = Status::Error(format!("collection « {name} » introuvable"));
                    return Vec::new();
                }
                self.status = Status::Info(format!(
                    "« {name} » supprimée de l'index (vos fichiers ne sont pas touchés)"
                ));
                self.stale.retain(|s| s.collection != name);
                if self
                    .last_index
                    .as_ref()
                    .is_some_and(|r| r.collection == name)
                {
                    self.last_index = None;
                }
                // The database already cleared it from the stored conversations.
                if self.rag_collection.as_deref() == Some(name.as_str()) {
                    self.rag_collection = None;
                    self.retrieved = None;
                }
                vec![Effect::Store(StoreRequest::ListCollections)]
            }
            StoreEvent::Collections { collections, now } => {
                let effects = match self.pending_rag.take() {
                    Some(name) => self.apply_rag(&name, &collections),
                    None => Vec::new(),
                };
                self.collections = Some((collections, now));
                effects
            }
            StoreEvent::Error(error) => {
                if let Some(sidebar) = &mut self.sidebar
                    && sidebar.items.is_none()
                {
                    sidebar.items = Some(Vec::new());
                }
                self.status = Status::Error(error);
                Vec::new()
            }
        }
    }

    /// Switches to the model highlighted in the popup.
    fn select_model(&mut self) -> Vec<Effect> {
        let Some(choice) = self
            .model_picker()
            .and_then(ModelPicker::selected_choice)
            .cloned()
        else {
            return Vec::new();
        };
        self.overlay = None;
        self.set_selection(&choice.provider, &choice.model)
    }

    /// `/model <arg>`: `model`, `provider` (its default model) or `provider model`.
    fn set_model_from_arg(&mut self, arg: &str) -> Vec<Effect> {
        let find = |word: &str| {
            self.providers
                .iter()
                .find(|p| p.id.eq_ignore_ascii_case(word) || p.label.eq_ignore_ascii_case(word))
                .cloned()
        };
        let (first, rest) = arg.split_once(char::is_whitespace).unwrap_or((arg, ""));
        match (find(first), rest.trim()) {
            (Some(provider), "") => {
                let model = provider.default_model.clone().unwrap_or_default();
                self.set_selection(&provider.id, &model)
            }
            (Some(provider), model) => self.set_selection(&provider.id, model),
            (None, _) => {
                let provider = self.provider.clone();
                self.set_selection(&provider, arg)
            }
        }
    }

    /// Uses `provider` / `model` from the next request on, and records them on the current
    /// conversation.
    fn set_selection(&mut self, provider: &str, model: &str) -> Vec<Effect> {
        if provider == self.provider && model == self.model {
            return Vec::new();
        }
        self.provider = provider.to_owned();
        self.model = model.to_owned();
        if !self.is_generating() {
            self.status = if model.is_empty() {
                Status::Info(format!(
                    "{} : choisissez un modèle (/model)",
                    self.provider_label(provider)
                ))
            } else {
                Status::Info(format!("modèle : {}", self.model_display()))
            };
        }
        let mut effects = match &self.conversation_id {
            Some(id) if !model.is_empty() => vec![Effect::Store(StoreRequest::SetModel {
                id: id.clone(),
                provider: self.provider.clone(),
                model: self.model.clone(),
            })],
            _ => Vec::new(),
        };
        effects.extend(self.detect_window());
        effects
    }

    /// `Provider › model` for display.
    pub fn model_display(&self) -> String {
        let label = self.provider_label(&self.provider);
        if self.model.is_empty() {
            format!("{label} › aucun modèle")
        } else {
            format!("{label} › {}", self.model)
        }
    }

    /// Starts an empty conversation (the current one is already saved).
    fn new_conversation(&mut self) -> Vec<Effect> {
        let effects = self.cancel_generation();
        self.replace_conversation(Conversation::new(), None, None);
        effects
    }

    fn open_selected(&mut self) -> Vec<Effect> {
        let Some(item) = self.sidebar.as_ref().and_then(Sidebar::selected_item) else {
            return Vec::new();
        };
        if Some(&item.id) == self.conversation_id.as_ref() {
            self.sidebar = None; // already displayed
            return Vec::new();
        }
        vec![Effect::Store(StoreRequest::Load(item.id.clone()))]
    }

    fn load(&mut self, stored: StoredConversation) -> Vec<Effect> {
        let effects = self.cancel_generation();
        let summary = stored.summary;
        // Conversations stored before providers existed keep the current provider.
        if !summary.provider.is_empty() {
            self.provider = summary.provider;
        }
        if !summary.model.is_empty() {
            self.model = summary.model;
        }
        self.rag_collection = stored.rag_collection;
        self.replace_conversation(
            Conversation::from_messages(stored.messages, stored.context_start),
            Some(summary.id),
            Some(summary.title),
        );
        let mut effects = effects;
        effects.extend(self.detect_window());
        effects
    }

    fn replace_conversation(
        &mut self,
        conversation: Conversation,
        id: Option<ConversationId>,
        title: Option<String>,
    ) {
        self.conversation = conversation;
        self.measured = None;
        // The collection carries over to a new conversation (`load` sets its own).
        self.retrieved = None;
        self.conversation_id = id;
        self.conversation_title = title;
        // Message ids restart in every conversation: the cache must be dropped.
        self.transcript.clear();
        self.scroll = ScrollState::default();
        self.sidebar = None;
        self.status = Status::Ready;
    }
}

/// Keeps only the citations the reply refers to (`[2]`, `[1, 3]`), when it cites any.
fn keep_cited(message: &mut crate::state::Message) {
    let cited = cited_numbers(&message.content);
    if message.citations.iter().any(|c| cited.contains(&c.number)) {
        message.citations.retain(|c| cited.contains(&c.number));
    }
}

/// Numbers written in square brackets in `text`: `[2]`, `[1, 3]`, `[4][5]`.
fn cited_numbers(text: &str) -> Vec<usize> {
    let mut numbers = Vec::new();
    for part in text.split('[').skip(1) {
        let Some((inside, _)) = part.split_once(']') else {
            continue;
        };
        let parsed: Option<Vec<usize>> = inside
            .split([',', ';'])
            .map(|n| n.trim().parse().ok())
            .collect();
        numbers.extend(parsed.unwrap_or_default());
    }
    numbers
}

/// First line of the first message, shortened.
fn title_from(text: &str) -> String {
    let first = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default();
    if first.chars().count() <= TITLE_MAX_CHARS {
        return first.to_owned();
    }
    let mut title: String = first.chars().take(TITLE_MAX_CHARS - 1).collect();
    title.push('…');
    title
}

/// Builds an empty, styled input editor.
fn new_input() -> TextArea<'static> {
    let mut input = TextArea::default();
    input.set_block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Message "),
    );
    input.set_placeholder_text("Écrivez votre message…");
    input.set_placeholder_style(Style::default().fg(Color::DarkGray));
    input.set_cursor_line_style(Style::default());
    input.set_cursor_style(Style::default().add_modifier(Modifier::REVERSED));
    input.set_wrap_mode(WrapMode::WordOrGlyph);
    input
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::*;
    use crate::{
        llm::{ModelInfo, ProviderModels},
        state::Message,
    };

    fn app() -> App {
        App::new(&Config::default(), false)
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            let key = KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);
            app.update(Action::Edit(key));
        }
    }

    /// Types and submits a message; returns the started job.
    fn send(app: &mut App, text: &str) -> CompletionJob {
        type_text(app, text);
        let effects = app.update(Action::Submit);
        effects
            .into_iter()
            .find_map(|c| match c {
                Effect::StartCompletion(job) => Some(job),
                _ => None,
            })
            .expect("StartCompletion")
    }

    fn llm(app: &mut App, request_id: RequestId, event: LlmEvent) {
        app.update(Action::Llm { request_id, event });
    }

    fn token(app: &mut App, request_id: RequestId, text: &str) {
        llm(app, request_id, LlmEvent::Token(text.to_owned()));
    }

    fn last(app: &App) -> &Message {
        app.conversation.messages().last().expect("a message")
    }

    #[test]
    fn submit_starts_a_completion() {
        let mut app = app();
        let job = send(&mut app, "Bonjour");

        assert_eq!(job.model, "llama3.2");
        assert_eq!(job.system_prompt, Config::default().system_prompt);
        assert_eq!(job.history.len(), 1, "history excludes the placeholder");
        assert_eq!(job.history[0].content, "Bonjour");

        let messages = app.conversation.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].status, MessageStatus::Streaming);
        assert_eq!(app.status, Status::Generating);
        assert_eq!(app.input_text(), "");
    }

    #[test]
    fn tokens_are_appended_then_done_completes() {
        let mut app = app();
        let job = send(&mut app, "Salut");
        token(&mut app, job.request_id, "Bon");
        token(&mut app, job.request_id, "jour");
        assert_eq!(last(&app).content, "Bonjour");
        assert!(app.is_generating());

        llm(&mut app, job.request_id, LlmEvent::Done);
        assert_eq!(last(&app).status, MessageStatus::Complete);
        assert_eq!(app.status, Status::Ready);
        assert!(!app.is_generating());
    }

    #[test]
    fn cancel_keeps_partial_reply_and_ignores_late_tokens() {
        let mut app = app();
        let job = send(&mut app, "Raconte");
        token(&mut app, job.request_id, "Il était");

        let effects = app.update(Action::Cancel);
        assert_eq!(effects[0], Effect::CancelCompletion(job.request_id));
        assert_eq!(last(&app).status, MessageStatus::Cancelled);
        assert_eq!(app.status, Status::Ready);

        token(&mut app, job.request_id, " une fois");
        llm(&mut app, job.request_id, LlmEvent::Done);
        assert_eq!(last(&app).content, "Il était");
        assert_eq!(last(&app).status, MessageStatus::Cancelled);
    }

    #[test]
    fn cancel_without_generation_does_nothing() {
        let mut app = app();
        assert!(app.update(Action::Cancel).is_empty());
    }

    #[test]
    fn error_marks_message_failed_and_sets_status() {
        let mut app = app();
        let job = send(&mut app, "?");
        let error = "Ollama injoignable sur http://localhost:11434/v1".to_owned();
        llm(&mut app, job.request_id, LlmEvent::Error(error.clone()));

        assert_eq!(last(&app).status, MessageStatus::Failed(error.clone()));
        assert_eq!(app.status, Status::Error(error));
        assert!(!app.is_generating());
    }

    #[test]
    fn next_submit_clears_error_and_uses_new_request_id() {
        let mut app = app();
        let first = send(&mut app, "a");
        llm(&mut app, first.request_id, LlmEvent::Error("x".into()));
        let second = send(&mut app, "b");
        assert_ne!(first.request_id, second.request_id);
        assert_eq!(app.status, Status::Generating);
        // The failed empty reply stays visible but the new job still carries it; the prompt
        // builder is responsible for dropping it.
        assert_eq!(second.history.len(), 3);
    }

    #[test]
    fn events_of_an_old_request_are_ignored() {
        let mut app = app();
        let first = send(&mut app, "a");
        app.update(Action::Cancel);
        let second = send(&mut app, "b");
        token(&mut app, first.request_id, "stale");
        token(&mut app, second.request_id, "fresh");
        assert_eq!(last(&app).content, "fresh");
    }

    #[test]
    fn submit_is_ignored_while_generating() {
        let mut app = app();
        send(&mut app, "first");
        type_text(&mut app, "wait");
        assert!(app.update(Action::Submit).is_empty());
        assert_eq!(app.conversation.messages().len(), 2);
        assert_eq!(app.input_text(), "wait");
    }

    #[test]
    fn quit_cancels_the_running_generation() {
        let mut app = app();
        let job = send(&mut app, "a");
        let effects = app.update(Action::Quit);
        assert!(!app.running);
        assert_eq!(effects[0], Effect::CancelCompletion(job.request_id));
        assert!(
            matches!(&effects[1], Effect::Store(StoreRequest::SaveMessage { message, .. })
                if message.status == MessageStatus::Cancelled),
            "partial reply is saved before quitting"
        );
    }

    #[test]
    fn blank_input_is_not_submitted() {
        let mut app = app();
        type_text(&mut app, "   ");
        app.update(Action::InsertNewline);
        assert!(app.update(Action::Submit).is_empty());
        assert!(app.conversation.is_empty());
    }

    #[test]
    fn newline_keeps_multi_line_input() {
        let mut app = app();
        type_text(&mut app, "a");
        app.update(Action::InsertNewline);
        type_text(&mut app, "b");
        assert_eq!(app.input_text(), "a\nb");

        send(&mut app, "");
        assert_eq!(app.conversation.messages()[0].content, "a\nb");
    }

    #[test]
    fn paste_inserts_text_with_line_breaks() {
        let mut app = app();
        app.update(Action::Paste("line 1\r\nline 2".to_owned()));
        assert_eq!(app.input_text(), "line 1\nline 2");
        assert!(app.conversation.is_empty(), "paste must not submit");
    }

    /// App with a 40×14 terminal: 40 − 2 columns of text, 14 − 3 (input) − 1 (status) rows.
    fn sized_app() -> App {
        let mut app = app();
        app.update(Action::Resize {
            width: 40,
            height: 14,
        });
        app
    }

    fn many_lines(n: usize) -> String {
        (1..=n).map(|i| format!("ligne {i}\n\n")).collect()
    }

    #[test]
    fn tokens_are_rendered_on_the_next_tick_only() {
        let mut app = sized_app();
        let job = send(&mut app, "Salut");
        let revision = app.transcript.revision();

        token(&mut app, job.request_id, "Bonjour");
        assert_eq!(app.transcript.revision(), revision, "no work per token");

        app.update(Action::Tick);
        assert_eq!(app.transcript.last_rendered(), 1, "only the reply");
        let visible: Vec<String> = app
            .transcript
            .visible(0, 10)
            .iter()
            .map(|l| l.to_string())
            .collect();
        assert!(visible.contains(&"Bonjour▍".to_owned()));
    }

    #[test]
    fn view_follows_the_streamed_reply() {
        let mut app = sized_app();
        let job = send(&mut app, "Compte");
        token(&mut app, job.request_id, &many_lines(20));
        app.update(Action::Tick);

        let height = usize::from(app.chat_area().height);
        let total = app.transcript.total_lines();
        assert!(total > height);
        assert_eq!(app.scroll_offset(), total - height);
    }

    #[test]
    fn scrolling_up_stops_following_until_back_at_the_bottom() {
        let mut app = sized_app();
        let job = send(&mut app, "Compte");
        token(&mut app, job.request_id, &many_lines(20));
        app.update(Action::Tick);

        app.update(Action::ScrollUp(3));
        let pinned = app.scroll_offset();
        token(&mut app, job.request_id, &many_lines(5));
        app.update(Action::Tick);
        assert_eq!(
            app.scroll_offset(),
            pinned,
            "view stays put while tokens arrive"
        );
        assert!(!app.scroll.is_following());

        app.update(Action::ScrollDown(1000));
        assert!(app.scroll.is_following());
        token(&mut app, job.request_id, "fin");
        app.update(Action::Tick);
        let height = usize::from(app.chat_area().height);
        assert_eq!(app.scroll_offset(), app.transcript.total_lines() - height);
    }

    #[test]
    fn page_keys_move_by_a_screen() {
        let mut app = sized_app();
        let job = send(&mut app, "Compte");
        token(&mut app, job.request_id, &many_lines(30));
        llm(&mut app, job.request_id, LlmEvent::Done);

        let bottom = app.scroll_offset();
        let page = usize::from(app.chat_area().height) - 2;
        app.update(Action::PageUp);
        assert_eq!(app.scroll_offset(), bottom - page);
        app.update(Action::ScrollToTop);
        assert_eq!(app.scroll_offset(), 0);
        app.update(Action::PageDown);
        assert_eq!(app.scroll_offset(), page);
        app.update(Action::ScrollToBottom);
        assert_eq!(app.scroll_offset(), bottom);
    }

    #[test]
    fn sending_a_message_returns_to_the_bottom() {
        let mut app = sized_app();
        let job = send(&mut app, "Compte");
        token(&mut app, job.request_id, &many_lines(30));
        llm(&mut app, job.request_id, LlmEvent::Done);
        app.update(Action::ScrollToTop);

        send(&mut app, "Merci");
        assert!(app.scroll.is_following());
    }

    #[test]
    fn resize_rewraps_the_transcript() {
        let mut app = sized_app();
        let job = send(&mut app, "Salut");
        token(
            &mut app,
            job.request_id,
            "un deux trois quatre cinq six sept huit neuf dix",
        );
        llm(&mut app, job.request_id, LlmEvent::Done);
        let wide = app.transcript.total_lines();

        app.update(Action::Resize {
            width: 16,
            height: 14,
        });
        assert!(app.transcript.total_lines() > wide);
    }

    fn saved(effects: &[Effect]) -> Vec<(ConversationRecord, crate::state::Message)> {
        effects
            .iter()
            .filter_map(|c| match c {
                Effect::Store(StoreRequest::SaveMessage {
                    conversation,
                    message,
                }) => Some((conversation.clone(), message.clone())),
                _ => None,
            })
            .collect()
    }

    fn submit(app: &mut App, text: &str) -> Vec<Effect> {
        type_text(app, text);
        app.update(Action::Submit)
    }

    #[test]
    fn sending_saves_the_user_message_in_a_new_conversation() {
        let mut app = app().with_session_seed(0xabc);
        assert!(app.conversation_id.is_none());
        let effects = submit(&mut app, "Première question\nsuite");

        let saved = saved(&effects);
        assert_eq!(saved.len(), 1, "the streaming placeholder is not saved");
        let (conversation, message) = &saved[0];
        assert_eq!(conversation.id.0, "000000000abc-0000");
        assert_eq!(conversation.title, "Première question");
        assert_eq!(conversation.model, "llama3.2");
        assert_eq!(message.role, Role::User);
        assert_eq!(app.conversation_id.as_ref(), Some(&conversation.id));
    }

    #[test]
    fn finished_replies_are_saved_with_their_status() {
        let mut app = app();
        let job = send(&mut app, "a");
        token(&mut app, job.request_id, "réponse");
        let done = app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Done,
        });
        let (_, message) = &saved(&done)[0];
        assert_eq!(message.content, "réponse");
        assert_eq!(message.status, MessageStatus::Complete);

        let job = send(&mut app, "b");
        let failed = app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Error("boom".into()),
        });
        assert_eq!(
            saved(&failed)[0].1.status,
            MessageStatus::Failed("boom".into())
        );

        send(&mut app, "c");
        let cancelled = app.update(Action::Cancel);
        assert_eq!(saved(&cancelled)[0].1.status, MessageStatus::Cancelled);
    }

    #[test]
    fn a_conversation_keeps_its_id_and_first_title() {
        let mut app = app();
        let first = saved(&submit(&mut app, "Titre"))[0].0.clone();
        app.update(Action::Cancel);
        let second = saved(&submit(&mut app, "Autre chose"))[0].0.clone();
        assert_eq!(first.id, second.id);
        assert_eq!(second.title, "Titre");
    }

    #[test]
    fn long_titles_are_shortened() {
        let title = title_from(&"mot ".repeat(40));
        assert_eq!(title.chars().count(), TITLE_MAX_CHARS);
        assert!(title.ends_with('…'));
        assert_eq!(title_from("\n\n  Bonjour  \n"), "Bonjour");
    }

    #[test]
    fn new_conversation_stops_generation_and_starts_empty() {
        let mut app = sized_app();
        let job = send(&mut app, "a");
        token(&mut app, job.request_id, "partiel");
        let first_id = app.conversation_id.clone();

        let effects = app.update(Action::NewConversation);
        assert_eq!(effects[0], Effect::CancelCompletion(job.request_id));
        assert_eq!(saved(&effects)[0].1.content, "partiel");
        assert!(app.conversation.is_empty());
        assert!(app.conversation_id.is_none());
        assert_eq!(app.transcript.total_lines(), 0);

        let next = saved(&submit(&mut app, "b"))[0].0.id.clone();
        assert_ne!(Some(next), first_id, "a fresh id");
    }

    #[test]
    fn sidebar_opens_with_a_list_request_and_escape_closes_it_first() {
        let mut app = app();
        let job = send(&mut app, "a");
        assert_eq!(
            app.update(Action::ToggleSidebar),
            vec![Effect::Store(StoreRequest::List)]
        );
        assert!(app.key_context().sidebar_open);

        assert!(app.update(Action::Cancel).is_empty());
        assert!(app.sidebar.is_none());
        assert!(
            app.is_generating(),
            "Esc closed the panel, not the generation"
        );

        app.update(Action::ToggleSidebar);
        app.update(Action::ToggleSidebar);
        assert!(app.sidebar.is_none());
        llm(&mut app, job.request_id, LlmEvent::Done);
    }

    fn listed(app: &mut App, ids: &[&str]) {
        let conversations = ids
            .iter()
            .map(|id| crate::storage::ConversationSummary {
                id: ConversationId((*id).into()),
                title: (*id).into(),
                provider: "ollama".into(),
                model: "qwen2.5".into(),
                updated_at: 0,
            })
            .collect();
        app.update(Action::Storage(StoreEvent::Listed {
            conversations,
            now: 0,
        }));
    }

    #[test]
    fn opening_another_conversation_requests_it() {
        let mut app = app();
        app.update(Action::ToggleSidebar);
        listed(&mut app, &["x", "y"]);
        app.update(Action::SidebarDown);
        assert_eq!(
            app.update(Action::SidebarOpen),
            vec![Effect::Store(StoreRequest::Load(ConversationId(
                "y".into()
            )))]
        );
    }

    #[test]
    fn opening_the_current_conversation_just_closes_the_panel() {
        let mut app = app();
        send(&mut app, "a");
        let current = app.conversation_id.clone().expect("id");
        app.update(Action::ToggleSidebar);
        listed(&mut app, &["other", &current.0]);
        assert_eq!(app.sidebar.as_ref().map(|s| s.selected), Some(1));
        assert!(app.update(Action::SidebarOpen).is_empty());
        assert!(app.sidebar.is_none());
    }

    #[test]
    fn loaded_conversation_replaces_the_current_one() {
        let mut app = sized_app();
        send(&mut app, "ancienne");
        app.update(Action::ToggleSidebar);

        let mut stored = Conversation::new();
        stored.push(Role::User, "Question", MessageStatus::Complete);
        stored.push(Role::Assistant, "Réponse", MessageStatus::Complete);
        let effects = app.update(Action::Storage(StoreEvent::Loaded(StoredConversation {
            summary: crate::storage::ConversationSummary {
                id: ConversationId("old".into()),
                title: "Question".into(),
                provider: "claude".into(),
                model: "qwen2.5".into(),
                updated_at: 0,
            },
            messages: stored.messages().to_vec(),
            context_start: 0,
            rag_collection: None,
        })));

        assert!(
            matches!(effects[0], Effect::CancelCompletion(_)),
            "stops the stream"
        );
        assert_eq!(app.conversation.messages(), stored.messages());
        assert_eq!(app.conversation_id, Some(ConversationId("old".into())));
        assert_eq!(app.model, "qwen2.5", "the conversation's model is restored");
        assert!(app.sidebar.is_none());
        assert_eq!(app.transcript.total_lines(), 6, "re-rendered from scratch");

        // New messages continue the numbering and the stored conversation.
        let saved = saved(&submit(&mut app, "Suite"));
        assert_eq!(saved[0].0.id, ConversationId("old".into()));
        assert_eq!(saved[0].1.id, MessageId(2));
    }

    #[test]
    fn storage_errors_are_shown() {
        let mut app = app();
        app.update(Action::ToggleSidebar);
        app.update(Action::Storage(StoreEvent::Error(
            "historique : disque plein".into(),
        )));
        assert_eq!(
            app.status,
            Status::Error("historique : disque plein".into())
        );
        assert_eq!(
            app.sidebar.as_ref().and_then(|s| s.items.clone()),
            Some(Vec::new())
        );
    }

    /// Only the `SetModel` storage effects.
    fn saved_models(effects: &[Effect]) -> Vec<Effect> {
        effects
            .iter()
            .filter(|e| matches!(e, Effect::Store(StoreRequest::SetModel { .. })))
            .cloned()
            .collect()
    }

    fn open_picker(app: &mut App, models: &[&str]) {
        assert_eq!(
            app.update(Action::OpenModelPicker),
            vec![Effect::ListModels]
        );
        app.update(Action::ModelsListed(vec![ProviderModels {
            provider: "ollama".into(),
            result: Ok(models.iter().map(|m| ModelInfo::named(*m)).collect()),
        }]));
    }

    #[test]
    fn picking_a_model_for_a_new_conversation() {
        let mut app = app();
        open_picker(&mut app, &["llama3.2", "mistral"]);
        assert_eq!(
            app.key_context().overlay,
            Some(crate::state::OverlayKind::List)
        );
        app.update(Action::OverlayDown);
        assert!(
            saved_models(&app.update(Action::OverlaySelect)).is_empty(),
            "nothing stored yet"
        );
        assert_eq!(app.model, "mistral");

        let job = send(&mut app, "a");
        assert_eq!(job.model, "mistral");
    }

    #[test]
    fn picking_a_model_updates_the_stored_conversation() {
        let mut app = app();
        send(&mut app, "a");
        app.update(Action::Cancel);
        let id = app.conversation_id.clone().expect("stored");
        open_picker(&mut app, &["llama3.2", "mistral"]);
        app.update(Action::OverlayFilter('m'));
        app.update(Action::OverlayFilter('i'));
        assert_eq!(
            saved_models(&app.update(Action::OverlaySelect)),
            vec![Effect::Store(StoreRequest::SetModel {
                id,
                provider: "ollama".into(),
                model: "mistral".into()
            })]
        );
    }

    #[test]
    fn picking_the_current_model_changes_nothing() {
        let mut app = app();
        send(&mut app, "a");
        open_picker(&mut app, &["llama3.2", "mistral"]);
        assert!(app.update(Action::OverlaySelect).is_empty());
        assert!(app.model_picker().is_none());
    }

    #[test]
    fn escape_closes_the_picker_before_the_sidebar() {
        let mut app = app();
        app.update(Action::ToggleSidebar);
        open_picker(&mut app, &["x"]);
        app.update(Action::Cancel);
        assert!(app.model_picker().is_none());
        assert!(app.sidebar.is_some());
        app.update(Action::Cancel);
        assert!(app.sidebar.is_none());
    }

    #[test]
    fn model_list_errors_are_shown_in_the_popup() {
        let mut app = app();
        app.update(Action::OpenModelPicker);
        app.update(Action::ModelsListed(vec![ProviderModels {
            provider: "ollama".into(),
            result: Err("Ollama injoignable".into()),
        }]));
        assert_eq!(
            app.model_picker().map(|p| p.errors().to_vec()),
            Some(vec![("Ollama".to_owned(), "Ollama injoignable".to_owned())])
        );
        assert!(app.update(Action::OverlaySelect).is_empty());
        assert_eq!(app.model, "llama3.2");
    }

    #[test]
    fn late_model_list_is_ignored_once_closed() {
        let mut app = app();
        app.update(Action::OpenModelPicker);
        app.update(Action::OpenModelPicker);
        app.update(Action::ModelsListed(Vec::new()));
        assert!(app.model_picker().is_none());
    }

    #[test]
    fn custom_system_prompt_is_sent() {
        let config = Config {
            system_prompt: "Réponds en français.".into(),
            ..Config::default()
        };
        let mut app = App::new(&config, false);
        assert_eq!(send(&mut app, "hi").system_prompt, "Réponds en français.");
    }

    #[test]
    fn slash_new_starts_a_new_conversation() {
        let mut app = app();
        send(&mut app, "a");
        let effects = submit(&mut app, "/new");
        assert!(matches!(effects[0], Effect::CancelCompletion(_)));
        assert!(app.conversation.is_empty());
        assert_eq!(app.input_text(), "", "the command is consumed");
    }

    #[test]
    fn slash_commands_open_panels_and_popups() {
        let mut app = app();
        assert_eq!(
            submit(&mut app, "/historique"),
            vec![Effect::Store(StoreRequest::List)]
        );
        assert!(app.sidebar.is_some());
        app.update(Action::Cancel);

        assert_eq!(submit(&mut app, "/model"), vec![Effect::ListModels]);
        assert!(app.model_picker().is_some());
        app.update(Action::Cancel);

        assert!(submit(&mut app, "/help").is_empty());
        assert_eq!(app.overlay, Some(Overlay::Help { scroll: 0 }));
    }

    #[test]
    fn slash_model_with_a_name_switches_directly() {
        let mut app = app();
        assert!(saved_models(&submit(&mut app, "/model mistral:7b")).is_empty());
        assert_eq!(app.model, "mistral:7b");
        assert_eq!(
            app.status,
            Status::Info("modèle : Ollama › mistral:7b".into())
        );
        assert_eq!(send(&mut app, "x").model, "mistral:7b");
    }

    #[test]
    fn slash_quit_quits() {
        let mut app = app();
        submit(&mut app, "/quit");
        assert!(!app.running);
    }

    #[test]
    fn commands_work_while_generating() {
        let mut app = app();
        send(&mut app, "long");
        submit(&mut app, "/model qwen");
        assert_eq!(app.model, "qwen");
        assert!(app.is_generating(), "the running reply is not affected");
    }

    #[test]
    fn unknown_command_is_reported_and_kept() {
        let mut app = app();
        assert!(submit(&mut app, "/zzz").is_empty());
        assert_eq!(
            app.status,
            Status::Error("commande inconnue : /zzz (tapez /help)".into())
        );
        assert_eq!(app.input_text(), "/zzz");
        assert!(app.conversation.is_empty());
    }

    #[test]
    fn double_slash_sends_a_message_starting_with_a_slash() {
        let mut app = app();
        let job = send(&mut app, "//etc/hosts, c'est quoi ?");
        assert_eq!(job.history[0].content, "/etc/hosts, c'est quoi ?");
    }

    #[test]
    fn multi_line_input_starting_with_a_slash_is_a_message() {
        let mut app = app();
        type_text(&mut app, "/new");
        app.update(Action::InsertNewline);
        type_text(&mut app, "suite");
        let effects = app.update(Action::Submit);
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::StartCompletion(_)))
        );
    }

    #[test]
    fn suggestions_follow_the_input_and_run_on_enter() {
        let mut app = app();
        type_text(&mut app, "/h");
        let names: Vec<&str> = app.suggestions().iter().map(|c| c.name).collect();
        assert_eq!(names, ["history", "help"]);
        assert!(app.key_context().suggestions_open);

        app.update(Action::SuggestionDown);
        assert_eq!(
            app.selected_suggestion().map(|c| c.id),
            Some(CommandId::Help)
        );
        app.update(Action::Submit);
        assert_eq!(
            app.overlay,
            Some(Overlay::Help { scroll: 0 }),
            "runs the highlighted one"
        );
    }

    #[test]
    fn tab_completes_and_adds_a_space_before_arguments() {
        let mut app = app();
        type_text(&mut app, "/mo");
        app.update(Action::CompleteSuggestion);
        assert_eq!(app.input_text(), "/model ");
        assert!(app.suggestions().is_empty(), "argument expected now");

        let mut app = self::app();
        type_text(&mut app, "/qu");
        app.update(Action::CompleteSuggestion);
        assert_eq!(app.input_text(), "/quit");
    }

    #[test]
    fn escape_hides_suggestions_until_the_input_changes() {
        let mut app = app();
        type_text(&mut app, "/");
        app.update(Action::DismissSuggestions);
        assert!(app.suggestions().is_empty());
        type_text(&mut app, "n");
        assert_eq!(app.suggestions().len(), 1);
    }

    #[test]
    fn palette_runs_the_selected_command() {
        let mut app = app();
        app.update(Action::OpenPalette);
        assert_eq!(
            app.key_context().overlay,
            Some(crate::state::OverlayKind::List)
        );
        for c in "hist".chars() {
            app.update(Action::OverlayFilter(c));
        }
        assert_eq!(
            app.update(Action::OverlaySelect),
            vec![Effect::Store(StoreRequest::List)]
        );
        assert!(app.overlay.is_none());
        assert!(app.sidebar.is_some());
    }

    #[test]
    fn palette_and_help_toggle_with_their_shortcut() {
        let mut app = app();
        app.update(Action::OpenPalette);
        app.update(Action::OpenPalette);
        assert!(app.overlay.is_none());
        app.update(Action::OpenHelp);
        app.update(Action::OverlayDown);
        assert_eq!(app.overlay, Some(Overlay::Help { scroll: 1 }));
        app.update(Action::OpenHelp);
        assert!(app.overlay.is_none());
    }

    #[test]
    fn slash_model_can_switch_provider() {
        let mut app = app();
        submit(&mut app, "/model claude claude-test");
        assert_eq!(
            (app.provider.as_str(), app.model.as_str()),
            ("claude", "claude-test")
        );
        assert!(!app.is_local());
        assert_eq!(app.model_display(), "Claude › claude-test");

        // A provider alone switches to its default model (none for Claude here).
        submit(&mut app, "/model Ollama");
        assert_eq!(
            (app.provider.as_str(), app.model.as_str()),
            ("ollama", "llama3.2")
        );
        submit(&mut app, "/model openai");
        assert_eq!(app.model, "");
        assert_eq!(
            app.status,
            Status::Info("OpenAI : choisissez un modèle (/model)".into())
        );
    }

    #[test]
    fn sending_without_a_model_is_refused() {
        let mut app = app();
        submit(&mut app, "/model openai");
        assert!(submit(&mut app, "Bonjour").is_empty());
        assert_eq!(
            app.status,
            Status::Error("OpenAI : aucun modèle choisi (F2 ou /model)".into())
        );
        assert_eq!(app.input_text(), "Bonjour", "the message is kept");
    }

    #[test]
    fn jobs_and_saves_carry_the_provider() {
        let mut app = app();
        submit(&mut app, "/model claude claude-test");
        type_text(&mut app, "Salut");
        let effects = app.update(Action::Submit);
        let (record, _) = &saved(&effects)[0];
        assert_eq!(record.provider, "claude");
        let job = effects
            .iter()
            .find_map(|e| match e {
                Effect::StartCompletion(job) => Some(job),
                _ => None,
            })
            .expect("started");
        assert_eq!(
            (job.provider.as_str(), job.model.as_str()),
            ("claude", "claude-test")
        );
    }

    fn usage(input: u64, output: u64) -> LlmEvent {
        LlmEvent::Usage(Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
        })
    }

    #[test]
    fn context_usage_is_estimated_until_measured() {
        let mut app = app();
        let before = app.context_usage();
        assert!(!before.measured);
        assert!(before.tokens > 0, "the system prompt counts");

        let job = send(&mut app, "Bonjour");
        token(&mut app, job.request_id, "Salut !");
        llm(&mut app, job.request_id, usage(40, 5));
        llm(&mut app, job.request_id, LlmEvent::Done);
        assert_eq!(
            app.context_usage(),
            ContextUsage {
                tokens: 45,
                measured: true
            }
        );
        assert_eq!(app.measured.map(|m| m.messages), Some(2));
    }

    #[test]
    fn usage_parts_are_merged_across_events() {
        // Anthropic reports input tokens first and output tokens at the end.
        let mut app = app();
        let job = send(&mut app, "a");
        llm(
            &mut app,
            job.request_id,
            LlmEvent::Usage(Usage {
                input_tokens: Some(100),
                output_tokens: None,
            }),
        );
        llm(
            &mut app,
            job.request_id,
            LlmEvent::Usage(Usage {
                input_tokens: None,
                output_tokens: Some(20),
            }),
        );
        llm(&mut app, job.request_id, LlmEvent::Done);
        assert_eq!(app.context_usage().tokens, 120);
    }

    #[test]
    fn messages_after_the_measure_are_estimated_on_top() {
        let mut app = app();
        let job = send(&mut app, "a");
        llm(&mut app, job.request_id, usage(40, 5));
        llm(&mut app, job.request_id, LlmEvent::Done);
        send(&mut app, &"x".repeat(400));
        let used = app.context_usage();
        assert!(!used.measured);
        // 45 measured + ~100 for the new message + framing of it and the empty reply.
        assert_eq!(used.tokens, 45 + 100 + 4 + 4);
    }

    #[test]
    fn a_new_conversation_forgets_the_measure() {
        let mut app = app();
        let job = send(&mut app, "a");
        llm(&mut app, job.request_id, usage(40, 5));
        llm(&mut app, job.request_id, LlmEvent::Done);
        app.update(Action::NewConversation);
        assert!(app.measured.is_none());
        assert!(!app.context_usage().measured);
    }

    #[test]
    fn context_window_comes_from_config_or_server() {
        let mut app = app();
        assert!(app.context_window().is_none());
        assert_eq!(
            app.update(Action::Init),
            vec![
                Effect::DetectContextWindow {
                    provider: "ollama".into(),
                    model: "llama3.2".into()
                },
                Effect::CheckCollections
            ]
        );
        app.update(Action::ContextWindowDetected {
            provider: "ollama".into(),
            model: "llama3.2".into(),
            tokens: Some(4_096),
        });
        assert_eq!(app.context_window(), Some((4_096, WindowSource::Server)));
        assert_eq!(
            app.update(Action::Init),
            vec![Effect::CheckCollections],
            "window already known"
        );

        let mut config = Config::default();
        if let Some(ollama) = config.providers.get_mut("ollama") {
            ollama.context_window = Some(32_768);
        }
        let app = App::new(&config, false);
        assert_eq!(app.context_window(), Some((32_768, WindowSource::Config)));
    }

    #[test]
    fn unknown_window_is_asked_again_after_a_reply() {
        let mut app = app();
        let job = send(&mut app, "a");
        let effects = app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Done,
        });
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::DetectContextWindow { .. }))
        );
    }

    #[test]
    fn model_lists_teach_context_windows() {
        let mut app = app();
        app.update(Action::OpenModelPicker);
        app.update(Action::ModelsListed(vec![ProviderModels {
            provider: "claude".into(),
            result: Ok(vec![ModelInfo {
                id: "claude-test".into(),
                context_window: Some(200_000),
            }]),
        }]));
        app.update(Action::OverlaySelect);
        assert_eq!(app.context_window(), Some((200_000, WindowSource::Server)));
    }

    #[test]
    fn context_and_prompt_popups() {
        let mut app = sized_app();
        submit(&mut app, "/context");
        assert_eq!(app.overlay, Some(Overlay::Context { scroll: 0 }));
        app.update(Action::Cancel);
        submit(&mut app, "/prompt");
        assert_eq!(app.overlay, Some(Overlay::Prompt { scroll: 0 }));
    }

    #[test]
    fn popup_scroll_stays_within_the_content() {
        let mut app = sized_app();
        for i in 0..30 {
            let job = send(&mut app, &format!("question {i}"));
            llm(&mut app, job.request_id, LlmEvent::Done);
        }
        submit(&mut app, "/prompt");
        app.update(Action::OverlayUp);
        assert_eq!(app.overlay, Some(Overlay::Prompt { scroll: 0 }));
        for _ in 0..1_000 {
            app.update(Action::OverlayPageDown);
        }
        let max = crate::ui::popup_max_scroll(&app);
        assert!(max > 0);
        assert_eq!(app.overlay, Some(Overlay::Prompt { scroll: max }));
        app.update(Action::OverlayUp);
        assert_eq!(app.overlay, Some(Overlay::Prompt { scroll: max - 1 }));
    }

    #[test]
    fn add_needs_a_path_and_waits_for_the_reply() {
        let mut app = app();
        assert!(submit(&mut app, "/add").is_empty());
        assert_eq!(app.status, Status::Error("usage : /add <fichier>".into()));

        assert_eq!(
            submit(&mut app, "/add notes.md"),
            vec![Effect::ReadFile("notes.md".into())]
        );
        send(&mut app, "question");
        assert!(submit(&mut app, "/add notes.md").is_empty());
        assert_eq!(app.input_text(), "/add notes.md", "kept for later");
    }

    #[test]
    fn attaching_first_titles_the_conversation_and_saves_it() {
        let mut app = app();
        let effects = app.update(Action::FileRead(Ok(Attachment {
            source: "~/docs/plan.md".into(),
            content: "contenu".into(),
        })));
        let saved = saved(&effects);
        assert_eq!(saved[0].0.title, "📎 plan.md");
        assert_eq!(saved[0].1.role, Role::Attachment);
        assert_eq!(saved[0].1.source.as_deref(), Some("~/docs/plan.md"));
    }

    #[test]
    fn clear_and_compact_need_something_to_work_on() {
        let mut app = app();
        assert!(submit(&mut app, "/clear").is_empty());
        assert_eq!(app.status, Status::Info("le contexte est déjà vide".into()));
        assert!(submit(&mut app, "/compact").is_empty());
        assert_eq!(app.status, Status::Info("rien à résumer".into()));
    }

    #[test]
    fn clear_moves_the_boundary_and_forgets_the_measure() {
        let mut app = app();
        let job = send(&mut app, "a");
        llm(&mut app, job.request_id, usage(40, 5));
        llm(&mut app, job.request_id, LlmEvent::Done);
        let id = app.conversation_id.clone().expect("stored");

        let effects = submit(&mut app, "/vider");
        assert_eq!(
            effects,
            vec![Effect::Store(StoreRequest::SetContextStart {
                id,
                start: 2
            })]
        );
        assert!(app.measured.is_none());
        assert!(app.conversation.context_messages().is_empty());
        assert_eq!(app.prompt().len(), 1, "only the system prompt remains");
    }

    #[test]
    fn failed_summary_keeps_the_context() {
        let mut app = app();
        let job = send(&mut app, "a");
        token(&mut app, job.request_id, "b");
        llm(&mut app, job.request_id, LlmEvent::Done);
        let effects = submit(&mut app, "/compact");
        let Some(Effect::StartCompletion(summary_job)) = effects.first() else {
            panic!("summary started");
        };
        assert_eq!(summary_job.kind, JobKind::Summary);
        assert_eq!(
            summary_job.history.len(),
            2,
            "the summary placeholder is not included"
        );
        llm(
            &mut app,
            summary_job.request_id,
            LlmEvent::Error("boom".into()),
        );
        assert_eq!(app.conversation.context_start(), 0);
        assert_eq!(last(&app).role, Role::Summary);
        assert_eq!(app.prompt().len(), 3, "system + a + reply");
    }

    #[test]
    fn path_argument_covers_add_and_the_folder_of_index() {
        assert_eq!(path_argument("/add ~/a.md"), Some(("/add ", "~/a.md")));
        assert_eq!(
            path_argument("/index ~/cours"),
            Some(("/index ", "~/cours"))
        );
        assert_eq!(path_argument("/index ~/cours rust"), None);
        assert_eq!(path_argument("/model x"), None);
    }

    #[test]
    fn index_usage_errors() {
        let mut app = App::new(&Config::default(), false);
        assert!(app.run_command(CommandId::Index, "").is_empty());
        assert_eq!(
            app.status,
            Status::Error("usage : /index <dossier|collection> [nom] [--types pdf,md,…]".into())
        );
        assert!(app.run_command(CommandId::Index, "a b c").is_empty());
        assert!(app.run_command(CommandId::Index, "/").is_empty());
        assert!(app.indexing.is_none());
        let effects = app.run_command(CommandId::Index, "\"~/Mes cours/\" rust");
        assert_eq!(
            effects,
            vec![Effect::StartIndex {
                collection: "rust".into(),
                root: "~/Mes cours/".into(),
                types: None
            }]
        );
    }

    fn collection(name: &str) -> CollectionSummary {
        CollectionSummary {
            name: name.into(),
            root: format!("/docs/{name}"),
            embedding_model: "bge-m3".into(),
            documents: 3,
            chunks: 12,
            updated_at: 0,
            types: Vec::new(),
        }
    }

    fn collections_listed(app: &mut App, names: &[&str]) -> Vec<Effect> {
        app.update(Action::Storage(StoreEvent::Collections {
            collections: names.iter().map(|n| collection(n)).collect(),
            now: 0,
        }))
    }

    #[test]
    fn rag_is_checked_against_the_collections_then_sent_with_each_reply() {
        let mut app = app();
        assert_eq!(
            app.run_command(CommandId::Rag, "cours"),
            vec![Effect::Store(StoreRequest::ListCollections)]
        );
        assert_eq!(app.rag_collection, None, "not applied before the check");
        collections_listed(&mut app, &["autre"]);
        assert_eq!(
            app.status,
            Status::Error("collection « cours » introuvable (collections : autre)".into())
        );
        assert_eq!(app.rag_collection, None);

        app.run_command(CommandId::Rag, "cours");
        assert!(
            collections_listed(&mut app, &["autre", "cours"]).is_empty(),
            "no id yet"
        );
        assert_eq!(app.rag_collection.as_deref(), Some("cours"));
        assert!(
            matches!(&app.status, Status::Info(m) if m.starts_with("réponses à partir de « cours » (3 documents)"))
        );

        let job = send(&mut app, "Question");
        assert_eq!(job.rag_collection.as_deref(), Some("cours"));
        // Once stored, a change is saved right away.
        llm(&mut app, job.request_id, LlmEvent::Done);
        let effects = app.run_command(CommandId::Rag, "off");
        assert!(matches!(
            effects.as_slice(),
            [Effect::Store(StoreRequest::SetRag {
                collection: None,
                ..
            })]
        ));
        assert_eq!(send(&mut app, "Encore").rag_collection, None);
    }

    #[test]
    fn retrieved_passages_become_the_citations_of_the_reply() {
        let mut app = app();
        app.rag_collection = Some("cours".into());
        let job = send(&mut app, "Question");
        let chunk = |source: &str, location: &str| ContextChunk {
            source: source.into(),
            location: location.into(),
            text: "texte".into(),
        };
        llm(
            &mut app,
            job.request_id,
            LlmEvent::Retrieved {
                first_number: 2,
                chunks: vec![
                    chunk("a.md", "§ A"),
                    chunk("b.pdf", "p. 3"),
                    chunk("c.rs", ""),
                ],
            },
        );
        assert_eq!(last(&app).citations.len(), 3);
        assert_eq!(app.retrieved.as_ref().map(|r| r.first_number), Some(2));
        token(&mut app, job.request_id, "D'après [3] et [2, 9], oui.");
        llm(&mut app, job.request_id, LlmEvent::Done);
        let numbers: Vec<usize> = last(&app).citations.iter().map(|c| c.number).collect();
        assert_eq!(numbers, vec![2, 3], "only the cited passages are kept");
        assert_eq!(last(&app).citations[1].label(), "b.pdf p. 3");
        // The next prompt shows the same passages, numbered after the attachments.
        assert!(app.prompt()[0].content.contains("[1] a.md § A"));
    }

    #[test]
    fn uncited_replies_keep_every_passage() {
        assert_eq!(
            cited_numbers("voir [1], [2,3] et [x] ou [ 4 ]"),
            vec![1, 2, 3, 4]
        );
        let mut message = Conversation::new();
        message.push(
            Role::Assistant,
            "Réponse sans renvoi.",
            MessageStatus::Complete,
        );
        let mut message = message.messages()[0].clone();
        message.citations = vec![Citation {
            number: 1,
            path: "a.md".into(),
            location: String::new(),
        }];
        keep_cited(&mut message);
        assert_eq!(message.citations.len(), 1);
    }

    #[test]
    fn loading_a_conversation_restores_its_collection() {
        let mut app = app();
        app.rag_collection = Some("cours".into());
        app.update(Action::Storage(StoreEvent::Loaded(StoredConversation {
            summary: crate::storage::ConversationSummary {
                id: ConversationId("old".into()),
                title: "t".into(),
                provider: "ollama".into(),
                model: "llama3.2".into(),
                updated_at: 0,
            },
            messages: Vec::new(),
            context_start: 0,
            rag_collection: Some("rust".into()),
        })));
        assert_eq!(app.rag_collection.as_deref(), Some("rust"));
        app.update(Action::NewConversation);
        assert_eq!(
            app.rag_collection.as_deref(),
            Some("rust"),
            "kept for a new conversation"
        );
    }

    #[test]
    fn index_types_option() {
        assert_eq!(
            index_arguments("~/cours rust --types pdf,.MD,word"),
            Ok((
                vec!["~/cours".to_owned(), "rust".to_owned()],
                Some(vec!["pdf".to_owned(), "md".to_owned(), "docx".to_owned()])
            ))
        );
        assert_eq!(
            index_arguments("rust --types=all"),
            Ok((vec!["rust".to_owned()], Some(Vec::new())))
        );
        assert_eq!(index_arguments("~/c").map(|a| a.1), Ok(None));
        assert!(index_arguments("~/c --types").is_err());
        assert!(index_arguments("~/c --types exe").is_err());
        assert!(index_arguments("~/c --force").is_err());

        let mut app = app();
        let effects = app.run_command(CommandId::Index, "~/cours --types code");
        assert_eq!(
            effects,
            vec![Effect::StartIndex {
                collection: "cours".into(),
                root: "~/cours".into(),
                types: Some(vec!["code".into()])
            }]
        );
    }

    #[test]
    fn forget_asks_for_confirmation_then_turns_rag_off() {
        let mut app = app();
        app.rag_collection = Some("cours".into());
        assert!(app.run_command(CommandId::Forget, "cours").is_empty());
        assert_eq!(
            app.input_text(),
            "/forget cours",
            "ready to confirm with Enter"
        );
        // Anything else in between cancels the confirmation.
        app.run_command(CommandId::Help, "");
        assert!(app.run_command(CommandId::Forget, "cours").is_empty());
        assert_eq!(
            app.update(Action::Submit),
            vec![Effect::Store(StoreRequest::DeleteCollection(
                "cours".into()
            ))]
        );
        let effects = app.update(Action::Storage(StoreEvent::CollectionDeleted {
            name: "cours".into(),
            found: true,
        }));
        assert_eq!(effects, vec![Effect::Store(StoreRequest::ListCollections)]);
        assert_eq!(app.rag_collection, None);
        assert!(
            matches!(&app.status, Status::Info(m) if m.contains("vos fichiers ne sont pas touchés"))
        );
    }

    #[test]
    fn changed_collections_are_announced_once_at_startup() {
        let stale = |name: &str, modified: usize| Staleness {
            collection: name.into(),
            modified,
            added: 1,
            ..Staleness::default()
        };
        let mut app = app();
        app.update(Action::CollectionsChecked(Ok(vec![
            stale("cours", 2),
            Staleness {
                collection: "ok".into(),
                ..Staleness::default()
            },
        ])));
        assert_eq!(
            app.status,
            Status::Info(
                "« cours » : 1 nouveau fichier, 2 fichiers modifiés (/index cours pour mettre à jour)"
                    .into()
            )
        );
        assert_eq!(app.stale.len(), 1, "up-to-date collections are not listed");

        app.status = Status::Ready;
        app.update(Action::CollectionsChecked(Ok(vec![
            stale("cours", 1),
            stale("tp", 1),
        ])));
        assert_eq!(
            app.status,
            Status::Ready,
            "only the first check is announced"
        );
        assert_eq!(app.stale.len(), 2);

        app.update(Action::Index(IndexEvent::Finished(IndexReport {
            collection: "cours".into(),
            ..IndexReport::default()
        })));
        assert_eq!(app.stale.len(), 1, "re-indexed");
    }

    fn open_list(app: &mut App, titles: &[&str]) {
        app.update(Action::ToggleSidebar);
        app.update(Action::Storage(StoreEvent::Listed {
            conversations: titles
                .iter()
                .map(|t| crate::storage::ConversationSummary {
                    id: ConversationId((*t).into()),
                    title: (*t).into(),
                    provider: "ollama".into(),
                    model: "m".into(),
                    updated_at: 0,
                })
                .collect(),
            now: 0,
        }));
    }

    #[test]
    fn typing_in_the_list_searches_and_esc_clears_then_closes() {
        let mut app = app();
        open_list(&mut app, &["a", "b"]);
        assert_eq!(
            app.update(Action::SidebarType('r')),
            vec![Effect::Store(StoreRequest::Search("r".into()))]
        );
        app.update(Action::SidebarType('u'));
        // A stale answer (for "r") is ignored, the current one is shown.
        let hit = |id: &str| {
            (
                crate::storage::ConversationSummary {
                    id: ConversationId(id.into()),
                    title: id.into(),
                    provider: "ollama".into(),
                    model: "m".into(),
                    updated_at: 0,
                },
                Some("… du rust …".to_owned()),
            )
        };
        app.update(Action::Storage(StoreEvent::Searched {
            query: "r".into(),
            results: vec![hit("a"), hit("b")],
            now: 0,
        }));
        app.update(Action::Storage(StoreEvent::Searched {
            query: "ru".into(),
            results: vec![hit("b")],
            now: 0,
        }));
        let sidebar = app.sidebar.as_ref().expect("open");
        assert_eq!(sidebar.items.as_ref().map(Vec::len), Some(1));
        assert_eq!(sidebar.snippet(0), Some("… du rust …"));

        assert_eq!(
            app.update(Action::Cancel),
            vec![Effect::Store(StoreRequest::List)],
            "Esc clears the search first"
        );
        assert!(app.sidebar.is_some());
        app.update(Action::Cancel);
        assert!(app.sidebar.is_none());
    }

    #[test]
    fn renaming_and_deleting_from_the_list() {
        let mut app = app();
        send(&mut app, "Question");
        let current = app.conversation_id.clone().expect("stored");
        open_list(&mut app, &["x"]);
        // Make the current conversation the listed one.
        if let Some(sidebar) = &mut app.sidebar {
            sidebar.items = Some(vec![crate::storage::ConversationSummary {
                id: current.clone(),
                title: "Question".into(),
                provider: "ollama".into(),
                model: "m".into(),
                updated_at: 0,
            }]);
        }
        app.update(Action::SidebarRename);
        app.update(Action::SidebarBackspace);
        for c in "ns du jour".chars() {
            app.update(Action::SidebarType(c));
        }
        assert_eq!(
            app.update(Action::SidebarOpen),
            vec![Effect::Store(StoreRequest::Rename {
                id: current.clone(),
                title: "Questions du jour".into()
            })]
        );
        assert_eq!(app.conversation_title.as_deref(), Some("Questions du jour"));

        assert!(app.update(Action::SidebarDelete).is_empty(), "asks first");
        app.update(Action::SidebarDown);
        assert!(
            app.update(Action::SidebarDelete).is_empty(),
            "moving resets it"
        );
        assert_eq!(
            app.update(Action::SidebarDelete),
            vec![Effect::Store(StoreRequest::DeleteConversation(
                current.clone()
            ))]
        );
        let effects = app.update(Action::Storage(StoreEvent::ConversationDeleted(current)));
        assert!(effects.contains(&Effect::Store(StoreRequest::List)));
        assert!(
            app.conversation.is_empty(),
            "the deleted conversation is closed"
        );
        assert!(app.conversation_id.is_none());
        assert!(app.sidebar.is_some(), "the list stays open");
    }

    #[test]
    fn rename_and_delete_commands() {
        let mut app = app();
        assert!(
            app.run_command(CommandId::Rename, "Brouillon").is_empty(),
            "not stored yet"
        );
        assert_eq!(app.conversation_title.as_deref(), Some("Brouillon"));
        let job = send(&mut app, "Question");
        assert_eq!(app.conversation_title.as_deref(), Some("Brouillon"), "kept");
        llm(&mut app, job.request_id, LlmEvent::Done);
        let id = app.conversation_id.clone().expect("stored");
        assert!(app.run_command(CommandId::Delete, "").is_empty());
        assert_eq!(app.input_text(), "/delete");
        assert_eq!(
            app.update(Action::Submit),
            vec![Effect::Store(StoreRequest::DeleteConversation(id))]
        );
    }

    #[test]
    fn copy_the_last_reply_or_one_of_its_code_blocks() {
        let mut app = app();
        assert!(app.run_command(CommandId::Copy, "").is_empty());
        assert_eq!(app.status, Status::Error("aucune réponse à copier".into()));

        let job = send(&mut app, "Code ?");
        token(
            &mut app,
            job.request_id,
            "Deux :\n\n```rust\nfn a() {}\n```\n\n```sh\nls\n```\n",
        );
        assert!(
            app.update(Action::CopyLastReply).is_empty(),
            "still streaming"
        );
        llm(&mut app, job.request_id, LlmEvent::Done);

        let copied = |effects: Vec<Effect>| match effects.as_slice() {
            [Effect::Copy { text, what }] => (text.clone(), what.clone()),
            other => panic!("{other:?}"),
        };
        let (text, what) = copied(app.update(Action::CopyLastReply));
        assert!(text.starts_with("Deux :"));
        assert_eq!(what, "dernière réponse");
        assert_eq!(
            copied(app.run_command(CommandId::Copy, "code")).0,
            "ls\n",
            "last block"
        );
        assert_eq!(
            copied(app.run_command(CommandId::Copy, "code 1")),
            ("fn a() {}\n".to_owned(), "bloc de code #1".to_owned())
        );
        assert!(app.run_command(CommandId::Copy, "code 3").is_empty());
        assert_eq!(
            app.status,
            Status::Error("pas de bloc #3 : la dernière réponse en a 2".into())
        );

        app.update(Action::Copied {
            what: "bloc de code #1".into(),
            chars: 10,
            how: crate::clipboard::Copied::Tool("wl-copy"),
        });
        assert_eq!(
            app.status,
            Status::Info("copié : bloc de code #1 (10 caractères, via wl-copy)".into())
        );
    }

    /// An app whose model has an 8k window (from the configuration).
    fn small_window_app(auto_compact: bool) -> App {
        let mut config = Config {
            auto_compact,
            ..Config::default()
        };
        if let Some(ollama) = config.providers.get_mut("ollama") {
            ollama.context_window = Some(8_000);
        }
        App::new(&config, false)
    }

    fn fill(app: &mut App, input_tokens: u64) {
        let job = send(app, "Question");
        token(app, job.request_id, "Réponse");
        llm(
            app,
            job.request_id,
            LlmEvent::Usage(Usage {
                input_tokens: Some(input_tokens),
                output_tokens: Some(10),
            }),
        );
        llm(app, job.request_id, LlmEvent::Done);
    }

    #[test]
    fn a_nearly_full_context_suggests_compact() {
        let mut app = small_window_app(false);
        fill(&mut app, 1_000);
        assert_eq!(app.status, Status::Ready);
        fill(&mut app, 7_400);
        assert_eq!(
            app.status,
            Status::Info(
                "contexte rempli à 92 % : /compact le résume, /clear repart de zéro".into()
            )
        );
        // Without auto_compact, the next message is sent as usual.
        assert_eq!(send(&mut app, "Encore").kind, JobKind::Reply);

        let mut quiet = small_window_app(false);
        quiet.compact_threshold = 0;
        fill(&mut quiet, 7_400);
        assert_eq!(quiet.status, Status::Ready, "suggestion turned off");
    }

    #[test]
    fn auto_compact_summarizes_then_sends_the_message() {
        let mut app = small_window_app(true);
        fill(&mut app, 7_400);
        assert!(
            matches!(&app.status, Status::Info(m) if m.ends_with("avant votre prochain message"))
        );

        let summary = send(&mut app, "La suite ?");
        assert_eq!(
            summary.kind,
            JobKind::Summary,
            "the history is summarized first"
        );
        assert!(app.input.is_empty());
        token(&mut app, summary.request_id, "- résumé");
        let effects = app.update(Action::Llm {
            request_id: summary.request_id,
            event: LlmEvent::Done,
        });
        let reply = effects
            .iter()
            .find_map(|e| match e {
                Effect::StartCompletion(job) => Some(job.clone()),
                _ => None,
            })
            .expect("the queued message is sent");
        assert_eq!(reply.kind, JobKind::Reply);
        let contents: Vec<&str> = reply.history.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, vec!["- résumé", "La suite ?"]);
    }

    #[test]
    fn a_failed_auto_compact_gives_the_message_back() {
        let mut app = small_window_app(true);
        fill(&mut app, 7_400);
        let summary = send(&mut app, "La suite ?");
        llm(
            &mut app,
            summary.request_id,
            LlmEvent::Error("délai dépassé".into()),
        );
        assert_eq!(app.input_text(), "La suite ?");
        assert_eq!(
            app.status,
            Status::Error(
                "délai dépassé : message non envoyé (il est dans la zone de saisie)".into()
            )
        );

        // Enter again: the summary is retried.
        let retry = app.update(Action::Submit);
        assert!(matches!(
            retry.as_slice(),
            [Effect::StartCompletion(job)] if job.kind == JobKind::Summary
        ));
        app.update(Action::Cancel);
        assert_eq!(app.input_text(), "La suite ?");
    }
}
