//! End-to-end state tests without a terminal: `App` + streaming task + mocked `LlmClient`
//! + in-memory SQLite store.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use chatatui::{
    action::{Action, Effect},
    app::App,
    config::Config,
    context::NoContext,
    event::{AppEvent, Event},
    llm::{
        Clients, LlmClient, LlmError, ProviderModels,
        mock::{MockLlmClient, MockReply},
        stream_task::{self, Backends},
    },
    rag::{
        embed::HashEmbedder,
        indexer::{self, IndexRequest},
        retrieve::{RagContext, Selection},
    },
    state::{MessageStatus, Overlay, Role, Status},
    storage::Store,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A tiny stand-in for `Runtime`: executes effects and feeds events back into the app.
struct Harness {
    app: App,
    backends: Backends,
    tx: mpsc::UnboundedSender<Event>,
    rx: mpsc::UnboundedReceiver<Event>,
    cancel: Option<CancellationToken>,
    /// Where the answers to tool calls go (for the running job).
    decisions: Option<mpsc::UnboundedSender<bool>>,
    store: Store,
    /// Database file shared by the store and the indexer (`None`: in-memory store).
    database: Option<std::path::PathBuf>,
    index_cancel: Option<CancellationToken>,
    /// Download effects the app asked for (no HTTP runs in the harness).
    pulls: Vec<Effect>,
}

impl Harness {
    fn new(llm: Arc<MockLlmClient>) -> Self {
        Self::with_providers(&[("ollama", llm)])
    }

    /// A harness whose providers (presets: ollama, openai, claude) are served by mocks.
    fn with_providers(clients: &[(&str, Arc<MockLlmClient>)]) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let clients: Clients = clients
            .iter()
            .map(|(id, llm)| ((*id).to_owned(), llm.clone() as Arc<dyn LlmClient>))
            .collect();
        Self {
            app: App::new(&Config::default(), false),
            backends: Backends {
                clients,
                context: Arc::new(NoContext),
                mcp: Arc::default(),
            },
            tx,
            rx,
            cancel: None,
            decisions: None,
            store: Store::open_in_memory().expect("in-memory store"),
            database: None,
            index_cancel: None,
            pulls: Vec::new(),
        }
    }

    /// A harness whose store lives in `path`, so that `/index` can run (hash embeddings).
    fn with_database(path: &std::path::Path) -> Self {
        Self::with_database_and_llm(path, Arc::new(MockLlmClient::new([])))
    }

    /// Same, with replies searching the conversation's collection (hash embeddings).
    fn with_database_and_llm(path: &std::path::Path, llm: Arc<MockLlmClient>) -> Self {
        let mut h = Self::new(llm);
        h.store = Store::open(path).expect("file store");
        h.database = Some(path.to_owned());
        h.backends.context = Arc::new(RagContext::new(
            Ok(Arc::new(HashEmbedder::default())),
            Ok(path.to_owned()),
            Selection {
                top_k: 1,
                context_tokens: 10_000,
                min_score: 0.0,
                keywords: true,
            },
        ));
        h
    }

    fn dispatch(&mut self, action: Action) {
        let mut queue = VecDeque::from([action]);
        while let Some(action) = queue.pop_front() {
            for effect in self.app.update(action) {
                if let Some(follow_up) = self.execute(effect) {
                    queue.push_back(follow_up);
                }
            }
        }
    }

    /// Runs a effect; storage answers synchronously and are returned as the next action.
    fn execute(&mut self, effect: Effect) -> Option<Action> {
        {
            match effect {
                // Nothing to draw on: this harness has no frame and no clock. `Runtime` is
                // where an animation becomes visible.
                Effect::Animate(_) => {}
                Effect::StartCompletion(job) => {
                    let token = CancellationToken::new();
                    self.cancel = Some(token.clone());
                    let (decisions, answers) = mpsc::unbounded_channel();
                    self.decisions = Some(decisions);
                    tokio::spawn(stream_task::run_with_tools(
                        self.backends.clone(),
                        job,
                        token,
                        self.tx.clone(),
                        Some(answers),
                    ));
                }
                Effect::ToolDecision { allow, .. } => {
                    if let Some(decisions) = &self.decisions {
                        let _ = decisions.send(allow);
                    }
                }
                Effect::CheckCollections => {
                    let database = self.database.clone()?;
                    return Some(Action::CollectionsChecked(indexer::check_collections(
                        &database,
                        &[],
                    )));
                }
                Effect::StartIndex {
                    collection,
                    root,
                    types,
                } => {
                    let database = self.database.clone().expect("a file database");
                    let token = CancellationToken::new();
                    self.index_cancel = Some(token.clone());
                    let tx = self.tx.clone();
                    tokio::spawn(indexer::run(
                        IndexRequest {
                            collection,
                            root: root.into(),
                            chunk_tokens: 200,
                            types,
                            ..IndexRequest::default()
                        },
                        database,
                        Arc::new(HashEmbedder::default()),
                        token,
                        move |event| {
                            let _ = tx.send(Event::App(AppEvent::Index(event)));
                        },
                    ));
                }
                Effect::CancelIndex => {
                    if let Some(token) = self.index_cancel.take() {
                        token.cancel();
                    }
                }
                // The download tests drive the events in by hand, so the harness only
                // records what the app asked for; no HTTP ever runs here.
                Effect::ListGguf { .. } | Effect::StartPull { .. } | Effect::CancelPull => {
                    self.pulls.push(effect);
                }
                Effect::CancelCompletion(_) => {
                    if let Some(token) = self.cancel.take() {
                        token.cancel();
                    }
                }
                Effect::Store(request) => return self.store.handle(request).map(Action::Storage),
                Effect::ReadFile(path) => {
                    return Some(Action::FileRead(chatatui::files::read_attachment(&path)));
                }
                Effect::Export {
                    path,
                    suggested,
                    content,
                } => {
                    return Some(Action::Exported(chatatui::files::write_new(
                        path.as_deref(),
                        &suggested,
                        &content,
                    )));
                }
                Effect::Copy { what, text } => {
                    return Some(Action::Copied {
                        what,
                        chars: text.chars().count(),
                        how: chatatui::clipboard::Copied::TerminalOnly,
                    });
                }
                Effect::CompletePath(partial) => {
                    let candidates = chatatui::files::complete_path(&partial);
                    return Some(Action::PathCompleted {
                        partial,
                        candidates,
                    });
                }
                Effect::DetectContextWindow { provider, model } => {
                    let client = self.backends.clients.get(&provider)?;
                    let tokens = futures::executor::block_on(client.context_window(&model));
                    return Some(Action::ContextWindowDetected {
                        provider,
                        model,
                        tokens,
                    });
                }
                Effect::ListModels => {
                    // The mocks answer immediately, so blocking here is harmless.
                    let results = self
                        .app
                        .providers
                        .iter()
                        .filter_map(|p| {
                            let client = self.backends.clients.get(&p.id)?;
                            let result = futures::executor::block_on(client.list_models())
                                .map_err(|e| e.to_string());
                            Some(ProviderModels {
                                provider: p.id.clone(),
                                result,
                            })
                        })
                        .collect();
                    return Some(Action::ModelsListed(results));
                }
            }
        }
        None
    }

    fn send(&mut self, text: &str) {
        for c in text.chars() {
            self.dispatch(Action::Edit(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        self.dispatch(Action::Submit);
    }

    /// Processes the next event from the streaming task.
    async fn step(&mut self) {
        let event = tokio::time::timeout(Duration::from_secs(2), self.rx.recv())
            .await
            .expect("event within timeout")
            .expect("channel open");
        match event {
            Event::App(AppEvent::Llm { request_id, event }) => {
                self.dispatch(Action::Llm { request_id, event });
            }
            Event::App(AppEvent::Index(event)) => self.dispatch(Action::Index(event)),
            _ => panic!("unexpected event"),
        }
    }

    /// Processes events until the indexing job ends.
    async fn run_until_indexed(&mut self) {
        while self.app.indexing.is_some() {
            self.step().await;
        }
    }

    /// Processes events until the generation ends.
    async fn run_until_idle(&mut self) {
        while self.app.is_generating() {
            self.step().await;
        }
    }

    fn last_reply(&self) -> (&str, &MessageStatus) {
        let message = self.app.conversation.messages().last().expect("a message");
        (&message.content, &message.status)
    }
}

#[tokio::test]
async fn reply_is_streamed_into_the_conversation() {
    let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&[
        "Bonjour",
        " !",
        " Comment",
        " ça va ?",
    ])]));
    let mut h = Harness::new(llm);
    h.send("Salut");
    assert_eq!(h.app.status, Status::Generating);

    h.run_until_idle().await;
    assert_eq!(
        h.last_reply(),
        ("Bonjour ! Comment ça va ?", &MessageStatus::Complete)
    );
    assert_eq!(h.app.status, Status::Ready);
}

/// The whole point: the screen must stop being still. A reply that has not started yet
/// says which wait it is in, and the line is replaced by the text in place.
#[tokio::test]
async fn the_wait_before_the_first_token_is_visible() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new([MockReply::tokens(&[
        "Il était une fois",
    ])])));
    // A real width: the default (zero) viewport wraps at one column, breaking every
    // substring check below across newlines.
    harness.app.update(Action::Resize {
        width: 60,
        height: 20,
    });
    let effects = submit(&mut harness.app, "Raconte une histoire");
    let request_id = effects
        .into_iter()
        .find_map(|e| match e {
            Effect::StartCompletion(job) => Some(job.request_id),
            _ => None,
        })
        .expect("a job was started");

    harness.app.update(Action::Llm {
        request_id,
        event: chatatui::llm::LlmEvent::Phase(chatatui::llm::Phase::Waiting {
            model: "llama3.2".to_owned(),
        }),
    });
    for _ in 0..60 {
        harness.app.update(Action::Tick);
    }
    let text = harness.app.transcript.line_texts().join("\n");
    assert!(text.contains("llama3.2 réfléchit…"), "{text}");
    assert!(text.contains("2 s"), "{text}");

    harness.app.update(Action::Llm {
        request_id,
        event: chatatui::llm::LlmEvent::Token("Il était".to_owned()),
    });
    harness.app.update(Action::Tick);
    let text = harness.app.transcript.line_texts().join("\n");
    assert!(!text.contains("réfléchit"), "replaced in place: {text}");
    assert!(text.contains("Il était"), "{text}");
}

#[tokio::test]
async fn second_turn_sends_the_whole_history() {
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Paris."]),
        MockReply::tokens(&["Environ 2 millions."]),
    ]));
    let mut h = Harness::new(llm.clone());
    h.send("Capitale de la France ?");
    h.run_until_idle().await;
    h.send("Population ?");
    h.run_until_idle().await;

    let second = &llm.requests()[1];
    let contents: Vec<&str> = second.messages.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(
        contents[1..],
        ["Capitale de la France ?", "Paris.", "Population ?"]
    );
}

#[tokio::test]
async fn escape_cancels_a_running_generation() {
    let llm = Arc::new(MockLlmClient::new([MockReply::TokensThenHang(vec![
        "Il était".into(),
        " une fois".into(),
    ])]));
    let mut h = Harness::new(llm.clone());
    h.send("Raconte une histoire");
    // Retrieving, connecting, waiting, then the two tokens.
    for _ in 0..5 {
        h.step().await;
    }

    h.dispatch(Action::Cancel);
    assert_eq!(
        h.last_reply(),
        ("Il était une fois", &MessageStatus::Cancelled)
    );
    assert_eq!(h.app.status, Status::Ready);

    // The task stops and releases the stream.
    tokio::time::timeout(Duration::from_secs(1), async {
        while llm.dropped_streams() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("stream dropped after cancellation");
}

#[tokio::test]
async fn unreachable_server_is_shown_as_an_error() {
    let llm = Arc::new(MockLlmClient::new([MockReply::Fail(
        LlmError::Unreachable {
            server: "Ollama".into(),
            url: "http://localhost:11434/v1".into(),
        },
    )]));
    let mut h = Harness::new(llm);
    h.send("Allô ?");
    h.run_until_idle().await;

    let message = "Ollama injoignable sur http://localhost:11434/v1".to_owned();
    // The failure is told in the conversation, where the reply would have been — not in the
    // status bar, which keeps its hints.
    assert_eq!(h.app.status, Status::Ready);
    assert_eq!(h.last_reply(), ("", &MessageStatus::Failed(message)));

    // The app keeps working: the next message starts a new generation.
    h.send("Et maintenant ?");
    assert_eq!(h.app.status, Status::Generating);
}

#[tokio::test]
async fn conversations_are_saved_and_reopened() {
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Paris."]),
        MockReply::TokensThenHang(vec!["Envi".into()]),
        MockReply::tokens(&["Bonjour !"]),
    ]));
    let mut h = Harness::new(llm.clone());

    // First conversation: one full exchange and one interrupted reply.
    h.send("Capitale de la France ?");
    h.run_until_idle().await;
    h.send("Population ?");
    // Retrieving, connecting, waiting, then the one token.
    for _ in 0..4 {
        h.step().await;
    }
    h.dispatch(Action::Cancel);
    let first_id = h.app.conversation_id.clone().expect("stored");
    let first_messages = h.app.conversation.messages().to_vec();

    // Second conversation.
    h.dispatch(Action::NewConversation);
    assert!(h.app.conversation.is_empty());
    assert!(
        h.app.conversation_id.is_none(),
        "not stored before its first message"
    );
    h.send("Salut");
    h.run_until_idle().await;
    assert_ne!(h.app.conversation_id.as_ref(), Some(&first_id));

    // The list shows both, most recent first; the current one is highlighted.
    h.dispatch(Action::ToggleSidebar);
    let sidebar = h.app.sidebar.clone().expect("open");
    let titles: Vec<&str> = sidebar
        .items
        .as_deref()
        .expect("listed")
        .iter()
        .map(|s| s.title.as_str())
        .collect();
    assert_eq!(titles.len(), 2);
    assert!(titles.contains(&"Capitale de la France ?"));
    assert_eq!(
        sidebar.selected_item().map(|s| &s.id),
        h.app.conversation_id.as_ref()
    );

    // Reopen the first one.
    let first_index = titles
        .iter()
        .position(|t| *t == "Capitale de la France ?")
        .expect("listed");
    while h.app.sidebar.as_ref().map(|s| s.selected) != Some(first_index) {
        h.dispatch(Action::SidebarDown);
    }
    h.dispatch(Action::SidebarOpen);
    assert!(h.app.sidebar.is_none(), "panel closes after loading");
    assert_eq!(h.app.conversation_id.as_ref(), Some(&first_id));
    assert_eq!(h.app.conversation.messages(), first_messages.as_slice());
    assert_eq!(
        h.app.conversation.messages()[3].status,
        MessageStatus::Cancelled
    );

    // Continuing it sends the stored history to the model.
    h.send("Et Lyon ?");
    h.run_until_idle().await;
    let last_request = llm.requests().pop().expect("request");
    let roles: Vec<_> = last_request.messages.iter().map(|m| m.role).collect();
    assert_eq!(roles.len(), 1 + 5, "system + 5 messages");
    assert_eq!(h.app.conversation.messages()[4].role, Role::User);
}

#[tokio::test]
async fn switching_model_applies_to_the_next_request_and_the_stored_conversation() {
    let llm = Arc::new(
        MockLlmClient::new([MockReply::tokens(&["un"]), MockReply::tokens(&["deux"])])
            .with_models(&["llama3.2", "mistral:7b", "qwen2.5:7b"]),
    );
    let mut h = Harness::new(llm.clone());
    h.send("Salut");
    h.run_until_idle().await;

    h.dispatch(Action::OpenModelPicker);
    let picker = h.app.model_picker().cloned().expect("open");
    assert_eq!(
        picker.selected_choice().map(|c| c.model.as_str()),
        Some("llama3.2"),
        "current model highlighted"
    );
    for c in "qwen".chars() {
        h.dispatch(Action::OverlayFilter(c));
    }
    h.dispatch(Action::OverlaySelect);
    assert!(h.app.model_picker().is_none());
    assert_eq!(h.app.model, "qwen2.5:7b");

    h.send("Encore");
    h.run_until_idle().await;
    let models: Vec<String> = llm.requests().into_iter().map(|r| r.model).collect();
    assert_eq!(models, ["llama3.2", "qwen2.5:7b"]);

    let id = h.app.conversation_id.clone().expect("stored");
    let stored = h.store.load(&id).expect("load");
    assert_eq!(stored.summary.model, "qwen2.5:7b");
}

#[tokio::test]
async fn a_conversation_can_move_from_a_local_model_to_claude() {
    let ollama =
        Arc::new(MockLlmClient::new([MockReply::tokens(&["local"])]).with_models(&["llama3.2"]));
    let claude =
        Arc::new(MockLlmClient::new([MockReply::tokens(&["cloud"])]).with_models(&["claude-test"]));
    let mut h = Harness::with_providers(&[("ollama", ollama.clone()), ("claude", claude.clone())]);
    assert!(h.app.is_local());

    h.send("Question locale");
    h.run_until_idle().await;

    // The picker shows both providers; choose Claude.
    h.dispatch(Action::OpenModelPicker);
    let picker = h.app.model_picker().cloned().expect("open");
    let labels: Vec<String> = picker
        .visible()
        .iter()
        .map(|c| format!("{} {}", c.label, c.model))
        .collect();
    assert_eq!(labels, ["Ollama llama3.2", "Claude claude-test"]);
    for c in "claude".chars() {
        h.dispatch(Action::OverlayFilter(c));
    }
    h.dispatch(Action::OverlaySelect);
    assert_eq!(
        (h.app.provider.as_str(), h.app.model.as_str()),
        ("claude", "claude-test")
    );
    assert!(!h.app.is_local());

    // Claude receives the whole history, including the local answer.
    h.send("Et toi ?");
    h.run_until_idle().await;
    let request = claude.requests().pop().expect("claude was called");
    let contents: Vec<&str> = request
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(contents[1..], ["Question locale", "local", "Et toi ?"]);
    assert_eq!(ollama.requests().len(), 1);

    // Reopening the conversation restores Claude.
    let id = h.app.conversation_id.clone().expect("stored");
    h.dispatch(Action::NewConversation);
    h.dispatch(Action::Submit);
    h.app.provider = "ollama".into();
    let stored = h.store.load(&id).expect("load");
    assert_eq!(stored.summary.provider, "claude");
    h.dispatch(Action::Storage(chatatui::storage::StoreEvent::Loaded(
        stored,
    )));
    assert_eq!(h.app.provider, "claude");
}

#[tokio::test]
async fn missing_key_is_explained_in_the_picker_and_on_send() {
    let ollama = Arc::new(MockLlmClient::new([]).with_models(&["llama3.2"]));
    let mut h = Harness::with_providers(&[("ollama", ollama)]);
    // Simulate the runtime: Claude without key gets an `Unavailable` client.
    h.backends.clients.insert(
        "claude".into(),
        Arc::new(chatatui::llm::Unavailable(LlmError::MissingKey {
            server: "Claude".into(),
            env: "ANTHROPIC_API_KEY".into(),
        })),
    );

    h.dispatch(Action::OpenModelPicker);
    let picker = h.app.model_picker().cloned().expect("open");
    assert_eq!(
        picker.errors(),
        [(
            "Claude".to_owned(),
            "Claude : clé API absente (définissez la variable ANTHROPIC_API_KEY)".to_owned()
        )]
    );
    h.dispatch(Action::Cancel);

    h.dispatch(Action::Submit);
    for c in "/model claude claude-x".chars() {
        h.dispatch(Action::Edit(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
    h.dispatch(Action::Submit);
    h.send("Allô ?");
    h.run_until_idle().await;
    // Sending anyway must still say why, and it says it in the conversation: the turn is
    // marked failed and carries the sentence the picker showed.
    assert_eq!(
        h.last_reply(),
        (
            "",
            &MessageStatus::Failed(
                "Claude : clé API absente (définissez la variable ANTHROPIC_API_KEY)".to_owned()
            )
        )
    );
    assert_eq!(h.app.status, Status::Ready);
}

#[tokio::test]
async fn context_gauge_uses_measured_tokens_and_the_detected_window() {
    let usage = chatatui::llm::Usage {
        input_tokens: Some(1_200),
        output_tokens: Some(300),
    };
    let llm = Arc::new(
        MockLlmClient::new([MockReply::TokensWithUsage(vec!["Réponse".into()], usage)])
            .with_context_window(8_192),
    );
    let mut h = Harness::new(llm);
    h.dispatch(Action::Init);
    assert_eq!(
        h.app.context_window(),
        Some((8_192, chatatui::app::WindowSource::Server))
    );

    h.send("Question");
    h.run_until_idle().await;
    let used = h.app.context_usage();
    assert!(used.measured);
    assert_eq!(used.tokens, 1_500);

    // A new message is added on top of the measure, as an estimate.
    for c in "Suite".chars() {
        h.dispatch(Action::Edit(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
    h.dispatch(Action::Submit);
    let used = h.app.context_usage();
    assert!(!used.measured);
    assert!(used.tokens > 1_500);
}

impl Harness {
    /// Types `text` in the input and presses Enter.
    fn command(&mut self, text: &str) {
        self.send(text);
    }

    fn reload(&mut self) {
        let id = self.app.conversation_id.clone().expect("stored");
        let stored = self.store.load(&id).expect("load");
        self.dispatch(Action::NewConversation);
        self.dispatch(Action::Storage(chatatui::storage::StoreEvent::Loaded(
            stored,
        )));
    }
}

fn contents(request: &chatatui::llm::ChatRequest) -> Vec<String> {
    request.messages.iter().map(|m| m.content.clone()).collect()
}

#[tokio::test]
async fn attached_file_is_sent_as_context_and_kept_on_reload() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("cours.md");
    std::fs::write(&file, "Séance 1 : ownership.\nSéance 2 : lifetimes.").expect("write");
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Deux séances."]),
        MockReply::tokens(&["Les lifetimes."]),
    ]));
    let mut h = Harness::new(llm.clone());

    h.command(&format!("/add {}", file.display()));
    let attached = h
        .app
        .conversation
        .messages()
        .last()
        .expect("attachment")
        .clone();
    assert_eq!(attached.role, Role::Attachment);
    assert!(matches!(&h.app.status, Status::Info(m) if m.starts_with("fichier joint : cours.md")));

    h.send("Combien de séances ?");
    h.run_until_idle().await;
    let request = llm.requests().pop().expect("request");
    let system = &request.messages[0].content;
    assert!(system.contains(&format!("[1] {}", file.display())));
    assert!(system.contains("Séance 2 : lifetimes."));
    assert_eq!(
        request.messages.len(),
        2,
        "system (with the file) + the question"
    );

    // After reloading, the file is still part of the context.
    h.reload();
    assert_eq!(h.app.conversation.messages()[0], attached);
    h.send("Et la deuxième ?");
    h.run_until_idle().await;
    let request = llm.requests().pop().expect("request");
    assert!(
        request.messages[0]
            .content
            .contains("Séance 2 : lifetimes.")
    );
}

#[tokio::test]
async fn unreadable_file_is_reported() {
    let mut h = Harness::new(Arc::new(MockLlmClient::new([])));
    h.command("/add /definitely/not/here.md");
    assert_eq!(
        h.app.status,
        Status::Error("/definitely/not/here.md : fichier introuvable".into())
    );
    assert!(h.app.conversation.is_empty());
}

#[tokio::test]
async fn tab_completes_the_path_after_add() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("rapport-final.md"), "x").expect("write");
    let mut h = Harness::new(Arc::new(MockLlmClient::new([])));
    let partial = format!("/add {}/rap", dir.path().display());
    for c in partial.chars() {
        h.dispatch(Action::Edit(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
    assert!(h.app.key_context().completing_path);
    h.dispatch(Action::CompletePath);
    assert_eq!(
        h.app.input_text(),
        format!("/add {}/rapport-final.md", dir.path().display())
    );
}

#[tokio::test]
async fn clear_stops_sending_older_messages_even_after_reload() {
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Réponse 1"]),
        MockReply::tokens(&["Réponse 2"]),
        MockReply::tokens(&["Réponse 3"]),
    ]));
    let mut h = Harness::new(llm.clone());
    h.send("Sujet A");
    h.run_until_idle().await;

    h.command("/clear");
    assert!(matches!(&h.app.status, Status::Info(m) if m.starts_with("contexte vidé")));
    assert_eq!(h.app.conversation.messages().len(), 2, "still displayed");

    h.send("Sujet B");
    h.run_until_idle().await;
    let sent = contents(&llm.requests().pop().expect("request"));
    assert!(!sent.iter().any(|c| c.contains("Sujet A")));
    assert!(sent.iter().any(|c| c == "Sujet B"));

    h.reload();
    h.send("Sujet C");
    h.run_until_idle().await;
    let sent = contents(&llm.requests().pop().expect("request"));
    assert!(
        !sent.iter().any(|c| c.contains("Sujet A")),
        "boundary persisted"
    );
    assert!(sent.iter().any(|c| c == "Sujet B"));
}

#[tokio::test]
async fn compact_replaces_the_history_with_a_summary() {
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Paris."]),
        MockReply::tokens(&["- La capitale de la France est Paris."]),
        MockReply::tokens(&["Environ 2 millions."]),
    ]));
    let mut h = Harness::new(llm.clone());
    h.send("Capitale de la France ?");
    h.run_until_idle().await;

    h.command("/compact");
    assert!(h.app.is_generating(), "the summary streams like a reply");
    h.run_until_idle().await;
    let summary_request = llm.requests().pop().expect("summary request");
    assert!(contents(&summary_request)[1].contains("### User\nCapitale de la France ?"));
    let summary = h
        .app
        .conversation
        .messages()
        .last()
        .expect("summary")
        .clone();
    assert_eq!(summary.role, Role::Summary);
    assert_eq!(h.app.conversation.context_start(), summary.id.0);

    h.send("Et sa population ?");
    h.run_until_idle().await;
    let sent = contents(&llm.requests().pop().expect("request"));
    assert!(
        sent[0].contains("- La capitale de la France est Paris."),
        "summary in system"
    );
    assert_eq!(sent[1..], ["Et sa population ?"], "old messages replaced");

    h.reload();
    assert_eq!(h.app.conversation.context_start(), summary.id.0);
}

#[tokio::test]
async fn cancelled_compact_leaves_the_context_unchanged() {
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Paris."]),
        MockReply::TokensThenHang(vec!["- La capi".into()]),
    ]));
    let mut h = Harness::new(llm);
    h.send("Capitale ?");
    h.run_until_idle().await;
    h.command("/compact");
    h.step().await;
    h.dispatch(Action::Cancel);
    assert_eq!(h.app.conversation.context_start(), 0);
    assert_eq!(
        h.app.prompt().len(),
        3,
        "system + question + answer: summary ignored"
    );
}

fn write_docs(dir: &std::path::Path) {
    std::fs::write(dir.join("cours.md"), "# Cours\n\nL'ownership en Rust.").expect("write");
    std::fs::write(dir.join("notes.txt"), "Les traits et les génériques.").expect("write");
    std::fs::write(dir.join("image.bin"), [0u8, 159, 146, 150]).expect("write");
}

#[tokio::test]
async fn index_builds_a_collection_listed_in_collections() {
    let dir = tempfile::tempdir().expect("temp dir");
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).expect("mkdir");
    write_docs(&docs);
    let mut h = Harness::with_database(&dir.path().join("db.sqlite"));

    h.command(&format!("/index {}", docs.display()));
    assert_eq!(
        h.app.indexing.as_ref().map(|p| p.collection.as_str()),
        Some("docs")
    );
    h.run_until_indexed().await;
    let report = h.app.last_index.clone().expect("report");
    assert_eq!((report.files, report.added), (2, 2));
    assert!(
        matches!(&h.app.status, Status::Info(m) if m.starts_with("« docs » indexée : 2 fichiers, 2 ajoutés"))
    );

    h.command("/collections");
    assert!(matches!(h.app.overlay, Some(Overlay::Collections { .. })));
    let (collections, _) = h.app.collections.clone().expect("listed");
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "docs");
    assert_eq!(collections[0].documents, 2);

    // A second run with an explicit name: nothing changed in the folder.
    h.dispatch(Action::Cancel);
    h.command(&format!("/index {} docs", docs.display()));
    h.run_until_indexed().await;
    let report = h.app.last_index.clone().expect("report");
    assert_eq!((report.added, report.unchanged), (0, 2));
}

#[tokio::test]
async fn index_refuses_a_second_run_and_escape_stops_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    write_docs(dir.path());
    let mut h = Harness::with_database(&dir.path().join("db.sqlite"));
    let root = dir.path().display().to_string();

    h.command(&format!("/index {root} a"));
    h.command(&format!("/index {root} b"));
    assert!(matches!(&h.app.status, Status::Error(m) if m.contains("« a » déjà en cours")));

    h.dispatch(Action::Cancel);
    h.run_until_indexed().await;
    // Either the job was stopped in time, or it had already finished.
    assert!(matches!(&h.app.status, Status::Info(m)
        if m == "indexation de « a » arrêtée" || m.starts_with("« a » indexée")));
    assert!(h.app.indexing.is_none());
}

#[tokio::test]
async fn index_of_a_missing_folder_fails_cleanly() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut h = Harness::with_database(&dir.path().join("db.sqlite"));
    h.command("/index /definitely/not/here");
    h.run_until_indexed().await;
    assert!(matches!(&h.app.status, Status::Error(m) if m.starts_with("indexation de « here »")));
}

#[tokio::test]
async fn tab_completes_the_folder_after_index() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::create_dir(dir.path().join("documents")).expect("mkdir");
    let mut h = Harness::new(Arc::new(MockLlmClient::new([])));
    for c in format!("/index {}/doc", dir.path().display()).chars() {
        h.dispatch(Action::Edit(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
    assert!(h.app.key_context().completing_path);
    h.dispatch(Action::CompletePath);
    assert_eq!(
        h.app.input_text(),
        format!("/index {}/documents/", dir.path().display())
    );
    // Once a name is being typed, Tab no longer completes paths.
    for c in " nom".chars() {
        h.dispatch(Action::Edit(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
    assert!(!h.app.key_context().completing_path);
}

#[tokio::test]
async fn rag_answers_from_the_collection_and_keeps_the_sources() {
    let dir = tempfile::tempdir().expect("temp dir");
    let docs = dir.path().join("cours");
    std::fs::create_dir(&docs).expect("mkdir");
    std::fs::write(
        docs.join("ownership.md"),
        "# Ownership\n\nChaque valeur a un propriétaire unique.",
    )
    .expect("write");
    std::fs::write(
        docs.join("traits.md"),
        "# Traits\n\nUn trait décrit un comportement commun aux types.",
    )
    .expect("write");
    let llm = Arc::new(MockLlmClient::new([
        MockReply::tokens(&["Un trait décrit un comportement commun [1]."]),
        MockReply::tokens(&["Sans documents."]),
    ]));
    let mut h = Harness::with_database_and_llm(&dir.path().join("db.sqlite"), llm.clone());
    h.command(&format!("/index {}", docs.display()));
    h.run_until_indexed().await;

    h.command("/rag inconnue");
    assert!(
        matches!(&h.app.status, Status::Error(m) if m.contains("introuvable (collections : cours)"))
    );
    h.command("/rag cours");
    assert_eq!(h.app.rag_collection.as_deref(), Some("cours"));

    h.send("Qu'est-ce qu'un trait, ce comportement commun aux types ?");
    h.run_until_idle().await;
    let request = llm.requests().pop().expect("request");
    let system = &request.messages[0].content;
    assert!(system.contains("[1] traits.md § Traits\n# Traits\n\nUn trait décrit"));
    assert!(
        !request.messages[0].content.contains("propriétaire unique"),
        "top_k = 1"
    );
    let reply = h.app.conversation.messages().last().expect("reply").clone();
    assert_eq!(reply.citations.len(), 1);
    assert_eq!(reply.citations[0].label(), "traits.md § Traits");

    // Reloaded from the database: the sources and the collection are back.
    h.reload();
    assert_eq!(h.app.conversation.messages().last(), Some(&reply));
    assert_eq!(h.app.rag_collection.as_deref(), Some("cours"));

    h.command("/rag off");
    h.send("Et sinon ?");
    h.run_until_idle().await;
    let request = llm.requests().pop().expect("request");
    assert!(!request.messages[0].content.contains("traits.md"));
    assert!(
        h.app
            .conversation
            .messages()
            .last()
            .expect("reply")
            .citations
            .is_empty()
    );
}

#[tokio::test]
async fn keywords_find_codes_changes_are_noticed_and_forget_deletes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let docs = dir.path().join("support");
    std::fs::create_dir(&docs).expect("mkdir");
    std::fs::write(
        docs.join("tickets.md"),
        "# Tickets\n\nLe module XK-42 gère les tickets.",
    )
    .expect("write");
    std::fs::write(
        docs.join("accueil.md"),
        "# Accueil\n\nBienvenue sur la plateforme, voici comment démarrer.",
    )
    .expect("write");
    let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&["Voir [1]."])]));
    let mut h = Harness::with_database_and_llm(&dir.path().join("db.sqlite"), llm.clone());
    h.command(&format!("/index {}", docs.display()));
    h.run_until_indexed().await;
    h.command("/rag support");

    // Hash embeddings know nothing of "XK-42": the keyword search finds it.
    h.send("XK-42 ?");
    h.run_until_idle().await;
    let request = llm.requests().pop().expect("request");
    assert!(
        request.messages[0]
            .content
            .contains("[1] tickets.md § Tickets")
    );

    // A file changes: the startup check notices it.
    std::fs::write(docs.join("nouveau.md"), "# Nouveau\n\ntexte").expect("write");
    h.app.status = Status::Ready;
    h.dispatch(Action::Init);
    assert!(
        matches!(&h.app.status, Status::Info(m) if m == "« support » : 1 nouveau fichier (/index support pour mettre à jour)"),
        "{:?}",
        h.app.status
    );
    // Updating by name clears it.
    h.command("/index support");
    h.run_until_indexed().await;
    assert_eq!(h.app.last_index.as_ref().map(|r| r.added), Some(1));
    assert!(h.app.stale.is_empty());

    h.command("/forget support");
    h.dispatch(Action::Submit);
    assert_eq!(h.app.rag_collection, None);
    h.command("/collections");
    assert_eq!(h.app.collections.as_ref().map(|c| c.0.len()), Some(0));
}

#[tokio::test]
async fn the_model_reads_a_file_once_allowed() {
    use chatatui::{llm::ToolCall, state::Overlay};
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("plan.md");
    std::fs::write(&file, "Séance 3 : les traits.").expect("write");
    let call = ToolCall {
        id: "call_1".into(),
        name: "read_file".into(),
        arguments: serde_json::json!({ "path": file }).to_string(),
    };
    let llm = Arc::new(MockLlmClient::new([
        MockReply::ToolCalls(vec!["Je regarde.".into()], vec![call.clone()]),
        MockReply::tokens(&["La séance 3 porte sur les traits."]),
        MockReply::ToolCalls(
            Vec::new(),
            vec![ToolCall {
                id: "call_2".into(),
                ..call
            }],
        ),
        MockReply::tokens(&["Sans le fichier, je ne sais pas."]),
    ]));
    let mut h = Harness::new(llm.clone());
    h.command("/tools on");
    assert!(h.app.tools_enabled);

    h.send("Que dit mon plan ?");
    while !matches!(h.app.overlay, Some(Overlay::ToolConfirm { .. })) {
        h.step().await;
    }
    assert!(!llm.requests()[0].tools.is_empty(), "tools offered");
    h.dispatch(Action::ToolAnswer {
        allow: true,
        always: false,
    });
    h.run_until_idle().await;

    let roles: Vec<Role> = h
        .app
        .conversation
        .messages()
        .iter()
        .map(|m| m.role)
        .collect();
    assert_eq!(
        roles,
        vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
    let tool = &h.app.conversation.messages()[2];
    assert_eq!(tool.content, "Séance 3 : les traits.");
    assert!(
        tool.source
            .as_deref()
            .is_some_and(|s| s.starts_with("lire "))
    );
    let second = &llm.requests()[1];
    let last = second.messages.last().expect("tool result");
    assert_eq!(last.tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(last.content, "Séance 3 : les traits.");

    // Refused: the model is told and answers without it.
    h.send("Et la séance 4 ?");
    while !matches!(h.app.overlay, Some(Overlay::ToolConfirm { .. })) {
        h.step().await;
    }
    h.dispatch(Action::ToolAnswer {
        allow: false,
        always: false,
    });
    // Review Focus 4 / Task 2's guarantee: a refused call never sends `Phase::RunningTool`,
    // so the waiting line must never show the tool as running while this plays out.
    while h.app.is_generating() {
        h.step().await;
        assert!(
            !matches!(
                h.app.waiting().map(|w| w.phase),
                Some(chatatui::llm::Phase::RunningTool { .. })
            ),
            "a refused tool call must never show as running"
        );
    }
    let refused = &llm.requests()[3];
    assert!(
        refused
            .messages
            .last()
            .expect("result")
            .content
            .contains("refused")
    );
    let messages = h.app.conversation.messages();
    assert!(matches!(
        messages[messages.len() - 2].status,
        MessageStatus::Failed(_)
    ));
    // The first file read is replayed as plain text in later requests.
    assert!(
        refused
            .messages
            .iter()
            .any(|m| m.content.starts_with("[outil : lire "))
    );
}

/// Types `text` into the input and submits it, returning the effects it produced.
fn submit(app: &mut App, text: &str) -> Vec<Effect> {
    for c in text.chars() {
        app.update(Action::Edit(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
    app.update(Action::Submit)
}

#[tokio::test]
async fn pull_lists_the_files_then_downloads_the_chosen_one() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    let effects = submit(&mut harness.app, "/pull owner/name");

    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::ListGguf { repo, .. } if repo == "owner/name")),
        "{effects:?}"
    );

    let files = vec![chatatui::models::hub::RemoteFile {
        path: "m-Q4_K_M.gguf".to_owned(),
        bytes: 4_000_000_000,
        sha256: Some("abc".to_owned()),
    }];
    harness.app.update(Action::GgufFiles {
        repo: "owner/name".to_owned(),
        result: Ok(files),
    });

    assert!(matches!(harness.app.overlay, Some(Overlay::GgufPicker(_))));

    let effects = harness.app.update(Action::OverlaySelect);

    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::StartPull { repo, file, .. }
                if repo == "owner/name" && file.path == "m-Q4_K_M.gguf"
        )),
        "{effects:?}"
    );
    assert!(harness.app.overlay.is_none(), "the picker closes on Enter");
}

/// A pasted `/tree/<rev>` URL must download that revision, not `main` recorded as `main`.
#[tokio::test]
async fn a_pasted_revision_is_carried_to_the_download() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    let effects = submit(
        &mut harness.app,
        "/pull https://huggingface.co/owner/name/tree/v2.0",
    );

    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::ListGguf { repo, revision }
                if repo == "owner/name" && revision.as_deref() == Some("v2.0")
        )),
        "{effects:?}"
    );

    harness.app.update(Action::GgufFiles {
        repo: "owner/name".to_owned(),
        result: Ok(vec![chatatui::models::hub::RemoteFile {
            path: "m-Q4_K_M.gguf".to_owned(),
            bytes: 10,
            sha256: None,
        }]),
    });
    let effects = harness.app.update(Action::OverlaySelect);

    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::StartPull { revision, .. } if revision.as_deref() == Some("v2.0")
        )),
        "{effects:?}"
    );
}

/// Without a revision in the URL, the runtime's default is used.
#[tokio::test]
async fn a_plain_repository_names_no_revision() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    let effects = submit(&mut harness.app, "/pull owner/name");

    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::ListGguf { revision, .. } if revision.is_none())),
        "{effects:?}"
    );
}

#[tokio::test]
async fn a_repository_without_gguf_says_so_instead_of_opening_an_empty_picker() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    submit(&mut harness.app, "/pull owner/name");
    harness.app.update(Action::GgufFiles {
        repo: "owner/name".to_owned(),
        result: Ok(Vec::new()),
    });

    assert!(harness.app.overlay.is_none());
    assert!(
        matches!(&harness.app.status, Status::Error(message) if message.contains("gguf")
            || message.contains("GGUF")),
        "{:?}",
        harness.app.status
    );
}

#[tokio::test]
async fn a_second_pull_is_refused_while_one_runs() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    harness.app.update(Action::Pull(
        chatatui::models::download::PullEvent::Progress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 10,
            total: Some(100),
            rate: 5,
        },
    ));

    let effects = submit(&mut harness.app, "/pull other/name");

    assert!(
        !effects.iter().any(|e| matches!(e, Effect::ListGguf { .. })),
        "{effects:?}"
    );
    assert!(matches!(harness.app.status, Status::Error(_)));
}

/// The picker is a second way into the download, and it must be refused just like a second
/// `/pull`: starting one leaves the running job's cancellation token with no owner, so Esc
/// and quitting stop nothing.
#[tokio::test]
async fn the_picker_cannot_start_a_second_download() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    // A listing asked for before the running download arrives late and opens the picker.
    submit(&mut harness.app, "/pull other/name");
    harness.app.update(Action::Pull(
        chatatui::models::download::PullEvent::Progress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 10,
            total: Some(100),
            rate: 5,
        },
    ));
    harness.app.update(Action::GgufFiles {
        repo: "other/name".to_owned(),
        result: Ok(vec![chatatui::models::hub::RemoteFile {
            path: "other-Q4_K_M.gguf".to_owned(),
            bytes: 10,
            sha256: None,
        }]),
    });
    assert!(matches!(harness.app.overlay, Some(Overlay::GgufPicker(_))));

    let effects = harness.app.update(Action::OverlaySelect);

    assert!(
        !effects
            .iter()
            .any(|e| matches!(e, Effect::StartPull { .. })),
        "{effects:?}"
    );
    assert!(matches!(harness.app.status, Status::Error(_)));
    // The running download is untouched.
    assert_eq!(
        harness.app.pulling.as_ref().map(|p| p.file.clone()),
        Some("m.gguf".to_owned())
    );
}

#[tokio::test]
async fn esc_cancels_the_download_when_no_overlay_is_open() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    harness.app.update(Action::Pull(
        chatatui::models::download::PullEvent::Progress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 10,
            total: Some(100),
            rate: 5,
        },
    ));

    let effects = harness.app.update(Action::Cancel);

    assert!(
        effects.iter().any(|e| matches!(e, Effect::CancelPull)),
        "{effects:?}"
    );
}

#[tokio::test]
async fn a_finished_pull_records_the_model_and_refreshes_the_list() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    let model = chatatui::models::store::LocalModel {
        repo: "owner/name".to_owned(),
        revision: "main".to_owned(),
        file: "m-Q4_K_M.gguf".to_owned(),
        path: "/models/owner/name/m-Q4_K_M.gguf".to_owned(),
        bytes: 4_000_000_000,
        sha256: Some("abc".to_owned()),
        architecture: Some("qwen2".to_owned()),
        quantization: Some("Q4_K_M".to_owned()),
        context_length: Some(32768),
        parameters: Some(7_615_616_512),
        downloaded_at: 1_760_000_000,
    };

    let effects = harness.app.update(Action::Pull(
        chatatui::models::download::PullEvent::Finished(Box::new(model)),
    ));

    assert!(harness.app.pulling.is_none());
    assert!(matches!(harness.app.status, Status::Info(_)));
    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::Store(chatatui::storage::StoreRequest::ListModels)
        )),
        "{effects:?}"
    );
}

/// A downloaded model and the catalogue share one list, and `Entrée` on an entry that is
/// not on disk asks the Hub for its files — the same path `/pull <dépôt>` takes.
#[tokio::test]
async fn models_offers_the_catalog_and_enter_starts_a_download() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    let effects = submit(&mut harness.app, "/models");
    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::Store(chatatui::storage::StoreRequest::ListModels)
        )),
        "{effects:?}"
    );
    let Some(Overlay::Models(picker)) = &harness.app.overlay else {
        panic!(
            "the /models popup must be a list: {:?}",
            harness.app.overlay
        );
    };
    // Nothing downloaded yet, so every row is an offer.
    assert_eq!(
        picker.rows.len(),
        chatatui::models::catalog::entries().len()
    );

    let effects = harness.app.update(Action::OverlaySelect);
    let wanted = chatatui::models::catalog::entries()[0].repo;
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::ListGguf { repo, .. } if repo == wanted)),
        "{effects:?}"
    );
    assert!(harness.app.overlay.is_none(), "the list closes on Enter");
}

/// The inventory arrives after the popup opens, so the open list must take it.
#[tokio::test]
async fn the_open_models_list_takes_the_inventory_when_it_arrives() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    submit(&mut harness.app, "/models");

    harness
        .app
        .update(Action::Storage(chatatui::storage::StoreEvent::Models {
            models: vec![local_model("someone/private-GGUF", "x.gguf")],
            now: 2_000,
        }));

    let Some(Overlay::Models(picker)) = &harness.app.overlay else {
        panic!("the popup must still be open");
    };
    assert!(picker.loaded);
    assert_eq!(picker.rows[0].repo(), "someone/private-GGUF");
}

/// `Suppr` deletes the highlighted model, and says so when there is nothing to delete.
#[tokio::test]
async fn suppr_deletes_a_downloaded_model_and_refuses_an_offer() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    submit(&mut harness.app, "/models");
    harness
        .app
        .update(Action::Storage(chatatui::storage::StoreEvent::Models {
            models: vec![local_model("someone/private-GGUF", "x.gguf")],
            now: 2_000,
        }));

    let effects = harness.app.update(Action::OverlayDelete);
    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::Store(chatatui::storage::StoreRequest::DeleteModel { repo, file })
                if repo == "someone/private-GGUF" && file == "x.gguf"
        )),
        "{effects:?}"
    );

    // The second row is an offer: nothing on disk, so nothing to delete.
    harness.app.update(Action::OverlayDown);
    let effects = harness.app.update(Action::OverlayDelete);
    assert!(effects.is_empty(), "{effects:?}");
    let Some(Overlay::Models(picker)) = &harness.app.overlay else {
        panic!("the popup must still be open");
    };
    assert!(
        picker
            .message
            .as_deref()
            .is_some_and(|m| m.contains("n'est pas téléchargé")),
        "{:?}",
        picker.message
    );
}

/// `Entrée` means download on every row, including one already on disk: that is the only
/// way to fetch a second quantization of a repository, since the catalogue entry is hidden
/// once anything from that repository has landed.
#[tokio::test]
async fn enter_on_a_downloaded_model_offers_another_quantization() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    submit(&mut harness.app, "/models");
    harness
        .app
        .update(Action::Storage(chatatui::storage::StoreEvent::Models {
            models: vec![local_model("someone/private-GGUF", "x.gguf")],
            now: 2_000,
        }));

    let effects = harness.app.update(Action::OverlaySelect);

    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::ListGguf { repo, .. } if repo == "someone/private-GGUF")),
        "{effects:?}"
    );
    assert!(harness.app.overlay.is_none(), "the list closes on Enter");
}

/// A refused download must not cost the user the open list: the guard is consulted before
/// the popup is closed, and the refusal is shown in the popup, where it can be read.
#[tokio::test]
async fn enter_in_the_models_list_is_refused_while_a_download_runs() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    harness.app.update(Action::Pull(
        chatatui::models::download::PullEvent::Progress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 10,
            total: Some(100),
            rate: 5,
        },
    ));
    submit(&mut harness.app, "/models");

    let effects = harness.app.update(Action::OverlaySelect);

    assert!(
        !effects.iter().any(|e| matches!(e, Effect::ListGguf { .. })),
        "{effects:?}"
    );
    let Some(Overlay::Models(picker)) = &harness.app.overlay else {
        panic!("the list must stay open when nothing was started");
    };
    assert!(
        picker
            .message
            .as_deref()
            .is_some_and(|m| m.contains("déjà en cours")),
        "{:?}",
        picker.message
    );
}

fn local_model(repo: &str, file: &str) -> chatatui::models::store::LocalModel {
    chatatui::models::store::LocalModel {
        repo: repo.to_owned(),
        revision: "main".to_owned(),
        file: file.to_owned(),
        path: format!("/models/{repo}/{file}"),
        bytes: 400_000_000,
        sha256: Some("abc".to_owned()),
        architecture: Some("qwen3".to_owned()),
        quantization: Some("Q4_K_M".to_owned()),
        context_length: Some(40_960),
        parameters: Some(596_000_000),
        downloaded_at: 1_000,
    }
}
