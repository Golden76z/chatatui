//! Actions (inputs to [`App::update`](crate::app::App::update)) and effects (its outputs).

use crossterm::event::KeyEvent;

use crate::{
    files::Attachment,
    llm::{LlmEvent, ProviderModels, RequestId, stream_task::CompletionJob},
    rag::indexer::{IndexEvent, Staleness},
    storage::{StoreEvent, StoreRequest},
};

/// Something that changes the application state.
///
/// Actions come from the keymap (user intents) and from background tasks (LLM events, …).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Leave the application.
    Quit,
    /// Send the current input as a user message.
    Submit,
    /// Insert a line break in the input.
    InsertNewline,
    /// `Esc`: close the topmost overlay or cancel the running generation.
    Cancel,
    /// Start a new, empty conversation.
    NewConversation,
    /// Open or close the conversation list.
    ToggleSidebar,
    /// Highlight the previous conversation in the list.
    SidebarUp,
    /// Highlight the next conversation in the list.
    SidebarDown,
    /// Open the highlighted conversation.
    SidebarOpen,
    /// Open the model selection popup.
    OpenModelPicker,
    /// Open the command palette.
    OpenPalette,
    /// Open the help screen.
    OpenHelp,
    /// Move up in the open popup (previous item, or scroll).
    OverlayUp,
    /// Move down in the open popup.
    OverlayDown,
    /// Scroll a text popup up by a page.
    OverlayPageUp,
    /// Scroll a text popup down by a page.
    OverlayPageDown,
    /// Choose the highlighted item of the open popup.
    OverlaySelect,
    /// Type a character in the popup filter.
    OverlayFilter(char),
    /// Delete the last character of the popup filter.
    OverlayBackspace,
    /// Highlight the previous slash-command suggestion.
    SuggestionUp,
    /// Highlight the next slash-command suggestion.
    SuggestionDown,
    /// Complete the input with the highlighted suggestion (Tab).
    CompleteSuggestion,
    /// Hide the suggestions until the input changes.
    DismissSuggestions,
    /// Tab after `/add `: complete the file path.
    CompletePath,
    /// Candidates for the path being completed.
    PathCompleted {
        partial: String,
        candidates: Vec<String>,
    },
    /// A file requested with `/add` was read (or could not be).
    FileRead(Result<Attachment, String>),
    /// Progress of the indexing job.
    Index(IndexEvent),
    /// How the collections' folders differ from their index.
    CollectionsChecked(Result<Vec<Staleness>, String>),
    /// A key that is not a shortcut, forwarded to the text input.
    Edit(KeyEvent),
    /// Text pasted by the terminal (bracketed paste); inserted verbatim.
    Paste(String),
    /// Scroll the conversation up by this many lines.
    ScrollUp(usize),
    /// Scroll the conversation down by this many lines.
    ScrollDown(usize),
    /// Scroll up by one screen.
    PageUp,
    /// Scroll down by one screen.
    PageDown,
    /// Jump to the first message.
    ScrollToTop,
    /// Jump to the last message and follow new content.
    ScrollToBottom,
    /// The terminal was resized.
    Resize { width: u16, height: u16 },
    /// Periodic tick: refreshes what changed since the last one (e.g. streamed tokens).
    Tick,
    /// The runtime has started (first action, once).
    Init,
    /// Context window of a model, as detected by its server.
    ContextWindowDetected {
        provider: String,
        model: String,
        tokens: Option<u64>,
    },
    /// Progress of a streamed reply.
    Llm {
        request_id: RequestId,
        event: LlmEvent,
    },
    /// Result from the storage worker.
    Storage(StoreEvent),
    /// Model lists of every provider.
    ModelsListed(Vec<ProviderModels>),
}

/// A side effect requested by [`App::update`](crate::app::App::update) and executed by the
/// [`Runtime`](crate::runtime::Runtime).
///
/// Keeping I/O out of `App` makes the state logic testable without a network or a database.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    /// Spawn a streaming task for this job.
    StartCompletion(CompletionJob),
    /// Abort the streaming task of this request.
    CancelCompletion(RequestId),
    /// Send a request to the storage worker.
    Store(StoreRequest),
    /// Fetch the models offered by the server.
    ListModels,
    /// Ask a provider for the context window of a model.
    DetectContextWindow { provider: String, model: String },
    /// Read a file to attach.
    ReadFile(String),
    /// List the completions of a partial path.
    CompletePath(String),
    /// Index a folder (`root` as typed; `~` allowed, or a collection name to update it)
    /// into a collection, limited to `types` (`None`: the collection's current choice).
    StartIndex {
        collection: String,
        root: String,
        types: Option<Vec<String>>,
    },
    /// Stop the running indexing job.
    CancelIndex,
    /// Compare the collections' folders with their index.
    CheckCollections,
}
