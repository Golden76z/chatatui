//! The main loop: draws, waits for events, turns them into actions and executes effects.

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use color_eyre::Result;
use crossterm::event::{Event as TermEvent, KeyEventKind, MouseEventKind};
use ratatui::DefaultTerminal;
use tokio_util::sync::CancellationToken;

use crate::{
    action::{Action, Effect},
    app::App,
    config::Config,
    event::{AppEvent, Event, EventHandler},
    files, keymap,
    llm::{self, ProviderModels, RequestId, stream_task},
    models::{
        download::{self, PullEvent, PullRequest},
        hub::{Hub, RemoteFile},
    },
    rag::{
        embed::{Embedder, OpenAiEmbedder},
        indexer::{self, IndexEvent, IndexRequest},
        ocr::Ocr,
        retrieve::{RagContext, Selection},
    },
    storage::worker::{Location, StoreHandle},
    ui,
};

/// How long to wait for a provider's model list.
const MODEL_LIST_TIMEOUT: Duration = Duration::from_secs(15);

/// Lines scrolled per mouse wheel notch.
const MOUSE_SCROLL_LINES: usize = 3;

/// Branch a model is downloaded from; the Hub's default.
const DEFAULT_REVISION: &str = "main";

/// Owns the state, the event source and the backends; the only place where side effects run.
#[derive(Debug)]
pub struct Runtime {
    app: App,
    events: EventHandler,
    backends: stream_task::Backends,
    /// Provider ids in display order (for the model list).
    provider_order: Vec<String>,
    /// Cancellation handle of the running streaming task, and where its tool decisions go.
    running_task: Option<(
        RequestId,
        CancellationToken,
        tokio::sync::mpsc::UnboundedSender<bool>,
    )>,
    /// Storage worker; `None` once shut down.
    store: Option<StoreHandle>,
    /// What `/index` needs.
    rag: RagBackend,
    /// Cancellation handle of the running indexing job.
    running_index: Option<CancellationToken>,
    /// What `/pull` needs.
    models: ModelsBackend,
    /// Cancellation handle of the running download.
    running_pull: Option<CancellationToken>,
    /// Watch over the collections' folders (`[rag] auto_index`).
    watch: Option<crate::rag::watch::Watch>,
    /// MCP servers to start.
    mcp: Vec<(String, crate::config::McpServerConfig)>,
}

/// The embedding backend and database file used by indexing, or why indexing is unavailable.
struct RagBackend {
    embedder: Result<Arc<dyn Embedder>, String>,
    database: Result<PathBuf, String>,
    chunk_tokens: usize,
    exclude: Vec<String>,
    /// Watch the collections' folders while running.
    auto_index: bool,
    /// OCR for scanned PDFs (`None`: turned off or tools missing).
    ocr: Option<Ocr>,
}

impl fmt::Debug for RagBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RagBackend")
            .field("embedder", &self.embedder.as_ref().map(|e| e.model()))
            .field("database", &self.database)
            .field("chunk_tokens", &self.chunk_tokens)
            .finish()
    }
}

impl RagBackend {
    fn new(config: &Config, providers: &[crate::config::Provider], database: &Location) -> Self {
        let rag = &config.rag;
        let embedder = providers
            .iter()
            .find(|p| p.id == rag.embedding_provider)
            .ok_or_else(|| {
                format!(
                    "fournisseur d'embeddings « {} » inconnu (section [rag])",
                    rag.embedding_provider
                )
            })
            .and_then(|provider| {
                OpenAiEmbedder::new(
                    provider,
                    &rag.embedding_model,
                    Duration::from_secs(config.connect_timeout_secs),
                )
                .map_err(|e| e.to_string())
            })
            .map(|e| Arc::new(e) as Arc<dyn Embedder>);
        let database = match database {
            Location::File(path) => Ok(path.clone()),
            Location::Memory => Err("index indisponible (base en mémoire)".to_owned()),
        };
        Self {
            embedder,
            database,
            chunk_tokens: rag.chunk_tokens,
            exclude: rag.exclude.clone(),
            auto_index: rag.auto_index,
            ocr: if rag.ocr {
                Ocr::detect(&rag.ocr_languages)
            } else {
                None
            },
        }
    }
}

/// The HuggingFace client and the store directory used by `/pull`, or why it is
/// unavailable.
struct ModelsBackend {
    hub: Result<Arc<Hub>, String>,
    /// Root of the model store.
    dir: PathBuf,
    database: Result<PathBuf, String>,
}

impl fmt::Debug for ModelsBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print the token: the `Hub`'s own `Debug` only says whether it has one.
        f.debug_struct("ModelsBackend")
            .field("hub", &self.hub)
            .field("dir", &self.dir)
            .field("database", &self.database)
            .finish()
    }
}

impl ModelsBackend {
    fn new(config: &Config, database: &Location) -> Self {
        let token = std::env::var(&config.models.token_env).ok();
        let hub = Hub::new(
            None,
            token,
            Duration::from_secs(config.connect_timeout_secs),
        )
        .map(Arc::new)
        .map_err(|e| e.to_string());
        let database = match database {
            Location::File(path) => Ok(path.clone()),
            Location::Memory => Err("téléchargement indisponible (base en mémoire)".to_owned()),
        };
        let configured = config.models.dir.trim();
        let dir = if configured.is_empty() {
            // Next to the database, so the store travels with the history.
            database
                .as_ref()
                .ok()
                .and_then(|path| path.parent().map(Path::to_path_buf))
                .unwrap_or_default()
                .join("models")
        } else {
            files::expand_home(configured)
        };
        Self { hub, dir, database }
    }
}

impl Runtime {
    /// Creates the runtime with the configured backend, starts the storage worker (history in
    /// `database`) and spawns the terminal event task.
    pub fn new(config: Config, keyboard_enhanced: bool, database: Location) -> Result<Self> {
        let providers = config.resolve_providers(|name| std::env::var(name).ok());
        let clients =
            llm::build_clients(&providers, Duration::from_secs(config.connect_timeout_secs));
        let provider_order = providers.iter().map(|p| p.id.clone()).collect();
        let rag = RagBackend::new(&config, &providers, &database);
        let models = ModelsBackend::new(&config, &database);
        let mut context = RagContext::new(
            rag.embedder.clone(),
            rag.database.clone(),
            Selection::from(&config.rag),
        );
        if let Some(reranker) = build_reranker(&config, &providers) {
            context = context.with_reranker(reranker, config.rag.rerank_candidates);
        }
        let backends = stream_task::Backends {
            clients,
            context: Arc::new(context),
            mcp: Arc::default(),
        };
        let events = EventHandler::new();
        let sender = events.sender();
        let store = StoreHandle::spawn(database, move |event| {
            // Fails only while shutting down.
            let _ = sender.send(Event::App(AppEvent::Storage(event)));
        });
        Ok(Self {
            app: App::new(&config, keyboard_enhanced).with_session_seed(session_seed()),
            events,
            backends,
            provider_order,
            running_task: None,
            store: Some(store),
            rag,
            running_index: None,
            models,
            running_pull: None,
            watch: None,
            mcp: config
                .mcp
                .iter()
                .filter(|(_, server)| server.enabled)
                .map(|(name, server)| (name.clone(), server.clone()))
                .collect(),
        })
    }

    /// Runs until the user quits.
    pub async fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        let size = terminal.size()?;
        self.dispatch(Action::Resize {
            width: size.width,
            height: size.height,
        });
        self.dispatch(Action::Init);
        self.refresh_watch();
        self.start_mcp();
        let mut needs_redraw = true;
        while self.app.running {
            if needs_redraw {
                terminal.draw(|frame| ui::render(&self.app, frame))?;
            }
            // Wait for one event, then drain whatever is already queued (typing bursts,
            // token bursts) so we draw once per batch.
            let event = self.events.next().await?;
            needs_redraw = self.handle(event);
            while self.app.running
                && let Some(event) = self.events.try_next()
            {
                needs_redraw |= self.handle(event);
            }
        }
        if let Some(token) = self.running_index.take() {
            token.cancel();
        }
        if let Some(token) = self.running_pull.take() {
            token.cancel();
        }
        // Let the storage thread write what is still queued (e.g. a reply cut by quitting).
        if let Some(store) = self.store.take() {
            store.shutdown();
        }
        Ok(())
    }

    /// Handles one event. Returns `true` if the screen must be redrawn.
    fn handle(&mut self, event: Event) -> bool {
        let action = match event {
            Event::Tick => {
                // Redraw only if the tick actually re-rendered something (streamed tokens).
                let before = self.app.transcript.revision();
                self.dispatch(Action::Tick);
                return self.app.transcript.revision() != before;
            }
            Event::Crossterm(TermEvent::Key(key)) if key.kind == KeyEventKind::Press => {
                keymap::map_key(key, self.app.key_context())
            }
            Event::Crossterm(TermEvent::Paste(text)) => Some(Action::Paste(text)),
            Event::Crossterm(TermEvent::Resize(width, height)) => {
                Some(Action::Resize { width, height })
            }
            Event::Crossterm(TermEvent::Mouse(mouse)) => match mouse.kind {
                MouseEventKind::ScrollUp => Some(Action::ScrollUp(MOUSE_SCROLL_LINES)),
                MouseEventKind::ScrollDown => Some(Action::ScrollDown(MOUSE_SCROLL_LINES)),
                _ => return false,
            },
            Event::Crossterm(_) => return false,
            Event::App(AppEvent::Llm { request_id, event }) => {
                Some(Action::Llm { request_id, event })
            }
            Event::App(AppEvent::Storage(event)) => {
                if matches!(event, crate::storage::StoreEvent::CollectionDeleted { .. }) {
                    self.refresh_watch();
                }
                Some(Action::Storage(event))
            }
            Event::App(AppEvent::Mcp(status)) => Some(Action::Mcp(status)),
            Event::App(AppEvent::CollectionsChanged(names)) => {
                Some(Action::CollectionsChanged(names))
            }
            Event::App(AppEvent::CollectionsChecked(result)) => {
                Some(Action::CollectionsChecked(result))
            }
            Event::App(AppEvent::Index(event)) => {
                if matches!(
                    event,
                    IndexEvent::Finished(_)
                        | IndexEvent::Failed { .. }
                        | IndexEvent::Cancelled { .. }
                ) {
                    self.running_index = None;
                }
                if matches!(event, IndexEvent::Finished(_)) {
                    // A new collection, or a moved folder.
                    self.refresh_watch();
                }
                Some(Action::Index(event))
            }
            Event::App(AppEvent::GgufFiles { repo, result }) => {
                Some(Action::GgufFiles { repo, result })
            }
            Event::App(AppEvent::Pull(event)) => {
                if matches!(
                    event,
                    PullEvent::Finished(_) | PullEvent::Failed { .. } | PullEvent::Cancelled { .. }
                ) {
                    self.running_pull = None;
                }
                Some(Action::Pull(event))
            }
            Event::App(AppEvent::Models(result)) => Some(Action::ModelsListed(result)),
            Event::App(AppEvent::FileRead(result)) => Some(Action::FileRead(result)),
            Event::App(AppEvent::Exported(result)) => Some(Action::Exported(result)),
            Event::App(AppEvent::Copied { what, chars, how }) => {
                Some(Action::Copied { what, chars, how })
            }
            Event::App(AppEvent::PathCompletions {
                partial,
                candidates,
            }) => Some(Action::PathCompleted {
                partial,
                candidates,
            }),
            Event::App(AppEvent::ContextWindow {
                provider,
                model,
                tokens,
            }) => Some(Action::ContextWindowDetected {
                provider,
                model,
                tokens,
            }),
        };
        if let Some(action) = action {
            self.dispatch(action);
        }
        true
    }

    /// Updates the app and runs the resulting effects.
    fn dispatch(&mut self, action: Action) {
        for effect in self.app.update(action) {
            self.execute(effect);
        }
    }

    /// Runs a side effect requested by the app.
    fn execute(&mut self, effect: Effect) {
        match effect {
            Effect::StartCompletion(job) => {
                // Only one generation at a time: stop any leftover task first.
                if let Some((_, token, _)) = self.running_task.take() {
                    token.cancel();
                }
                let token = CancellationToken::new();
                let (decisions, answers) = tokio::sync::mpsc::unbounded_channel();
                self.running_task = Some((job.request_id, token.clone(), decisions));
                tokio::spawn(stream_task::run_with_tools(
                    self.backends.clone(),
                    job,
                    token,
                    self.events.sender(),
                    Some(answers),
                ));
            }
            Effect::ListModels => {
                // Ask every provider at once; a slow or failing one does not hide the others.
                let requests: Vec<_> = self
                    .provider_order
                    .iter()
                    .filter_map(|id| {
                        let client = self.backends.clients.get(id)?.clone();
                        let id = id.clone();
                        Some(async move {
                            let result =
                                tokio::time::timeout(MODEL_LIST_TIMEOUT, client.list_models())
                                    .await
                                    .unwrap_or_else(|_| {
                                        Err(llm::LlmError::Protocol(
                                            "pas de réponse (délai dépassé)".into(),
                                        ))
                                    })
                                    .map_err(|e| e.to_string());
                            ProviderModels {
                                provider: id,
                                result,
                            }
                        })
                    })
                    .collect();
                let sender = self.events.sender();
                tokio::spawn(async move {
                    let results = futures::future::join_all(requests).await;
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::Models(results)));
                });
            }
            Effect::ReadFile(path)
                if path.starts_with("http://") || path.starts_with("https://") =>
            {
                let sender = self.events.sender();
                tokio::spawn(async move {
                    let result = crate::web::fetch(&path)
                        .await
                        .map(|page| files::Attachment {
                            content: match page.title {
                                Some(title) => format!("# {title}\n\n{}", page.text),
                                None => page.text,
                            },
                            source: path,
                            image: None,
                        });
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::FileRead(result)));
                });
            }
            Effect::ReadFile(path) => {
                let sender = self.events.sender();
                tokio::task::spawn_blocking(move || {
                    let result = files::read_attachment(&path);
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::FileRead(result)));
                });
            }
            Effect::Export {
                path,
                suggested,
                content,
            } => {
                let sender = self.events.sender();
                tokio::task::spawn_blocking(move || {
                    let result = crate::files::write_new(path.as_deref(), &suggested, &content);
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::Exported(result)));
                });
            }
            Effect::Copy { text, what } => {
                // OSC 52 goes to the terminal right away, between two frames.
                let _ = crate::clipboard::osc52(&mut std::io::stdout(), &text);
                let sender = self.events.sender();
                tokio::task::spawn_blocking(move || {
                    let how = crate::clipboard::copy_with_tool(&text);
                    let chars = text.chars().count();
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::Copied { what, chars, how }));
                });
            }
            Effect::CompletePath(partial) => {
                let sender = self.events.sender();
                tokio::task::spawn_blocking(move || {
                    let candidates = files::complete_path(&partial);
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::PathCompletions {
                        partial,
                        candidates,
                    }));
                });
            }
            Effect::DetectContextWindow { provider, model } => {
                let Some(client) = self.backends.clients.get(&provider).cloned() else {
                    return;
                };
                let sender = self.events.sender();
                tokio::spawn(async move {
                    let tokens = client.context_window(&model).await;
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::ContextWindow {
                        provider,
                        model,
                        tokens,
                    }));
                });
            }
            Effect::Store(request) => {
                if let Some(store) = &self.store {
                    store.send(request);
                }
            }
            Effect::StartIndex {
                collection,
                root,
                types,
            } => self.start_index(collection, &root, types),
            Effect::CheckCollections => {
                let Ok(database) = self.rag.database.clone() else {
                    return;
                };
                let exclude = self.rag.exclude.clone();
                let sender = self.events.sender();
                tokio::task::spawn_blocking(move || {
                    let result = indexer::check_collections(&database, &exclude);
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::CollectionsChecked(result)));
                });
            }
            Effect::CancelIndex => {
                if let Some(token) = &self.running_index {
                    token.cancel();
                }
            }
            Effect::ListGguf { repo, revision } => {
                let sender = self.events.sender();
                let hub = match &self.models.hub {
                    Ok(hub) => Arc::clone(hub),
                    Err(error) => {
                        // Fails only while shutting down.
                        let _ = sender.send(Event::App(AppEvent::GgufFiles {
                            repo,
                            result: Err(error.clone()),
                        }));
                        return;
                    }
                };
                let revision = revision.unwrap_or_else(|| DEFAULT_REVISION.to_owned());
                tokio::spawn(async move {
                    let result = hub
                        .list_gguf(&repo, &revision)
                        .await
                        .map_err(|e| e.to_string());
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::GgufFiles { repo, result }));
                });
            }
            Effect::StartPull {
                repo,
                revision,
                file,
            } => self.start_pull(repo, revision, file),
            Effect::CancelPull => {
                if let Some(token) = &self.running_pull {
                    token.cancel();
                }
            }
            Effect::CancelCompletion(request_id) => {
                if let Some((id, token, decisions)) = self.running_task.take() {
                    if id == request_id {
                        token.cancel();
                    } else {
                        self.running_task = Some((id, token, decisions));
                    }
                }
            }
            Effect::ToolDecision { request_id, allow } => {
                if let Some((id, _, decisions)) = &self.running_task
                    && *id == request_id
                {
                    // Fails only if the task already ended.
                    let _ = decisions.send(allow);
                }
            }
        }
    }
}

impl Runtime {
    /// (Re)starts watching the collections' folders, when `auto_index` is on.
    fn refresh_watch(&mut self) {
        if !self.rag.auto_index {
            return;
        }
        let Ok(database) = &self.rag.database else {
            return;
        };
        // A quick read (a few rows); the watch itself runs on notify's threads.
        let roots: Vec<(String, PathBuf)> = crate::storage::Store::open(database)
            .ok()
            .map(crate::storage::Store::into_connection)
            .and_then(|conn| crate::rag::store::collection_sources(&conn).ok())
            .unwrap_or_default()
            .into_iter()
            .map(|source| (source.name, PathBuf::from(source.root)))
            .collect();
        let sender = self.events.sender();
        self.watch = crate::rag::watch::watch(roots, move |names| {
            // Fails only while shutting down.
            let _ = sender.send(Event::App(AppEvent::CollectionsChanged(names)));
        })
        .ok();
    }

    /// Spawns the indexing job, or reports right away why it cannot run.
    fn start_index(&mut self, collection: String, root: &str, types: Option<Vec<String>>) {
        let sender = self.events.sender();
        let fail = |error: String| {
            // Fails only while shutting down.
            let _ = sender.send(Event::App(AppEvent::Index(IndexEvent::Failed {
                collection: collection.clone(),
                error,
            })));
        };
        if self.running_index.is_some() {
            return fail("une indexation est déjà en cours".into());
        }
        let embedder = match &self.rag.embedder {
            Ok(embedder) => Arc::clone(embedder),
            Err(error) => return fail(error.clone()),
        };
        let database = match &self.rag.database {
            Ok(path) => path.clone(),
            Err(error) => return fail(error.clone()),
        };
        // Not necessarily a folder: a collection name re-indexes that collection.
        let root = files::expand_home(root);
        let request = IndexRequest {
            collection: collection.clone(),
            root,
            chunk_tokens: self.rag.chunk_tokens,
            types,
            exclude: self.rag.exclude.clone(),
            ocr: self.rag.ocr.clone(),
        };
        let token = CancellationToken::new();
        self.running_index = Some(token.clone());
        let sender = self.events.sender();
        tokio::spawn(indexer::run(
            request,
            database,
            embedder,
            token,
            move |event| {
                // Fails only while shutting down.
                let _ = sender.send(Event::App(AppEvent::Index(event)));
            },
        ));
    }
}

impl Runtime {
    /// Spawns the download job, or reports right away why it cannot run.
    fn start_pull(&mut self, repo: String, revision: Option<String>, file: RemoteFile) {
        let Err(error) = self.spawn_pull(repo.clone(), revision, file.clone()) else {
            return;
        };
        // Straight to the app rather than through `AppEvent::Pull`: receiving a terminal
        // event on that channel clears `running_pull`, and a refusal says nothing about the
        // download already running — clearing its token there would orphan it.
        self.dispatch(Action::Pull(PullEvent::Failed {
            repo,
            file: file.path,
            error,
        }));
    }

    /// Starts the job, or says why it cannot start.
    fn spawn_pull(
        &mut self,
        repo: String,
        revision: Option<String>,
        file: RemoteFile,
    ) -> Result<(), String> {
        if self.running_pull.is_some() {
            return Err("un téléchargement est déjà en cours".to_owned());
        }
        let hub = Arc::clone(self.models.hub.as_ref().map_err(Clone::clone)?);
        let database = self.models.database.clone()?;
        let request = PullRequest {
            repo,
            revision: revision.unwrap_or_else(|| DEFAULT_REVISION.to_owned()),
            file,
            dir: self.models.dir.clone(),
        };
        let token = CancellationToken::new();
        self.running_pull = Some(token.clone());
        let sender = self.events.sender();
        tokio::spawn(download::run(request, database, hub, token, move |event| {
            // Fails only while shutting down.
            let _ = sender.send(Event::App(AppEvent::Pull(event)));
        }));
        Ok(())
    }
}

impl Runtime {
    /// Starts the configured MCP servers in the background.
    fn start_mcp(&mut self) {
        for (name, config) in std::mem::take(&mut self.mcp) {
            let registry = self.backends.mcp.clone();
            let sender = self.events.sender();
            tokio::spawn(crate::mcp::start(registry, name, config, move |status| {
                // Fails only while shutting down.
                let _ = sender.send(Event::App(AppEvent::Mcp(status)));
            }));
        }
    }
}

/// The reranker of `[rag]`, if one is configured (a misconfigured one is left out:
/// replies then use the hybrid order).
fn build_reranker(
    config: &Config,
    providers: &[crate::config::Provider],
) -> Option<Arc<dyn crate::rag::rerank::Reranker>> {
    let rag = &config.rag;
    let timeout = Duration::from_secs(config.connect_timeout_secs);
    if !rag.rerank_url.trim().is_empty() {
        return crate::rag::rerank::HttpReranker::at_url(
            &rag.rerank_url,
            rag.rerank_model.trim(),
            timeout,
        )
        .ok()
        .map(|r| Arc::new(r) as Arc<dyn crate::rag::rerank::Reranker>);
    }
    if rag.rerank_model.trim().is_empty() {
        return None;
    }
    let id = if rag.rerank_provider.is_empty() {
        &rag.embedding_provider
    } else {
        &rag.rerank_provider
    };
    let provider = providers.iter().find(|p| &p.id == id)?;
    crate::rag::rerank::HttpReranker::new(provider, &rag.rerank_model, timeout)
        .ok()
        .map(|r| Arc::new(r) as Arc<dyn crate::rag::rerank::Reranker>)
}

/// A value unique to this process run, used to build conversation ids.
fn session_seed() -> u64 {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    (millis << 16) ^ u64::from(std::process::id() & 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review Focus 1: the revision a pasted URL names must be the one asked for, all the
    /// way from what the user types to the HTTP request.
    #[tokio::test]
    async fn a_pasted_revision_reaches_the_hub_request() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let (seen, mut requests) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut buffer = [0u8; 2048];
            let read = socket.read(&mut buffer).await.unwrap_or(0);
            let _ = seen.send(String::from_utf8_lossy(&buffer[..read]).into_owned());
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n[]")
                .await;
        });

        let dir = tempfile::tempdir().expect("tempdir");
        let mut runtime = Runtime::new(
            Config::default(),
            false,
            Location::File(dir.path().join("test.db")),
        )
        .expect("runtime");
        runtime.models.hub = Ok(Arc::new(
            Hub::new(Some(base), None, Duration::from_secs(2)).expect("hub"),
        ));

        for c in "/pull https://huggingface.co/owner/name/tree/v2.0".chars() {
            runtime.dispatch(Action::Edit(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        runtime.dispatch(Action::Submit);

        let request = tokio::time::timeout(Duration::from_secs(5), requests.recv())
            .await
            .expect("the request must be sent")
            .expect("a request");
        assert!(
            request.contains("/api/models/owner/name/tree/v2.0"),
            "{request}"
        );
    }

    /// Refusing a download must not clear the running one's token: `Esc`, `/pull`'s own
    /// guard and the shutdown path all go through it, and without it the job runs on with
    /// nothing able to stop it.
    #[tokio::test]
    async fn refusing_a_second_pull_keeps_the_running_job_cancellable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut runtime = Runtime::new(
            Config::default(),
            false,
            Location::File(dir.path().join("test.db")),
        )
        .expect("runtime");
        let running = CancellationToken::new();
        runtime.running_pull = Some(running.clone());

        runtime.execute(Effect::StartPull {
            repo: "owner/second".to_owned(),
            revision: None,
            file: RemoteFile {
                path: "m.gguf".to_owned(),
                bytes: 1,
                sha256: None,
            },
        });
        // Drain whatever the refusal produced, exactly as the main loop would.
        while let Some(event) = runtime.events.try_next() {
            runtime.handle(event);
        }

        assert!(
            runtime.running_pull.is_some(),
            "the running download keeps its token"
        );
        runtime.execute(Effect::CancelPull);
        assert!(
            running.is_cancelled(),
            "Esc must still stop the download that is actually running"
        );
        assert!(
            matches!(runtime.app.status, crate::state::Status::Error(_)),
            "and the user is told why the second one did not start: {:?}",
            runtime.app.status
        );
    }

    #[test]
    fn the_reranker_comes_from_rerank_url_or_a_provider() {
        let mut config = Config::default();
        let providers = config.resolve_providers(|_| None);
        assert!(
            build_reranker(&config, &providers).is_none(),
            "off by default"
        );

        config.rag.rerank_url = "http://localhost:8081".into();
        assert!(
            build_reranker(&config, &providers).is_some(),
            "dedicated server"
        );

        config.rag.rerank_url = "pas une url".into();
        assert!(
            build_reranker(&config, &providers).is_none(),
            "invalid: left out"
        );

        config.rag.rerank_url.clear();
        config.rag.rerank_model = "bge-reranker-v2-m3".into();
        assert!(
            build_reranker(&config, &providers).is_some(),
            "embedding provider"
        );
        config.rag.rerank_provider = "inconnu".into();
        assert!(
            build_reranker(&config, &providers).is_none(),
            "unknown provider"
        );
    }
}
