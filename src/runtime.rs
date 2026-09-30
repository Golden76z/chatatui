//! The main loop: draws, waits for events, turns them into actions and executes effects.

use std::{sync::Arc, time::Duration};

use color_eyre::Result;
use crossterm::event::{Event as TermEvent, KeyEventKind, MouseEventKind};
use ratatui::DefaultTerminal;
use tokio_util::sync::CancellationToken;

use crate::{
    action::{Action, Effect},
    app::App,
    config::Config,
    context::NoContext,
    event::{AppEvent, Event, EventHandler},
    files, keymap,
    llm::{self, ProviderModels, RequestId, stream_task},
    storage::worker::{Location, StoreHandle},
    ui,
};

/// How long to wait for a provider's model list.
const MODEL_LIST_TIMEOUT: Duration = Duration::from_secs(15);

/// Lines scrolled per mouse wheel notch.
const MOUSE_SCROLL_LINES: usize = 3;

/// Owns the state, the event source and the backends; the only place where side effects run.
#[derive(Debug)]
pub struct Runtime {
    app: App,
    events: EventHandler,
    backends: stream_task::Backends,
    /// Provider ids in display order (for the model list).
    provider_order: Vec<String>,
    /// Cancellation handle of the running streaming task.
    running_task: Option<(RequestId, CancellationToken)>,
    /// Storage worker; `None` once shut down.
    store: Option<StoreHandle>,
}

impl Runtime {
    /// Creates the runtime with the configured backend, starts the storage worker (history in
    /// `database`) and spawns the terminal event task.
    pub fn new(config: Config, keyboard_enhanced: bool, database: Location) -> Result<Self> {
        let providers = config.resolve_providers(|name| std::env::var(name).ok());
        let clients =
            llm::build_clients(&providers, Duration::from_secs(config.connect_timeout_secs));
        let provider_order = providers.iter().map(|p| p.id.clone()).collect();
        let backends = stream_task::Backends {
            clients,
            context: Arc::new(NoContext),
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
            Event::App(AppEvent::Storage(event)) => Some(Action::Storage(event)),
            Event::App(AppEvent::Models(result)) => Some(Action::ModelsListed(result)),
            Event::App(AppEvent::FileRead(result)) => Some(Action::FileRead(result)),
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
                if let Some((_, token)) = self.running_task.take() {
                    token.cancel();
                }
                let token = CancellationToken::new();
                self.running_task = Some((job.request_id, token.clone()));
                tokio::spawn(stream_task::run(
                    self.backends.clone(),
                    job,
                    token,
                    self.events.sender(),
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
            Effect::ReadFile(path) => {
                let sender = self.events.sender();
                tokio::task::spawn_blocking(move || {
                    let result = files::read_attachment(&path);
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::FileRead(result)));
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
            Effect::CancelCompletion(request_id) => {
                if let Some((id, token)) = self.running_task.take() {
                    if id == request_id {
                        token.cancel();
                    } else {
                        self.running_task = Some((id, token));
                    }
                }
            }
        }
    }
}

/// A value unique to this process run, used to build conversation ids.
fn session_seed() -> u64 {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    (millis << 16) ^ u64::from(std::process::id() & 0xffff)
}
