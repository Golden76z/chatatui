//! Event plumbing, adapted from the ratatui `event-driven-async` template.
//!
//! A background task merges terminal input and a fixed-rate tick into a single channel.
//! Background work (LLM streaming, storage, …) will push its results into the same channel
//! as [`AppEvent`]s, so the UI loop only ever awaits one receiver and never blocks on I/O.

use color_eyre::eyre::OptionExt;

use crate::{
    files::Attachment,
    llm::{LlmEvent, ProviderModels, RequestId},
    storage::StoreEvent,
};
use crossterm::event::Event as CrosstermEvent;
use futures::{FutureExt, StreamExt};
use std::time::Duration;
use tokio::sync::mpsc;

/// The frequency at which tick events are emitted.
const TICK_FPS: f64 = 30.0;

/// Representation of all possible events.
#[derive(Clone, Debug)]
pub enum Event {
    /// Emitted on a regular schedule; used to batch expensive work (e.g. re-rendering markdown).
    Tick,
    /// Terminal events.
    Crossterm(CrosstermEvent),
    /// Results of background work.
    App(AppEvent),
}

/// Application events produced by background tasks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppEvent {
    /// Progress of a streamed reply.
    Llm {
        request_id: RequestId,
        event: LlmEvent,
    },
    /// Result from the storage worker.
    Storage(StoreEvent),
    /// Model lists of every provider.
    Models(Vec<ProviderModels>),
    /// A file to attach was read.
    FileRead(Result<Attachment, String>),
    /// Completions of a partial path.
    PathCompletions {
        partial: String,
        candidates: Vec<String>,
    },
    /// Context window of a model (`None` if the server does not tell).
    ContextWindow {
        provider: String,
        model: String,
        tokens: Option<u64>,
    },
}

/// Terminal event handler.
#[derive(Debug)]
pub struct EventHandler {
    /// Event sender channel, cloned into background tasks.
    sender: mpsc::UnboundedSender<Event>,
    /// Event receiver channel.
    receiver: mpsc::UnboundedReceiver<Event>,
}

impl EventHandler {
    /// Constructs a new [`EventHandler`] and spawns the terminal event task.
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let actor = EventTask::new(sender.clone());
        tokio::spawn(async { actor.run().await });
        Self { sender, receiver }
    }

    /// Receives the next event, waiting until one is available.
    ///
    /// # Errors
    ///
    /// Returns an error if every sender has been dropped, which only happens if the event task
    /// died.
    pub async fn next(&mut self) -> color_eyre::Result<Event> {
        self.receiver
            .recv()
            .await
            .ok_or_eyre("Failed to receive event")
    }

    /// Returns a sender that background tasks use to report back to the UI loop.
    pub fn sender(&self) -> mpsc::UnboundedSender<Event> {
        self.sender.clone()
    }

    /// Returns an already queued event without waiting, if any.
    pub fn try_next(&mut self) -> Option<Event> {
        self.receiver.try_recv().ok()
    }
}

impl Default for EventHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// A task that reads crossterm events and emits tick events on a regular schedule.
struct EventTask {
    /// Event sender channel.
    sender: mpsc::UnboundedSender<Event>,
}

impl EventTask {
    /// Constructs a new instance of [`EventTask`].
    fn new(sender: mpsc::UnboundedSender<Event>) -> Self {
        Self { sender }
    }

    /// Emits tick events at a fixed rate and forwards crossterm events in between.
    async fn run(self) -> color_eyre::Result<()> {
        let tick_rate = Duration::from_secs_f64(1.0 / TICK_FPS);
        let mut reader = crossterm::event::EventStream::new();
        let mut tick = tokio::time::interval(tick_rate);
        loop {
            let tick_delay = tick.tick();
            let crossterm_event = reader.next().fuse();
            tokio::select! {
              _ = self.sender.closed() => {
                break;
              }
              _ = tick_delay => {
                self.send(Event::Tick);
              }
              Some(Ok(evt)) = crossterm_event => {
                self.send(Event::Crossterm(evt));
              }
            };
        }
        Ok(())
    }

    /// Sends an event to the receiver.
    fn send(&self, event: Event) {
        // Shutting down the app drops the receiver, which makes sending fail. That is expected.
        let _ = self.sender.send(event);
    }
}
