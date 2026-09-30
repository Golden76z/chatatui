//! A dedicated thread owning the [`Store`].
//!
//! Requests are processed in order (a message is never written before its conversation),
//! off the async runtime. Dropping the [`StoreHandle`] lets the thread finish the queued
//! writes; [`StoreHandle::shutdown`] waits for that.

use std::{
    path::PathBuf,
    sync::mpsc,
    thread::{self, JoinHandle},
};

use super::{Store, StoreEvent, StoreRequest};

/// Where the database lives.
#[derive(Clone, Debug)]
pub enum Location {
    File(PathBuf),
    Memory,
}

/// Sender side of the storage thread.
#[derive(Debug)]
pub struct StoreHandle {
    requests: Option<mpsc::Sender<StoreRequest>>,
    thread: Option<JoinHandle<()>>,
}

impl StoreHandle {
    /// Starts the storage thread. Results are passed to `report`.
    ///
    /// If the database cannot be opened, every request is answered with an error event, so
    /// the app keeps working without history.
    pub fn spawn(location: Location, report: impl Fn(StoreEvent) + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel::<StoreRequest>();
        let thread = thread::Builder::new()
            .name("chatatui-storage".into())
            .spawn(move || run(location, &rx, &report))
            .ok();
        Self {
            requests: Some(tx),
            thread,
        }
    }

    /// Queues a request.
    pub fn send(&self, request: StoreRequest) {
        if let Some(requests) = &self.requests {
            // Fails only if the thread is gone, in which case there is nobody to tell.
            let _ = requests.send(request);
        }
    }

    /// Waits until every queued request has been processed.
    pub fn shutdown(mut self) {
        self.requests = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(location: Location, requests: &mpsc::Receiver<StoreRequest>, report: &dyn Fn(StoreEvent)) {
    let opened = match &location {
        Location::File(path) => Store::open(path),
        Location::Memory => Store::open_in_memory(),
    };
    let mut store = match opened {
        Ok(store) => store,
        Err(error) => {
            let message = format!("historique indisponible : {error}");
            report(StoreEvent::Error(message.clone()));
            // Keep answering so that loads and lists do not wait forever.
            for request in requests {
                if !matches!(
                    request,
                    StoreRequest::SaveMessage { .. }
                        | StoreRequest::SetModel { .. }
                        | StoreRequest::SetContextStart { .. }
                        | StoreRequest::SetRag { .. }
                        | StoreRequest::Rename { .. }
                ) {
                    report(StoreEvent::Error(message.clone()));
                }
            }
            return;
        }
    };
    for request in requests {
        if let Some(event) = store.handle(request) {
            report(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, PoisonError};

    use super::*;
    use crate::{
        state::{Message, MessageId, MessageStatus, Role},
        storage::{ConversationId, ConversationRecord},
    };

    fn collector() -> (
        Arc<Mutex<Vec<StoreEvent>>>,
        impl Fn(StoreEvent) + Send + 'static,
    ) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        (events, move |e| {
            sink.lock().unwrap_or_else(PoisonError::into_inner).push(e);
        })
    }

    #[test]
    fn processes_requests_in_order() {
        let (events, report) = collector();
        let handle = StoreHandle::spawn(Location::Memory, report);
        handle.send(StoreRequest::SaveMessage {
            conversation: ConversationRecord {
                id: ConversationId("c".into()),
                title: "t".into(),
                provider: "p".into(),
                model: "m".into(),
                rag_collection: None,
            },
            message: Message {
                id: MessageId(0),
                role: Role::User,
                content: "x".into(),
                status: MessageStatus::Complete,
                source: None,
                citations: Vec::new(),
            },
        });
        handle.send(StoreRequest::Load(ConversationId("c".into())));
        handle.shutdown();

        let events = events.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(matches!(events.as_slice(), [StoreEvent::Loaded(c)] if c.messages.len() == 1));
    }

    #[test]
    fn unopenable_database_reports_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        // A directory cannot be opened as a database file.
        let (events, report) = collector();
        let handle = StoreHandle::spawn(Location::File(dir.path().to_path_buf()), report);
        handle.send(StoreRequest::List);
        handle.shutdown();

        let events = events.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(events.len(), 2, "startup error + answer to List");
        assert!(
            matches!(&events[1], StoreEvent::Error(m) if m.starts_with("historique indisponible"))
        );
    }
}
