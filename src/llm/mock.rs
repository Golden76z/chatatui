//! Scripted [`LlmClient`] for tests (and, later, a possible offline demo mode).

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use futures::{StreamExt, stream};

use super::{ChatRequest, LlmClient, LlmError, ModelInfo, StreamItem, TokenStream, Usage};

/// What the mock answers to the next `chat_stream` call.
#[derive(Clone, Debug)]
pub enum MockReply {
    /// Streams these tokens, then ends normally.
    Tokens(Vec<String>),
    /// Streams these tokens and token counts, then ends normally.
    TokensWithUsage(Vec<String>, Usage),
    /// Streams these tokens, then fails.
    TokensThenError(Vec<String>, LlmError),
    /// Streams these tokens, then never ends (for cancellation tests).
    TokensThenHang(Vec<String>),
    /// Fails before streaming (e.g. server unreachable).
    Fail(LlmError),
}

impl MockReply {
    /// Convenience constructor for [`MockReply::Tokens`].
    pub fn tokens(tokens: &[&str]) -> Self {
        Self::Tokens(tokens.iter().map(|t| (*t).to_owned()).collect())
    }
}

/// A scripted backend that records the requests it receives.
#[derive(Debug, Default)]
pub struct MockLlmClient {
    replies: Mutex<VecDeque<MockReply>>,
    requests: Mutex<Vec<ChatRequest>>,
    models: Vec<ModelInfo>,
    context_window: Option<u64>,
    dropped_streams: Arc<AtomicUsize>,
}

impl MockLlmClient {
    /// Creates a mock that answers the given replies in order.
    pub fn new(replies: impl IntoIterator<Item = MockReply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            ..Self::default()
        }
    }

    /// Sets the models returned by `list_models`.
    pub fn with_models(mut self, models: &[&str]) -> Self {
        self.models = models.iter().map(|m| ModelInfo::named(*m)).collect();
        self
    }

    /// Sets the context window reported for every model.
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Requests received so far.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Number of token streams that have been dropped (finished or aborted).
    pub fn dropped_streams(&self) -> usize {
        self.dropped_streams.load(Ordering::SeqCst)
    }
}

/// Increments a counter when dropped; carried by each stream.
struct DropGuard(Arc<AtomicUsize>);

impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request);
        let reply = self
            .replies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
            .unwrap_or(MockReply::Tokens(Vec::new()));

        let text = |tokens: Vec<String>| -> Vec<Result<StreamItem, LlmError>> {
            tokens
                .into_iter()
                .map(|t| Ok(StreamItem::Text(t)))
                .collect()
        };
        let (items, tail): (
            Vec<Result<StreamItem, LlmError>>,
            Option<Result<StreamItem, LlmError>>,
        ) = match reply {
            MockReply::Fail(error) => return Err(error),
            MockReply::Tokens(tokens) => (text(tokens), None),
            MockReply::TokensWithUsage(tokens, usage) => {
                (text(tokens), Some(Ok(StreamItem::Usage(usage))))
            }
            MockReply::TokensThenError(tokens, error) => (text(tokens), Some(Err(error))),
            MockReply::TokensThenHang(tokens) => {
                let guard = DropGuard(self.dropped_streams.clone());
                let s = stream::iter(text(tokens))
                    .chain(stream::pending())
                    .map(move |item| {
                        let _keep = &guard;
                        item
                    });
                return Ok(s.boxed());
            }
        };
        let guard = DropGuard(self.dropped_streams.clone());
        let s = stream::iter(items.into_iter().chain(tail)).map(move |item| {
            let _keep = &guard;
            item
        });
        Ok(s.boxed())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Ok(self.models.clone())
    }

    async fn context_window(&self, _model: &str) -> Option<u64> {
        self.context_window
    }
}
