//! Additional context injected into the prompt.
//!
//! This is the extension point for retrieval-augmented generation: a future RAG provider
//! will search the user's documents for the conversation and return the relevant chunks.
//! For now only [`NoContext`] exists.

use async_trait::async_trait;

use crate::state::Message;

mod none;

pub use none::NoContext;

/// A piece of retrieved text and where it comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextChunk {
    /// Human-readable origin (file path, URL, …), to be shown as a citation later.
    pub source: String,
    /// The text given to the model.
    pub text: String,
}

/// Context returned by a [`ContextProvider`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    pub chunks: Vec<ContextChunk>,
}

impl Context {
    /// `true` when there is nothing to inject.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Failure of a context provider; shown to the user.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("contexte indisponible : {0}")]
pub struct ContextError(pub String);

/// Computes extra context for a conversation.
///
/// Called from the streaming task (never from the UI loop), so implementations may be slow.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Returns the context to inject before answering the last message of `conversation`.
    async fn provide(&self, conversation: &[Message]) -> Result<Context, ContextError>;
}
