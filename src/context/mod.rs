//! Additional context injected into the prompt.
//!
//! This is the extension point for retrieval-augmented generation: a provider receives the
//! conversation (and the document collection it uses, if any) and returns the passages to
//! show the model. [`NoContext`] adds nothing; `crate::rag::retrieve::RagContext` searches
//! an indexed collection.

use async_trait::async_trait;

use crate::state::Message;

mod none;

pub use none::NoContext;

/// A piece of retrieved text and where it comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextChunk {
    /// Human-readable origin (file path, URL, …), shown in the prompt and as a citation.
    pub source: String,
    /// Where in the source (`p. 3`, `§ Séance 1`, `L12-40`); empty when not applicable.
    pub location: String,
    /// The text given to the model.
    pub text: String,
}

impl ContextChunk {
    /// `source location`, as written in the prompt.
    pub fn label(&self) -> String {
        if self.location.is_empty() {
            self.source.clone()
        } else {
            format!("{} {}", self.source, self.location)
        }
    }
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

/// What a provider is asked for.
#[derive(Clone, Copy, Debug)]
pub struct ContextQuery<'a> {
    /// Document collection chosen for the conversation (`/rag`), if any.
    pub collection: Option<&'a str>,
    /// Messages in the context, up to and including the new user message.
    pub history: &'a [Message],
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
    /// Returns the context to inject before answering the last message of the query.
    async fn provide(&self, query: ContextQuery<'_>) -> Result<Context, ContextError>;
}
