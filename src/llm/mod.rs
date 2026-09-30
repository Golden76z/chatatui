//! LLM backends.
//!
//! [`LlmClient`] abstracts the server so the app can be tested with [`mock::MockLlmClient`].
//! Implementations: [`openai::OpenAiCompatibleClient`] (OpenAI, Ollama, llama.cpp, LM Studio,
//! vLLM…), [`anthropic::AnthropicClient`] (Claude) and [`Unavailable`] (a provider that
//! cannot be used, e.g. its API key is missing).

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Serialize;

use crate::config::{Provider, ProviderKind};

pub mod anthropic;
pub mod http;
pub mod mock;
pub mod openai;
pub mod sse;
pub mod stream_task;

/// Role of a message sent to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// A message in the wire format of the chat completions API.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    /// Creates a message.
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// A chat completion request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

/// Token counts reported by the server for one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens of the prompt (system prompt, context and history).
    pub input_tokens: Option<u64>,
    /// Tokens of the reply.
    pub output_tokens: Option<u64>,
}

/// An item of a streamed reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamItem {
    /// A fragment of text.
    Text(String),
    /// Token counts (usually at the end of the stream).
    Usage(Usage),
}

/// Stream of reply items. Dropping it aborts the request.
pub type TokenStream = BoxStream<'static, Result<StreamItem, LlmError>>;

/// A model offered by a server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    /// Context window in tokens, when the server says.
    pub context_window: Option<u64>,
}

impl ModelInfo {
    /// A model with no known context window.
    pub fn named(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            context_window: None,
        }
    }
}

/// Result of listing the models of one provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderModels {
    /// Provider id.
    pub provider: String,
    /// Models, or a user-facing error.
    pub result: Result<Vec<ModelInfo>, String>,
}

/// A chat backend.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Starts a streamed completion. Fails early if the server cannot be reached or rejects
    /// the request; later failures are reported as items of the stream.
    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError>;

    /// Lists the chat models the server offers.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Context window of `model` in tokens, if the server tells (best effort, never fails).
    async fn context_window(&self, _model: &str) -> Option<u64> {
        None
    }
}

/// Backend failures. The `Display` output is shown as-is in the UI.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LlmError {
    /// Nothing listens at the configured address.
    #[error("{server} injoignable sur {url}")]
    Unreachable { server: String, url: String },
    /// The server did not answer in time.
    #[error("délai dépassé en contactant {server} ({url})")]
    Timeout { server: String, url: String },
    /// The provider needs an API key that was not found.
    #[error("{server} : clé API absente (définissez la variable {env})")]
    MissingKey { server: String, env: String },
    /// The API key was refused (HTTP 401 / 403).
    #[error("{server} : clé API refusée ({message})")]
    Auth { server: String, message: String },
    /// Rate limit or quota reached (HTTP 429).
    #[error("{server} : limite atteinte ({message})")]
    RateLimited { server: String, message: String },
    /// The server answered with another HTTP error status.
    #[error("erreur HTTP {status} : {message}")]
    Http { status: u16, message: String },
    /// The server reported an error inside the stream.
    #[error("erreur du serveur : {0}")]
    Server(String),
    /// The request or the response could not be understood.
    #[error("réponse invalide : {0}")]
    Protocol(String),
}

impl LlmError {
    /// Classifies an HTTP error status.
    pub fn from_status(server: &str, status: u16, message: String) -> Self {
        let server = server.to_owned();
        match status {
            401 | 403 => Self::Auth { server, message },
            429 => Self::RateLimited { server, message },
            _ => Self::Http { status, message },
        }
    }
}

/// Stand-in for a provider that cannot be used; every call fails with the same error.
#[derive(Clone, Debug)]
pub struct Unavailable(pub LlmError);

#[async_trait]
impl LlmClient for Unavailable {
    async fn chat_stream(&self, _request: ChatRequest) -> Result<TokenStream, LlmError> {
        Err(self.0.clone())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Err(self.0.clone())
    }
}

/// Clients by provider id.
pub type Clients = HashMap<String, Arc<dyn LlmClient>>;

/// Builds a client for each provider. Providers without their API key, or with an invalid
/// configuration, get an [`Unavailable`] client that explains why.
pub fn build_clients(providers: &[Provider], connect_timeout: Duration) -> Clients {
    providers
        .iter()
        .map(|provider| {
            let client: Arc<dyn LlmClient> = if provider.missing_key() {
                Arc::new(Unavailable(LlmError::MissingKey {
                    server: provider.label.clone(),
                    env: provider.api_key_env.clone().unwrap_or_default(),
                }))
            } else {
                let built = match provider.kind {
                    ProviderKind::Openai => {
                        openai::OpenAiCompatibleClient::new(provider, connect_timeout)
                            .map(|c| Arc::new(c) as Arc<dyn LlmClient>)
                    }
                    ProviderKind::Anthropic => {
                        anthropic::AnthropicClient::new(provider, connect_timeout)
                            .map(|c| Arc::new(c) as Arc<dyn LlmClient>)
                    }
                };
                built.unwrap_or_else(|error| Arc::new(Unavailable(error)))
            };
            (provider.id.clone(), client)
        })
        .collect()
}

/// Message sent by the streaming task to the UI loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmEvent {
    /// A fragment of the assistant reply.
    Token(String),
    /// Token counts measured by the server.
    Usage(Usage),
    /// Passages retrieved for this reply (sent before the first token). They are numbered
    /// in the prompt from `first_number` on.
    Retrieved {
        first_number: usize,
        chunks: Vec<crate::context::ContextChunk>,
    },
    /// The reply is complete.
    Done,
    /// Generation failed; user-facing message.
    Error(String),
}

/// Identifies one generation, so that events of a cancelled one can be ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn error_messages_name_the_server() {
        let error = LlmError::Unreachable {
            server: "Ollama".into(),
            url: "http://localhost:11434/v1".into(),
        };
        assert_eq!(
            error.to_string(),
            "Ollama injoignable sur http://localhost:11434/v1"
        );
        let error = LlmError::MissingKey {
            server: "Claude".into(),
            env: "ANTHROPIC_API_KEY".into(),
        };
        assert_eq!(
            error.to_string(),
            "Claude : clé API absente (définissez la variable ANTHROPIC_API_KEY)"
        );
    }

    #[test]
    fn http_statuses_are_classified() {
        assert!(matches!(
            LlmError::from_status("OpenAI", 401, "bad key".into()),
            LlmError::Auth { .. }
        ));
        assert!(matches!(
            LlmError::from_status("OpenAI", 429, "quota".into()),
            LlmError::RateLimited { .. }
        ));
        assert!(matches!(
            LlmError::from_status("OpenAI", 500, "oops".into()),
            LlmError::Http { status: 500, .. }
        ));
    }

    #[tokio::test]
    async fn providers_without_key_are_unavailable() {
        let providers = Config::default().resolve_providers(|_| None);
        let clients = build_clients(&providers, Duration::from_secs(1));
        assert_eq!(clients.len(), 3);
        let error = clients["claude"].list_models().await.expect_err("no key");
        assert!(matches!(error, LlmError::MissingKey { .. }));
    }

    #[test]
    fn roles_serialize_lowercase() {
        let json = serde_json::to_string(&ChatMessage::new(ChatRole::Assistant, "hi"))
            .expect("serializable");
        assert_eq!(json, r#"{"role":"assistant","content":"hi"}"#);
    }
}
