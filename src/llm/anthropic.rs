//! Client for the Anthropic Messages API (Claude), used natively rather than through its
//! OpenAI compatibility layer, which Anthropic does not recommend for production.

use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    ChatMessage, ChatRequest, ChatRole, LlmClient, LlmError, ModelInfo, StreamItem, TokenStream,
    Usage,
    http::{self, Decoded, Endpoint},
};
use crate::config::Provider;

/// API version sent in the `anthropic-version` header.
const API_VERSION: &str = "2023-06-01";

/// `LlmClient` for `POST /v1/messages` with SSE streaming.
#[derive(Clone)]
pub struct AnthropicClient {
    http: Client,
    endpoint: Endpoint,
    api_key: String,
    max_output_tokens: u32,
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

impl AnthropicClient {
    /// Builds a client for a provider; the API key is required.
    pub fn new(provider: &Provider, connect_timeout: Duration) -> Result<Self, LlmError> {
        let api_key = provider
            .api_key
            .clone()
            .ok_or_else(|| LlmError::MissingKey {
                server: provider.label.clone(),
                env: provider
                    .api_key_env
                    .clone()
                    .unwrap_or_else(|| "ANTHROPIC_API_KEY".into()),
            })?;
        let (http, endpoint) = http::client_for(provider, connect_timeout)?;
        Ok(Self {
            http,
            endpoint,
            api_key,
            max_output_tokens: provider.max_output_tokens,
        })
    }

    fn authorize(&self, request: RequestBuilder) -> RequestBuilder {
        request
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
    }
}

/// A message in the Messages API format.
#[derive(Debug, PartialEq, Eq, Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: &'a str,
}

/// Splits chat messages into the top-level `system` text and the user/assistant turns.
fn to_wire(messages: &[ChatMessage]) -> (String, Vec<WireMessage<'_>>) {
    let system = messages
        .iter()
        .filter(|m| m.role == ChatRole::System)
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let turns = messages
        .iter()
        .filter_map(|m| {
            let role = match m.role {
                ChatRole::System => return None,
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            Some(WireMessage {
                role,
                content: &m.content,
            })
        })
        .collect();
    (system, turns)
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError> {
        let (system, messages) = to_wire(&request.messages);
        let mut body = json!({
            "model": request.model,
            "max_tokens": self.max_output_tokens,
            "messages": messages,
            "stream": true,
        });
        if !system.is_empty() {
            body["system"] = json!(system);
        }
        let post = self.http.post(self.endpoint.url("/messages"));
        let response = self.endpoint.send(self.authorize(post).json(&body)).await?;

        let endpoint = self.endpoint.clone();
        let bytes = response
            .bytes_stream()
            .map(move |chunk| chunk.map_err(|e| endpoint.map_error(&e)));
        Ok(token_stream(bytes).boxed())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        #[derive(Deserialize)]
        struct Page {
            data: Vec<Model>,
        }
        #[derive(Deserialize)]
        struct Model {
            id: String,
            max_input_tokens: Option<u64>,
        }

        let get = self.http.get(self.endpoint.url("/models?limit=1000"));
        let response = self.endpoint.send(self.authorize(get)).await?;
        let page: Page = response
            .json()
            .await
            .map_err(|e| LlmError::Protocol(e.to_string()))?;
        // The API lists the newest models first; keep that order.
        Ok(page
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                context_window: m.max_input_tokens,
            })
            .collect())
    }

    async fn context_window(&self, model: &str) -> Option<u64> {
        self.fetch_window(model).await
    }
}

impl AnthropicClient {
    /// `GET /v1/models/{id}`: the model's `max_input_tokens`.
    async fn fetch_window(&self, model: &str) -> Option<u64> {
        #[derive(Deserialize)]
        struct Model {
            max_input_tokens: Option<u64>,
        }
        let get = self
            .http
            .get(self.endpoint.url(&format!("/models/{model}")));
        let response = self
            .endpoint
            .send(self.authorize(get).timeout(PROBE_TIMEOUT))
            .await
            .ok()?;
        response.json::<Model>().await.ok()?.max_input_tokens
    }
}

/// Time limit of the metadata request.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Decodes the `data` of one Messages API stream event.
fn decode(data: &str) -> Result<Decoded, LlmError> {
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Event {
        MessageStart {
            message: StartMessage,
        },
        ContentBlockDelta {
            delta: Delta,
        },
        MessageDelta {
            usage: Option<WireUsage>,
        },
        MessageStop,
        Error {
            error: WireError,
        },
        #[serde(other)]
        Other,
    }
    #[derive(Deserialize)]
    struct StartMessage {
        usage: Option<WireUsage>,
    }
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Delta {
        TextDelta {
            text: String,
        },
        #[serde(other)]
        Other,
    }
    #[derive(Deserialize)]
    struct WireUsage {
        input_tokens: Option<u64>,
        cache_creation_input_tokens: Option<u64>,
        cache_read_input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    }
    #[derive(Deserialize)]
    struct WireError {
        message: String,
    }

    let event: Event = serde_json::from_str(data)
        .map_err(|e| LlmError::Protocol(format!("{e} dans « {} »", http::truncate(data, 80))))?;
    let decoded = match event {
        Event::MessageStart { message } => {
            let input = message.usage.map(|u| {
                // Cached prompt tokens still occupy the context window.
                u.input_tokens.unwrap_or(0)
                    + u.cache_creation_input_tokens.unwrap_or(0)
                    + u.cache_read_input_tokens.unwrap_or(0)
            });
            Decoded {
                items: vec![StreamItem::Usage(Usage {
                    input_tokens: input,
                    output_tokens: None,
                })],
                done: false,
            }
        }
        Event::ContentBlockDelta {
            delta: Delta::TextDelta { text },
        } if !text.is_empty() => Decoded {
            items: vec![StreamItem::Text(text)],
            done: false,
        },
        Event::MessageDelta { usage: Some(usage) } => Decoded {
            items: vec![StreamItem::Usage(Usage {
                input_tokens: None,
                output_tokens: usage.output_tokens,
            })],
            done: false,
        },
        Event::MessageStop => Decoded {
            items: Vec::new(),
            done: true,
        },
        Event::Error { error } => return Err(LlmError::Server(error.message)),
        Event::ContentBlockDelta { .. } | Event::MessageDelta { usage: None } | Event::Other => {
            Decoded::default()
        }
    };
    Ok(decoded)
}

/// Turns a stream of body bytes into reply items.
pub fn token_stream<S, B>(bytes: S) -> impl Stream<Item = Result<StreamItem, LlmError>> + Send
where
    S: Stream<Item = Result<B, LlmError>> + Send + 'static,
    B: AsRef<[u8]>,
{
    http::item_stream(bytes, decode)
}

#[cfg(test)]
mod tests {
    use futures::stream;

    use super::*;

    const STREAM: &str = include_str!("../../tests/fixtures/anthropic.sse");

    async fn collect(text: &str, size: usize) -> Vec<Result<StreamItem, LlmError>> {
        let chunks: Vec<Result<Vec<u8>, LlmError>> = text
            .as_bytes()
            .chunks(size)
            .map(|c| Ok(c.to_vec()))
            .collect();
        token_stream(stream::iter(chunks)).collect().await
    }

    #[tokio::test]
    async fn text_and_usage_from_a_real_stream() {
        for size in [1, 13, 10_000] {
            let items = collect(STREAM, size).await;
            let items: Vec<StreamItem> = items.into_iter().map(|i| i.expect("ok")).collect();
            assert_eq!(
                items,
                vec![
                    StreamItem::Usage(Usage {
                        input_tokens: Some(25 + 100),
                        output_tokens: None
                    }),
                    StreamItem::Text("Bonjour".into()),
                    StreamItem::Text(" à vous 🦀".into()),
                    StreamItem::Usage(Usage {
                        input_tokens: None,
                        output_tokens: Some(15)
                    }),
                ],
                "chunk size {size}"
            );
        }
    }

    #[tokio::test]
    async fn thinking_and_unknown_events_are_ignored() {
        let input = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n\
                     event: brand_new\ndata: {\"type\":\"brand_new\",\"x\":1}\n\n\
                     event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert!(collect(input, 7).await.is_empty());
    }

    #[tokio::test]
    async fn error_event_ends_the_stream() {
        let input = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
        assert_eq!(
            collect(input, 100).await,
            vec![Err(LlmError::Server("Overloaded".into()))]
        );
    }

    #[test]
    fn system_messages_are_hoisted() {
        let messages = vec![
            ChatMessage::new(ChatRole::System, "Be brief."),
            ChatMessage::new(ChatRole::User, "Salut"),
            ChatMessage::new(ChatRole::Assistant, "Bonjour"),
            ChatMessage::new(ChatRole::User, "Ça va ?"),
        ];
        let (system, turns) = to_wire(&messages);
        assert_eq!(system, "Be brief.");
        let roles: Vec<&str> = turns.iter().map(|t| t.role).collect();
        assert_eq!(roles, ["user", "assistant", "user"]);
    }
}
