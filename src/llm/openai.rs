//! Client for OpenAI-compatible servers: OpenAI itself, Ollama, llama.cpp server,
//! LM Studio, vLLM, …

use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use serde_json::json;

use super::{
    ChatMessage, ChatRequest, ChatRole, LlmClient, LlmError, ModelInfo, StreamItem, TokenStream,
    ToolSpec, Usage,
    http::{self, Decoded, Endpoint},
};
use crate::config::Provider;

/// `LlmClient` speaking `/chat/completions` with SSE streaming.
#[derive(Clone, Debug)]
pub struct OpenAiCompatibleClient {
    http: Client,
    endpoint: Endpoint,
    api_key: Option<String>,
}

impl OpenAiCompatibleClient {
    /// Builds a client for a provider.
    pub fn new(provider: &Provider, connect_timeout: Duration) -> Result<Self, LlmError> {
        let (http, endpoint) = http::client_for(provider, connect_timeout)?;
        Ok(Self {
            http,
            endpoint,
            api_key: provider.api_key.clone(),
        })
    }

    fn authorize(&self, request: RequestBuilder) -> RequestBuilder {
        match &self.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    /// OpenAI's own API also lists embedding, audio and image models.
    fn is_official_openai(&self) -> bool {
        self.endpoint.base_url.contains("api.openai.com")
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatibleClient {
    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError> {
        let mut body = json!({
            "model": request.model,
            "messages": wire_messages(&request.messages),
            "stream": true,
            // Ask for token counts in the last chunk (ignored by servers that do not know it).
            "stream_options": { "include_usage": true },
        });
        if !request.tools.is_empty() {
            body["tools"] = wire_tools(&request.tools);
        }
        let post = self.http.post(self.endpoint.url("/chat/completions"));
        let response = self.endpoint.send(self.authorize(post).json(&body)).await?;

        let endpoint = self.endpoint.clone();
        let bytes = response
            .bytes_stream()
            .map(move |chunk| chunk.map_err(|e| endpoint.map_error(&e)));
        Ok(token_stream(bytes).boxed())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        #[derive(Deserialize)]
        struct Models {
            data: Vec<Model>,
        }
        #[derive(Deserialize)]
        struct Model {
            id: String,
        }

        let get = self.http.get(self.endpoint.url("/models"));
        let response = self.endpoint.send(self.authorize(get)).await?;
        let models: Models = response
            .json()
            .await
            .map_err(|e| LlmError::Protocol(e.to_string()))?;
        let official = self.is_official_openai();
        let mut ids: Vec<String> = models
            .data
            .into_iter()
            .map(|m| m.id)
            .filter(|id| !official || is_openai_chat_model(id))
            .collect();
        ids.sort();
        Ok(ids.into_iter().map(ModelInfo::named).collect())
    }

    async fn context_window(&self, model: &str) -> Option<u64> {
        // OpenAI's API does not expose context windows.
        if self.is_official_openai() {
            return None;
        }
        probe_local_window(self, model).await
    }
}

/// Time limit of each context-window probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

impl OpenAiCompatibleClient {
    /// Server root: the base URL without its trailing `/v1`.
    fn origin(&self) -> &str {
        self.endpoint
            .base_url
            .strip_suffix("/v1")
            .unwrap_or(&self.endpoint.base_url)
    }

    async fn probe_json(&self, url: String) -> Option<serde_json::Value> {
        let get = self.authorize(self.http.get(url).timeout(PROBE_TIMEOUT));
        let response = self.endpoint.send(get).await.ok()?;
        response.json().await.ok()
    }
}

/// Effective context window from the native APIs of the usual local servers.
///
/// - Ollama `GET /api/ps`: `context_length` of the loaded model (only once loaded).
/// - LM Studio `GET /api/v1/models`: the loaded instance's `context_length`, else
///   `max_context_length`.
/// - llama.cpp `GET /props`: `n_ctx` of the running server.
async fn probe_local_window(client: &OpenAiCompatibleClient, model: &str) -> Option<u64> {
    let origin = client.origin().to_owned();
    if let Some(ps) = client.probe_json(format!("{origin}/api/ps")).await
        && let Some(n) = ollama_window(&ps, model)
    {
        return Some(n);
    }
    if let Some(models) = client.probe_json(format!("{origin}/api/v1/models")).await
        && let Some(n) = lmstudio_window(&models, model)
    {
        return Some(n);
    }
    let props = client.probe_json(format!("{origin}/props")).await?;
    llamacpp_window(&props)
}

fn same_model(name: &str, model: &str) -> bool {
    name == model || name.strip_suffix(":latest") == Some(model)
}

fn ollama_window(ps: &serde_json::Value, model: &str) -> Option<u64> {
    ps.get("models")?
        .as_array()?
        .iter()
        .find(|m| {
            ["name", "model"].iter().any(|key| {
                m.get(*key)
                    .and_then(|v| v.as_str())
                    .is_some_and(|name| same_model(name, model))
            })
        })?
        .get("context_length")?
        .as_u64()
}

fn lmstudio_window(models: &serde_json::Value, model: &str) -> Option<u64> {
    let entry = models.get("models")?.as_array()?.iter().find(|m| {
        m.get("key").and_then(|v| v.as_str()) == Some(model)
            || m.get("id").and_then(|v| v.as_str()) == Some(model)
    })?;
    entry
        .get("loaded_instances")
        .and_then(|i| i.as_array())
        .and_then(|i| i.first())
        .and_then(|i| i.get("config")?.get("context_length")?.as_u64())
        .or_else(|| entry.get("max_context_length")?.as_u64())
}

fn llamacpp_window(props: &serde_json::Value) -> Option<u64> {
    props
        .get("default_generation_settings")
        .and_then(|s| s.get("n_ctx"))
        .or_else(|| props.get("n_ctx"))?
        .as_u64()
}

/// Filters out OpenAI models that cannot chat (embeddings, audio, images, moderation, …).
fn is_openai_chat_model(id: &str) -> bool {
    const NOT_CHAT: &[&str] = &[
        "embedding",
        "tts",
        "whisper",
        "transcribe",
        "dall-e",
        "image",
        "moderation",
        "realtime",
        "audio",
        "davinci",
        "babbage",
        "search",
        "computer-use",
        "sora",
    ];
    !NOT_CHAT.iter().any(|word| id.contains(word))
}

/// Messages in the chat completions format; tool calls and results use its
/// `tool_calls` / `tool_call_id` fields.
fn wire_messages(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| match m.role {
            ChatRole::Tool => json!({
                "role": "tool",
                "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
                "content": m.content,
            }),
            ChatRole::Assistant if !m.tool_calls.is_empty() => json!({
                "role": "assistant",
                "content": if m.content.is_empty() { serde_json::Value::Null } else { json!(m.content) },
                "tool_calls": m.tool_calls.iter().map(|c| json!({
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })).collect::<Vec<_>>(),
            }),
            ChatRole::User if !m.images.is_empty() => {
                let mut parts = vec![json!({ "type": "text", "text": m.content })];
                parts.extend(m.images.iter().map(|image| {
                    json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{};base64,{}", image.media_type, image.base64)
                        },
                    })
                }));
                json!({ "role": "user", "content": parts })
            }
            _ => json!({ "role": m.role, "content": m.content }),
        })
        .collect()
}

/// Tools in the `tools` format of chat completions.
fn wire_tools(tools: &[ToolSpec]) -> serde_json::Value {
    json!(
        tools
            .iter()
            .map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": serde_json::from_str::<serde_json::Value>(&t.parameters)
                        .unwrap_or_else(|_| json!({ "type": "object" })),
                },
            }))
            .collect::<Vec<_>>()
    )
}

/// Decodes the `data` of one SSE event.
fn decode(data: &str) -> Result<Decoded, LlmError> {
    #[derive(Deserialize)]
    struct Completion {
        #[serde(default)]
        choices: Vec<Choice>,
        usage: Option<WireUsage>,
        error: Option<serde_json::Value>,
    }
    #[derive(Deserialize)]
    struct Choice {
        #[serde(default)]
        delta: Delta,
    }
    #[derive(Default, Deserialize)]
    struct Delta {
        content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<WireToolCall>,
    }
    #[derive(Deserialize)]
    struct WireToolCall {
        #[serde(default)]
        index: usize,
        id: Option<String>,
        function: Option<WireFunction>,
    }
    #[derive(Deserialize)]
    struct WireFunction {
        name: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
    }
    #[derive(Deserialize)]
    struct WireUsage {
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
    }

    if data.trim() == "[DONE]" {
        return Ok(Decoded {
            items: Vec::new(),
            done: true,
        });
    }
    let completion: Completion = serde_json::from_str(data)
        .map_err(|e| LlmError::Protocol(format!("{e} dans « {} »", http::truncate(data, 80))))?;
    if let Some(error) = completion.error {
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| error.to_string());
        return Err(LlmError::Server(message));
    }
    let mut text = String::new();
    let mut calls = Vec::new();
    for choice in completion.choices {
        text.extend(choice.delta.content);
        for call in choice.delta.tool_calls {
            let (name, arguments) = match call.function {
                Some(f) => (f.name, f.arguments.unwrap_or_default()),
                None => (None, String::new()),
            };
            calls.push(StreamItem::ToolCallDelta {
                index: call.index,
                id: call.id,
                name,
                arguments,
            });
        }
    }
    let mut items = Vec::new();
    if !text.is_empty() {
        items.push(StreamItem::Text(text));
    }
    items.extend(calls);
    if let Some(usage) = completion.usage {
        items.push(StreamItem::Usage(Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }));
    }
    Ok(Decoded { items, done: false })
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

    const OLLAMA: &str = include_str!("../../tests/fixtures/ollama.sse");
    const LLAMA_CPP: &str = include_str!("../../tests/fixtures/llamacpp.sse");

    async fn collect(chunks: Vec<Result<Vec<u8>, LlmError>>) -> Vec<Result<StreamItem, LlmError>> {
        token_stream(stream::iter(chunks)).collect().await
    }

    fn split(text: &str, size: usize) -> Vec<Result<Vec<u8>, LlmError>> {
        text.as_bytes()
            .chunks(size)
            .map(|c| Ok(c.to_vec()))
            .collect()
    }

    fn texts(items: &[Result<StreamItem, LlmError>]) -> Vec<String> {
        items
            .iter()
            .filter_map(|i| match i {
                Ok(StreamItem::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn ollama_tokens_in_order() {
        for size in [1, 7, 64, 10_000] {
            let items = collect(split(OLLAMA, size)).await;
            assert!(items.iter().all(Result::is_ok));
            assert_eq!(
                texts(&items).concat(),
                "Bonjour, ça va très bien 🦀",
                "chunk size {size}"
            );
        }
    }

    #[tokio::test]
    async fn llama_cpp_null_content_is_skipped_and_usage_reported() {
        let items = collect(split(LLAMA_CPP, 5)).await;
        assert_eq!(texts(&items), vec!["Salut", " à toi !"]);
        assert_eq!(
            items.last(),
            Some(&Ok(StreamItem::Usage(Usage {
                input_tokens: Some(12),
                output_tokens: Some(4)
            })))
        );
    }

    #[tokio::test]
    async fn usage_only_chunk_after_the_last_token() {
        let input = "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":null}\n\n\
                     data: {\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":2}}\n\n\
                     data: [DONE]\n\n";
        let items = collect(split(input, 9)).await;
        assert_eq!(
            items,
            vec![
                Ok(StreamItem::Text("a".into())),
                Ok(StreamItem::Usage(Usage {
                    input_tokens: Some(30),
                    output_tokens: Some(2)
                }))
            ]
        );
    }

    #[tokio::test]
    async fn nothing_after_done_is_read() {
        let input = "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: [DONE]\n\ndata: garbage\n\n";
        let items = collect(split(input, 3)).await;
        assert_eq!(items, vec![Ok(StreamItem::Text("a".to_owned()))]);
    }

    #[tokio::test]
    async fn error_object_in_stream() {
        let input = "data: {\"error\":{\"message\":\"model is loading\"}}\n\n";
        let items = collect(split(input, 100)).await;
        assert_eq!(
            items,
            vec![Err(LlmError::Server("model is loading".to_owned()))]
        );
    }

    #[tokio::test]
    async fn invalid_json_is_a_protocol_error() {
        let items = collect(split("data: {not json\n\n", 100)).await;
        assert!(matches!(items.as_slice(), [Err(LlmError::Protocol(_))]));
    }

    #[tokio::test]
    async fn transport_error_ends_the_stream() {
        let chunks = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n".to_vec()),
            Err(LlmError::Protocol("connection reset".into())),
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n".to_vec()),
        ];
        let items = collect(chunks).await;
        assert_eq!(items.len(), 2);
        assert!(items[1].is_err());
    }

    #[test]
    fn native_window_formats() {
        let ps = serde_json::json!({"models": [
            {"name": "qwen2.5:7b", "model": "qwen2.5:7b", "context_length": 32768},
            {"name": "llama3.2:latest", "model": "llama3.2:latest", "context_length": 4096}
        ]});
        assert_eq!(ollama_window(&ps, "llama3.2"), Some(4096));
        assert_eq!(ollama_window(&ps, "qwen2.5:7b"), Some(32768));
        assert_eq!(ollama_window(&ps, "mistral"), None, "not loaded");

        let lm = serde_json::json!({"models": [
            {"key": "google/gemma", "max_context_length": 262144,
             "loaded_instances": [{"id": "google/gemma", "config": {"context_length": 4096}}]},
            {"key": "qwen/qwen3", "max_context_length": 40960, "loaded_instances": []}
        ]});
        assert_eq!(
            lmstudio_window(&lm, "google/gemma"),
            Some(4096),
            "loaded size wins"
        );
        assert_eq!(lmstudio_window(&lm, "qwen/qwen3"), Some(40960));

        let props = serde_json::json!({"default_generation_settings": {"n_ctx": 8192}});
        assert_eq!(llamacpp_window(&props), Some(8192));
        assert_eq!(
            llamacpp_window(&serde_json::json!({"n_ctx": 2048})),
            Some(2048)
        );
    }

    #[test]
    fn openai_model_filter() {
        for id in ["gpt-4o", "gpt-4.1-mini", "o3", "chatgpt-4o-latest"] {
            assert!(is_openai_chat_model(id), "{id}");
        }
        for id in [
            "text-embedding-3-small",
            "tts-1",
            "whisper-1",
            "dall-e-3",
            "gpt-image-1",
            "omni-moderation-latest",
            "gpt-4o-realtime-preview",
        ] {
            assert!(!is_openai_chat_model(id), "{id}");
        }
    }

    #[tokio::test]
    async fn tool_calls_arrive_in_pieces() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.md\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let items = collect(split(sse, 9)).await;
        let deltas: Vec<(usize, Option<String>, Option<String>, String)> = items
            .into_iter()
            .filter_map(|i| match i {
                Ok(StreamItem::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                }) => Some((index, id, name, arguments)),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[0].1.as_deref(), Some("call_1"));
        assert_eq!(deltas[0].2.as_deref(), Some("read_file"));
        let arguments: String = deltas.iter().map(|d| d.3.as_str()).collect();
        assert_eq!(arguments, "{\"path\":\"a.md\"}");
    }

    #[test]
    fn tool_messages_use_the_chat_completions_format() {
        use crate::llm::ToolCall;
        let call = ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: "{\"path\":\"a.md\"}".into(),
        };
        let wire = wire_messages(&[
            ChatMessage::tool_request("", vec![call]),
            ChatMessage::tool_result("call_1", "contenu"),
        ]);
        assert_eq!(wire[0]["content"], serde_json::Value::Null);
        assert_eq!(wire[0]["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(wire[1]["role"], "tool");
        assert_eq!(wire[1]["tool_call_id"], "call_1");
    }

    #[test]
    fn images_become_image_url_parts() {
        let mut question = ChatMessage::new(ChatRole::User, "Et ça ?");
        question.images = vec![crate::state::Image {
            media_type: "image/jpeg".into(),
            base64: "AAAA".into(),
        }];
        let wire = wire_messages(&[question]);
        assert_eq!(wire[0]["content"][0]["text"], "Et ça ?");
        assert_eq!(
            wire[0]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,AAAA"
        );
    }
}
