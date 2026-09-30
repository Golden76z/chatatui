//! The HTTP clients (OpenAI-compatible and Anthropic) against a minimal local HTTP server.

use std::time::Duration;

use chatatui::{
    config::{Provider, ProviderKind},
    llm::{
        ChatMessage, ChatRequest, ChatRole, LlmClient, LlmError, ModelInfo, StreamItem, Usage,
        anthropic::AnthropicClient, openai::OpenAiCompatibleClient,
    },
};
use futures::StreamExt;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

const OLLAMA: &str = include_str!("fixtures/ollama.sse");
const ANTHROPIC: &str = include_str!("fixtures/anthropic.sse");

/// Serves one request: replies with `head` then each body part (with a small pause between
/// parts, so the client sees several network chunks). Returns the raw request received.
async fn serve_once(head: String, parts: Vec<Vec<u8>>) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let request = read_request(&mut socket).await;
        socket.write_all(head.as_bytes()).await.expect("write head");
        for part in parts {
            socket.write_all(&part).await.expect("write part");
            socket.flush().await.expect("flush");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let _ = tx.send(request);
    });
    (base_url, rx)
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut data = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = socket.read(&mut buf).await.expect("read");
        data.extend_from_slice(&buf[..n]);
        let text = String::from_utf8_lossy(&data).into_owned();
        if let Some(end) = text.find("\r\n\r\n") {
            let length = text
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            if data.len() >= end + 4 + length || n == 0 {
                return text;
            }
        }
        if n == 0 {
            return String::from_utf8_lossy(&data).into_owned();
        }
    }
}

fn sse_head() -> String {
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_owned()
}

fn provider(kind: ProviderKind, base_url: &str, api_key: Option<&str>) -> Provider {
    Provider {
        id: "test".into(),
        label: "Test".into(),
        kind,
        base_url: base_url.to_owned(),
        model: None,
        api_key: api_key.map(str::to_owned),
        api_key_env: None,
        max_output_tokens: 1024,
        context_window: None,
        local: true,
        prices: None,
    }
}

fn openai(base_url: &str, api_key: Option<&str>) -> OpenAiCompatibleClient {
    OpenAiCompatibleClient::new(
        &provider(ProviderKind::Openai, base_url, api_key),
        Duration::from_secs(2),
    )
    .expect("valid provider")
}

fn anthropic(base_url: &str) -> AnthropicClient {
    AnthropicClient::new(
        &provider(ProviderKind::Anthropic, base_url, Some("sk-ant-test")),
        Duration::from_secs(2),
    )
    .expect("valid provider")
}

fn request() -> ChatRequest {
    ChatRequest {
        model: "test-model".into(),
        messages: vec![
            ChatMessage::new(ChatRole::System, "Be brief."),
            ChatMessage::new(ChatRole::User, "Salut"),
        ],
        tools: Vec::new(),
    }
}

fn json_head(status: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
}

fn body_of(raw: &str) -> serde_json::Value {
    let body = &raw[raw.find("\r\n\r\n").expect("body") + 4..];
    serde_json::from_str(body).expect("json body")
}

fn texts(items: &[StreamItem]) -> String {
    items
        .iter()
        .filter_map(|i| match i {
            StreamItem::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn openai_streams_tokens_from_a_chunked_sse_body() {
    // Cut the body into 13-byte pieces: splits land inside lines and inside UTF-8 characters.
    let parts = OLLAMA.as_bytes().chunks(13).map(<[u8]>::to_vec).collect();
    let (base_url, received) = serve_once(sse_head(), parts).await;

    let stream = openai(&base_url, Some("secret"))
        .chat_stream(request())
        .await
        .expect("stream starts");
    let items: Vec<StreamItem> = stream.map(|t| t.expect("no error")).collect().await;
    assert_eq!(texts(&items), "Bonjour, ça va très bien 🦀");

    let raw = received.await.expect("request captured");
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(
        raw.to_ascii_lowercase()
            .contains("authorization: bearer secret")
    );
    let json = body_of(&raw);
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["stream"], true);
    assert_eq!(json["stream_options"]["include_usage"], true);
    assert_eq!(json["messages"][0]["role"], "system");
    assert_eq!(json["messages"][1]["content"], "Salut");
}

#[tokio::test]
async fn openai_http_error_carries_the_server_message() {
    let body = r#"{"error":{"message":"model \"nope\" not found, try pulling it first","type":"api_error"}}"#;
    let (base_url, _received) = serve_once(
        json_head("404 Not Found", body),
        vec![body.as_bytes().to_vec()],
    )
    .await;

    let error = match openai(&base_url, None).chat_stream(request()).await {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        LlmError::Http {
            status: 404,
            message: "model \"nope\" not found, try pulling it first".into()
        }
    );
}

#[tokio::test]
async fn refused_key_is_reported_as_such() {
    let body =
        r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#;
    let (base_url, _received) = serve_once(
        json_head("401 Unauthorized", body),
        vec![body.as_bytes().to_vec()],
    )
    .await;
    let error = match openai(&base_url, Some("bad")).chat_stream(request()).await {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "Test : clé API refusée (Incorrect API key provided)"
    );
}

#[tokio::test]
async fn closed_port_is_reported_as_unreachable() {
    // Grab a free port, then close it so nothing listens there.
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    drop(listener);

    let error = match openai(&base_url, None).chat_stream(request()).await {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        LlmError::Unreachable {
            server: "Test".into(),
            url: base_url
        }
    );
}

#[tokio::test]
async fn openai_lists_models_sorted() {
    let body = r#"{"object":"list","data":[{"id":"qwen2.5:7b","object":"model"},{"id":"llama3.2","object":"model"}]}"#;
    let (base_url, received) =
        serve_once(json_head("200 OK", body), vec![body.as_bytes().to_vec()]).await;

    let models = openai(&base_url, None).list_models().await.expect("models");
    assert_eq!(
        models,
        vec![ModelInfo::named("llama3.2"), ModelInfo::named("qwen2.5:7b")]
    );
    assert!(
        received
            .await
            .expect("request")
            .starts_with("GET /v1/models HTTP/1.1")
    );
}

#[tokio::test]
async fn anthropic_streams_with_its_own_protocol() {
    let parts = ANTHROPIC
        .as_bytes()
        .chunks(11)
        .map(<[u8]>::to_vec)
        .collect();
    let (base_url, received) = serve_once(sse_head(), parts).await;

    let stream = anthropic(&base_url)
        .chat_stream(request())
        .await
        .expect("stream starts");
    let items: Vec<StreamItem> = stream.map(|t| t.expect("no error")).collect().await;
    assert_eq!(texts(&items), "Bonjour à vous 🦀");
    assert!(items.contains(&StreamItem::Usage(Usage {
        input_tokens: None,
        output_tokens: Some(15)
    })));

    let raw = received.await.expect("request captured");
    assert!(raw.starts_with("POST /v1/messages HTTP/1.1"));
    let lower = raw.to_ascii_lowercase();
    assert!(lower.contains("x-api-key: sk-ant-test"));
    assert!(lower.contains("anthropic-version: 2023-06-01"));
    let json = body_of(&raw);
    assert_eq!(
        json["system"], "Be brief.",
        "system prompt is a top-level field"
    );
    assert_eq!(json["max_tokens"], 1024);
    assert_eq!(json["messages"].as_array().map(Vec::len), Some(1));
    assert_eq!(json["messages"][0]["role"], "user");
}

#[tokio::test]
async fn anthropic_lists_models_with_their_context_window() {
    let body = r#"{"data":[{"type":"model","id":"claude-b","display_name":"B","created_at":"2026-01-01T00:00:00Z","max_input_tokens":200000,"max_tokens":64000},{"type":"model","id":"claude-a","display_name":"A","created_at":"2025-01-01T00:00:00Z","max_input_tokens":null}],"has_more":false,"first_id":"claude-b","last_id":"claude-a"}"#;
    let (base_url, received) =
        serve_once(json_head("200 OK", body), vec![body.as_bytes().to_vec()]).await;

    let models = anthropic(&base_url).list_models().await.expect("models");
    assert_eq!(
        models,
        vec![
            ModelInfo {
                id: "claude-b".into(),
                context_window: Some(200_000)
            },
            ModelInfo::named("claude-a"),
        ]
    );
    assert!(
        received
            .await
            .expect("request")
            .starts_with("GET /v1/models?limit=1000 HTTP/1.1")
    );
}

#[tokio::test]
async fn anthropic_error_body_is_readable() {
    let body =
        r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#;
    let (base_url, _received) = serve_once(
        json_head("401 Unauthorized", body),
        vec![body.as_bytes().to_vec()],
    )
    .await;
    let error = match anthropic(&base_url).chat_stream(request()).await {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        LlmError::Auth {
            server: "Test".into(),
            message: "invalid x-api-key".into()
        }
    );
}

#[test]
fn invalid_base_url_is_rejected() {
    let result = OpenAiCompatibleClient::new(
        &provider(ProviderKind::Openai, "not a url", None),
        Duration::from_secs(1),
    );
    assert!(matches!(result, Err(LlmError::Protocol(_))));
}

/// Serves one connection per response, in order; returns the base URL and the requests
/// received (JSON bodies).
async fn serve_responses(
    responses: Vec<(&'static str, &'static str)>,
) -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        for (status, body) in responses {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_request(&mut socket).await;
            let json = request
                .split_once("\r\n\r\n")
                .and_then(|(head, body)| {
                    head.starts_with("POST /rerank ")
                        .then(|| serde_json::from_str(body).ok())
                        .flatten()
                })
                .unwrap_or(serde_json::Value::Null);
            let _ = tx.send(json);
            socket
                .write_all(json_head(status, body).as_bytes())
                .await
                .expect("write head");
            socket.write_all(body.as_bytes()).await.expect("write body");
            socket.flush().await.expect("flush");
        }
    });
    (base_url, rx)
}

fn documents() -> Vec<String> {
    vec![
        "le borrow checker".into(),
        "les traits".into(),
        "Cargo".into(),
    ]
}

#[tokio::test]
async fn local_rerank_server_in_cohere_format() {
    use chatatui::rag::rerank::{Format, HttpReranker, Reranker};
    // llama.cpp `--reranking`: results in any order, `relevance_score`.
    let (base_url, mut received) = serve_responses(vec![(
        "200 OK",
        r#"{"model":"x","results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":-2.5},{"index":2,"relevance_score":0.1}]}"#,
    )])
    .await;
    let reranker =
        HttpReranker::at_url(&format!("{base_url}/"), "", Duration::from_secs(2)).expect("url");
    assert_eq!(reranker.format(), Format::Unknown);
    let scores = reranker
        .rerank("Qu'est-ce qu'un trait ?", &documents())
        .await
        .expect("scores");
    assert_eq!(scores, vec![-2.5, 0.9, 0.1]);
    assert_eq!(reranker.format(), Format::Cohere);
    let body = received.recv().await.expect("request");
    assert_eq!(body["query"], "Qu'est-ce qu'un trait ?");
    assert_eq!(body["documents"][2], "Cargo");
    assert!(body.get("model").is_none(), "no model: the server's own");
}

#[tokio::test]
async fn text_embeddings_inference_format_is_detected_and_remembered() {
    use chatatui::rag::rerank::{Format, HttpReranker, Reranker};
    let (base_url, mut received) = serve_responses(vec![
        (
            "422 Unprocessable Entity",
            r#"{"error":"Json deserialize error: missing field `texts`","error_type":"Validation"}"#,
        ),
        ("200 OK", r#"[{"index":2,"score":0.7},{"index":0,"score":0.2},{"index":1,"score":0.01}]"#),
        ("200 OK", r#"[{"index":0,"score":0.5},{"index":1,"score":0.4},{"index":2,"score":0.3}]"#),
    ])
    .await;
    let reranker =
        HttpReranker::at_url(&base_url, "bge-reranker-base", Duration::from_secs(2)).expect("url");
    let scores = reranker
        .rerank("cargo", &documents())
        .await
        .expect("scores");
    assert_eq!(scores, vec![0.2, 0.01, 0.7]);
    assert_eq!(reranker.format(), Format::Tei);
    let cohere = received.recv().await.expect("first request");
    assert_eq!(cohere["model"], "bge-reranker-base");
    let tei = received.recv().await.expect("second request");
    assert_eq!(tei["texts"][0], "le borrow checker");
    assert!(tei.get("documents").is_none());

    // Known from now on: one request.
    let scores = reranker
        .rerank("cargo", &documents())
        .await
        .expect("scores");
    assert_eq!(scores, vec![0.5, 0.4, 0.3]);
    assert!(
        received
            .recv()
            .await
            .expect("third request")
            .get("texts")
            .is_some()
    );
}

#[tokio::test]
async fn rerank_server_errors_are_not_mistaken_for_another_format() {
    use chatatui::rag::rerank::{Format, HttpReranker, Reranker};
    let (base_url, _received) = serve_responses(vec![(
        "500 Internal Server Error",
        r#"{"error":{"message":"model is not a reranker"}}"#,
    )])
    .await;
    let reranker = HttpReranker::at_url(&base_url, "", Duration::from_secs(2)).expect("url");
    let error = reranker.rerank("x", &documents()).await.expect_err("fails");
    assert_eq!(
        error,
        LlmError::Http {
            status: 500,
            message: "model is not a reranker".into()
        }
    );
    assert_eq!(reranker.format(), Format::Unknown, "tried again next time");
    assert!(HttpReranker::at_url("pas une url", "", Duration::from_secs(2)).is_err());
}
