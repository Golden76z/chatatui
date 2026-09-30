//! HTTP plumbing shared by the clients: client construction, error classification and the
//! SSE → [`StreamItem`] pipeline.

use std::{collections::VecDeque, pin::Pin, time::Duration};

use futures::{Stream, StreamExt, stream};
use reqwest::{Client, RequestBuilder, Response, Url};

use super::{LlmError, StreamItem, sse::SseParser};
use crate::config::{Provider, is_loopback};

/// Server identity used in error messages.
#[derive(Clone, Debug)]
pub struct Endpoint {
    /// Provider label, e.g. `Claude`.
    pub server: String,
    /// Base URL without trailing slash.
    pub base_url: String,
}

impl Endpoint {
    /// URL of `path` (starting with `/`) under the base URL.
    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    /// Classifies a transport error.
    pub fn map_error(&self, error: &reqwest::Error) -> LlmError {
        let (server, url) = (self.server.clone(), self.base_url.clone());
        if error.is_connect() {
            LlmError::Unreachable { server, url }
        } else if error.is_timeout() {
            LlmError::Timeout { server, url }
        } else {
            LlmError::Protocol(error.to_string())
        }
    }

    /// Sends a request; non-2xx statuses become errors carrying the server's message.
    pub async fn send(&self, request: RequestBuilder) -> Result<Response, LlmError> {
        let response = request.send().await.map_err(|e| self.map_error(&e))?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let body = response.text().await.unwrap_or_default();
        let message = error_message(&body).unwrap_or_else(|| {
            status
                .canonical_reason()
                .unwrap_or("erreur inconnue")
                .to_owned()
        });
        Err(LlmError::from_status(
            &self.server,
            status.as_u16(),
            message,
        ))
    }
}

/// Builds the HTTP client and endpoint for a provider.
pub fn client_for(
    provider: &Provider,
    connect_timeout: Duration,
) -> Result<(Client, Endpoint), LlmError> {
    client_for_url(&provider.label, &provider.base_url, connect_timeout)
}

/// Builds the HTTP client and endpoint for the server `label` at `base_url`.
pub fn client_for_url(
    label: &str,
    base_url: &str,
    connect_timeout: Duration,
) -> Result<(Client, Endpoint), LlmError> {
    let base_url = base_url.trim().trim_end_matches('/').to_owned();
    Url::parse(&base_url)
        .map_err(|e| LlmError::Protocol(format!("URL invalide « {base_url} » : {e}")))?;
    let mut builder = Client::builder().connect_timeout(connect_timeout);
    // A system-wide HTTP proxy must not intercept requests to a local server.
    if is_loopback(&base_url) {
        builder = builder.no_proxy();
    }
    let client = builder
        .build()
        .map_err(|e| LlmError::Protocol(format!("client HTTP : {e}")))?;
    Ok((
        client,
        Endpoint {
            server: label.to_owned(),
            base_url,
        },
    ))
}

/// Extracts a human-readable message from an error body
/// (`{"error":{"message":…}}`, `{"error":"…"}` or plain text).
pub fn error_message(body: &str) -> Option<String> {
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return Some(body.chars().take(200).collect());
    };
    let error = value.get("error").unwrap_or(&value);
    error
        .get("message")
        .and_then(|m| m.as_str())
        .or_else(|| error.as_str())
        .map(str::to_owned)
        .or_else(|| Some(body.chars().take(200).collect()))
}

/// Shortens `text` to `max` characters for error messages.
pub fn truncate(text: &str, max: usize) -> String {
    let mut out: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        out.push('…');
    }
    out
}

/// What one SSE payload means.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Decoded {
    pub items: Vec<StreamItem>,
    /// The stream is complete; nothing after this payload is read.
    pub done: bool,
}

/// Turns a stream of body bytes into reply items, decoding each SSE payload with `decode`.
///
/// Ends when `decode` reports `done`, or at end of body; stops after the first error.
pub fn item_stream<S, B>(
    bytes: S,
    decode: fn(&str) -> Result<Decoded, LlmError>,
) -> impl Stream<Item = Result<StreamItem, LlmError>> + Send
where
    S: Stream<Item = Result<B, LlmError>> + Send + 'static,
    B: AsRef<[u8]>,
{
    struct State<S> {
        bytes: Pin<Box<S>>,
        parser: SseParser,
        pending: VecDeque<StreamItem>,
        payloads: VecDeque<String>,
        eof: bool,
        finished: bool,
    }

    let state = State {
        bytes: Box::pin(bytes),
        parser: SseParser::new(),
        pending: VecDeque::new(),
        payloads: VecDeque::new(),
        eof: false,
        finished: false,
    };

    stream::unfold(state, move |mut st| async move {
        loop {
            if let Some(item) = st.pending.pop_front() {
                return Some((Ok(item), st));
            }
            if st.finished {
                return None;
            }
            if let Some(data) = st.payloads.pop_front() {
                match decode(&data) {
                    Ok(decoded) => {
                        st.pending.extend(decoded.items);
                        st.finished = decoded.done;
                    }
                    Err(error) => {
                        st.finished = true;
                        st.pending.clear();
                        return Some((Err(error), st));
                    }
                }
                continue;
            }
            if st.eof {
                return None;
            }
            match st.bytes.next().await {
                Some(Ok(chunk)) => st.payloads.extend(st.parser.push(chunk.as_ref())),
                Some(Err(error)) => {
                    st.finished = true;
                    return Some((Err(error), st));
                }
                None => {
                    st.eof = true;
                    st.payloads.extend(st.parser.finish());
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_message_formats() {
        assert_eq!(
            error_message(
                r#"{"error":{"message":"model \"x\" not found, try pulling it first","type":"api_error"}}"#
            ),
            Some("model \"x\" not found, try pulling it first".to_owned())
        );
        assert_eq!(
            error_message(
                r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#
            ),
            Some("invalid x-api-key".to_owned())
        );
        assert_eq!(
            error_message(r#"{"error":"bad request"}"#),
            Some("bad request".to_owned())
        );
        assert_eq!(error_message("Not Found"), Some("Not Found".to_owned()));
        assert_eq!(error_message(""), None);
    }
}
