//! Re-ranking: a cross-encoder model reads the question with each candidate passage and
//! scores how well the passage answers it, which is more precise than comparing vectors.
//!
//! Two request formats of `POST …/rerank`, the first tried first:
//! - Cohere / Jina, served by llama.cpp server (`--reranking`), vLLM and Infinity:
//!   `{model, query, documents}` → `{results: [{index, relevance_score}]}`;
//! - Text Embeddings Inference: `{query, texts}` → `[{index, score}]`.
//!
//! The format that works is remembered. Either can run a cross-encoder locally
//! (`bge-reranker-v2-m3`…), on a provider or on a dedicated server (`rerank_url`).

use std::{
    sync::atomic::{AtomicU8, Ordering},
    time::Duration,
};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::{
    config::{Provider, ProviderKind},
    llm::{
        LlmError,
        http::{self, Endpoint},
    },
};

/// Time allowed for one re-ranking request.
const RERANK_TIMEOUT: Duration = Duration::from_secs(60);

/// Scores passages against a question.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// One score per document, in order (higher is better).
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, LlmError>;
}

/// Request format of a rerank server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Format {
    /// Not known yet: Cohere is tried, then TEI.
    Unknown = 0,
    /// `{model, query, documents}` (llama.cpp, vLLM, Infinity, Jina, Cohere).
    Cohere = 1,
    /// `{query, texts}` (Text Embeddings Inference).
    Tei = 2,
}

impl Format {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Cohere,
            2 => Self::Tei,
            _ => Self::Unknown,
        }
    }
}

/// [`Reranker`] over `…/rerank`.
pub struct HttpReranker {
    http: Client,
    endpoint: Endpoint,
    api_key: Option<String>,
    model: String,
    /// The [`Format`] that worked, once known.
    format: AtomicU8,
}

impl std::fmt::Debug for HttpReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpReranker")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("format", &self.format())
            .finish_non_exhaustive()
    }
}

impl HttpReranker {
    /// A reranker on `provider` with `model`.
    pub fn new(
        provider: &Provider,
        model: &str,
        connect_timeout: Duration,
    ) -> Result<Self, LlmError> {
        // `Local` serves nothing over HTTP: see the same guard in `rag::embed`.
        if matches!(provider.kind, ProviderKind::Anthropic | ProviderKind::Local) {
            return Err(LlmError::Protocol(format!(
                "{} ne fournit pas de re-classement : choisissez un autre rerank_provider",
                provider.label
            )));
        }
        let (http, endpoint) = http::client_for(provider, connect_timeout)?;
        Ok(Self {
            http,
            endpoint,
            api_key: provider.api_key.clone(),
            model: model.to_owned(),
            format: AtomicU8::new(Format::Unknown as u8),
        })
    }

    /// A reranker on a dedicated server at `url` (llama.cpp, TEI, Infinity…), without an
    /// API key. `model` may be empty for servers that serve a single model.
    pub fn at_url(url: &str, model: &str, connect_timeout: Duration) -> Result<Self, LlmError> {
        let (http, endpoint) =
            http::client_for_url("serveur de re-classement", url, connect_timeout)?;
        Ok(Self {
            http,
            endpoint,
            api_key: None,
            model: model.to_owned(),
            format: AtomicU8::new(Format::Unknown as u8),
        })
    }

    /// The format that worked so far.
    pub fn format(&self) -> Format {
        Format::from_u8(self.format.load(Ordering::Relaxed))
    }

    async fn request(
        &self,
        format: Format,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<f32>, LlmError> {
        let body = match format {
            Format::Tei => json!({
                "query": query,
                "texts": documents,
                "truncate": true,
            }),
            Format::Cohere | Format::Unknown => {
                let mut body = json!({
                    "query": query,
                    "documents": documents,
                    "top_n": documents.len(),
                });
                if !self.model.is_empty() {
                    body["model"] = json!(self.model);
                }
                body
            }
        };
        let mut request = self
            .http
            .post(self.endpoint.url("/rerank"))
            .timeout(RERANK_TIMEOUT)
            .json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = self.endpoint.send(request).await?;
        let body: Response = response
            .json()
            .await
            .map_err(|e| LlmError::Protocol(format!("re-classement : {e}")))?;
        let results = match body {
            Response::Cohere { results } | Response::Data { data: results } => results,
            Response::Tei(results) => results,
        };
        scores_in_order(
            results.into_iter().map(|r| (r.index, r.relevance_score)),
            documents.len(),
        )
    }
}

/// Response of either format.
#[derive(Deserialize)]
#[serde(untagged)]
enum Response {
    Cohere {
        results: Vec<Item>,
    },
    /// Some Jina-compatible servers.
    Data {
        data: Vec<Item>,
    },
    Tei(Vec<Item>),
}

#[derive(Deserialize)]
struct Item {
    index: usize,
    #[serde(alias = "score")]
    relevance_score: f32,
}

/// Whether an error to a Cohere-format request suggests the server wants another format
/// (unknown route, or a body it cannot read).
fn wrong_format(error: &LlmError) -> bool {
    // A 404 carries its own variant, so it is matched on its own: it is the plainest form of
    // "no such route" and the reason this function exists.
    matches!(error, LlmError::NotFound { .. })
        || matches!(error, LlmError::Http { status, .. } if matches!(status, 400 | 405 | 415 | 422))
        || matches!(error, LlmError::Protocol(m) if m.starts_with("re-classement"))
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, LlmError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        match self.format() {
            Format::Unknown => match self.request(Format::Cohere, query, documents).await {
                Ok(scores) => {
                    self.format.store(Format::Cohere as u8, Ordering::Relaxed);
                    Ok(scores)
                }
                Err(error) if wrong_format(&error) => {
                    let scores = self.request(Format::Tei, query, documents).await;
                    if scores.is_ok() {
                        self.format.store(Format::Tei as u8, Ordering::Relaxed);
                        return scores;
                    }
                    // Report the first error: TEI was only a guess.
                    Err(error)
                }
                Err(error) => Err(error),
            },
            format => self.request(format, query, documents).await,
        }
    }
}

/// Scores by document index; documents the server left out get the lowest score.
fn scores_in_order(
    results: impl Iterator<Item = (usize, f32)>,
    count: usize,
) -> Result<Vec<f32>, LlmError> {
    let mut scores = vec![f32::NEG_INFINITY; count];
    for (index, score) in results {
        let slot = scores
            .get_mut(index)
            .ok_or_else(|| LlmError::Protocol(format!("re-classement : index {index} inconnu")))?;
        *slot = score;
    }
    Ok(scores)
}

/// Fake reranker for tests: scores by the number of the query's words in the document.
#[derive(Debug, Default)]
pub struct WordReranker;

#[async_trait]
impl Reranker for WordReranker {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, LlmError> {
        let words: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(str::to_lowercase)
            .collect();
        Ok(documents
            .iter()
            .map(|d| {
                let d = d.to_lowercase();
                words.iter().filter(|w| d.contains(w.as_str())).count() as f32
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same fabricated `base_url` as the embedder's: it parses, so `rerank_provider = "local"`
    /// used to resolve and point a reranker at an address the user never wrote.
    #[test]
    fn the_local_engine_does_no_reranking() {
        let providers = crate::config::Config::default().resolve_providers(|_| None);
        let local = providers
            .iter()
            .find(|p| p.kind == ProviderKind::Local)
            .expect("the local provider is a preset");

        let error = HttpReranker::new(local, "bge-reranker", Duration::from_secs(1))
            .expect_err("the local engine only generates replies");

        let text = error.to_string();
        assert!(text.contains("rerank_provider"), "{text}");
        assert!(
            !text.contains("http://localhost/local"),
            "the fabricated URL must not reach the user: {text}"
        );
    }

    #[test]
    fn scores_are_put_back_in_document_order() {
        let scores = scores_in_order([(2, 0.9), (0, 0.1)].into_iter(), 3).expect("valid");
        assert_eq!(scores, vec![0.1, f32::NEG_INFINITY, 0.9]);
        assert!(scores_in_order([(5, 1.0)].into_iter(), 3).is_err());
    }
}
