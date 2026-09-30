//! Re-ranking: a cross-encoder model reads the question with each candidate passage and
//! scores how well the passage answers it, which is more precise than comparing vectors.
//!
//! Uses the `POST /v1/rerank` endpoint shared by llama.cpp server (`--reranking`), vLLM,
//! Text Embeddings Inference, Jina and Cohere-compatible servers:
//! `{model, query, documents}` → `{results: [{index, relevance_score}]}`.

use std::time::Duration;

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

/// [`Reranker`] over `/v1/rerank`.
#[derive(Clone)]
pub struct HttpReranker {
    http: Client,
    endpoint: Endpoint,
    api_key: Option<String>,
    model: String,
}

impl std::fmt::Debug for HttpReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpReranker")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
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
        if provider.kind == ProviderKind::Anthropic {
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
        })
    }
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, LlmError> {
        #[derive(Deserialize)]
        struct Response {
            results: Vec<Item>,
        }
        #[derive(Deserialize)]
        struct Item {
            index: usize,
            #[serde(alias = "score")]
            relevance_score: f32,
        }

        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut request = self
            .http
            .post(self.endpoint.url("/rerank"))
            .timeout(RERANK_TIMEOUT)
            .json(&json!({
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": documents.len(),
            }));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = self.endpoint.send(request).await?;
        let body: Response = response
            .json()
            .await
            .map_err(|e| LlmError::Protocol(format!("re-classement : {e}")))?;
        scores_in_order(
            body.results
                .into_iter()
                .map(|r| (r.index, r.relevance_score)),
            documents.len(),
        )
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

    #[test]
    fn scores_are_put_back_in_document_order() {
        let scores = scores_in_order([(2, 0.9), (0, 0.1)].into_iter(), 3).expect("valid");
        assert_eq!(scores, vec![0.1, f32::NEG_INFINITY, 0.9]);
        assert!(scores_in_order([(5, 1.0)].into_iter(), 3).is_err());
    }
}
