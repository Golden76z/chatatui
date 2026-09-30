//! Embeddings from an OpenAI-compatible `POST /v1/embeddings` endpoint (Ollama, LM Studio,
//! llama.cpp server, vLLM, OpenAI). Vectors are L2-normalized, so that cosine similarity
//! is a plain dot product.

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

/// Computes embeddings.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Model name, stored with the collection (vectors of different models do not mix).
    fn model(&self) -> &str;

    /// One normalized vector per text, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError>;
}

/// [`Embedder`] over `/v1/embeddings`.
#[derive(Clone)]
pub struct OpenAiEmbedder {
    http: Client,
    endpoint: Endpoint,
    api_key: Option<String>,
    model: String,
}

impl std::fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbedder")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// Time allowed for one batch (the first call may load the model).
const EMBED_TIMEOUT: Duration = Duration::from_secs(300);

impl OpenAiEmbedder {
    /// Builds an embedder on `provider` with `model`.
    pub fn new(
        provider: &Provider,
        model: &str,
        connect_timeout: Duration,
    ) -> Result<Self, LlmError> {
        if provider.kind == ProviderKind::Anthropic {
            return Err(LlmError::Protocol(format!(
                "{} ne fournit pas d'embeddings : choisissez un autre embedding_provider",
                provider.label
            )));
        }
        if provider.missing_key() {
            return Err(LlmError::MissingKey {
                server: provider.label.clone(),
                env: provider.api_key_env.clone().unwrap_or_default(),
            });
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
impl Embedder for OpenAiEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        #[derive(Deserialize)]
        struct Response {
            data: Vec<Item>,
        }
        #[derive(Deserialize)]
        struct Item {
            index: Option<usize>,
            embedding: Vec<f32>,
        }

        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut request = self
            .http
            .post(self.endpoint.url("/embeddings"))
            .timeout(EMBED_TIMEOUT)
            .json(&json!({ "model": self.model, "input": texts }));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = self.endpoint.send(request).await?;
        let mut body: Response = response
            .json()
            .await
            .map_err(|e| LlmError::Protocol(format!("embeddings : {e}")))?;
        if body.data.len() != texts.len() {
            return Err(LlmError::Protocol(format!(
                "embeddings : {} vecteurs reçus pour {} textes",
                body.data.len(),
                texts.len()
            )));
        }
        body.data.sort_by_key(|item| item.index.unwrap_or(0));
        Ok(body
            .data
            .into_iter()
            .map(|item| normalize(item.embedding))
            .collect())
    }
}

/// Scales a vector to unit length (zero vectors are left as they are).
pub fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|x| *x /= norm);
    }
    vector
}

/// Deterministic fake embedder for tests: vectors from word hashes, so texts sharing words
/// are close.
#[derive(Debug, Default)]
pub struct HashEmbedder {
    /// Number of `embed` calls made.
    pub calls: std::sync::atomic::AtomicUsize,
}

/// Dimensions of [`HashEmbedder`] vectors.
pub const HASH_DIMS: usize = 64;

#[async_trait]
impl Embedder for HashEmbedder {
    fn model(&self) -> &str {
        "hash-test"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(texts.iter().map(|t| hash_vector(t)).collect())
    }
}

/// Bag-of-words vector of `text` (lowercased words hashed into [`HASH_DIMS`] buckets).
pub fn hash_vector(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; HASH_DIMS];
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 2)
    {
        let bucket = super::fingerprint(word.to_lowercase().as_bytes()) % HASH_DIMS as u64;
        vector[usize::try_from(bucket).unwrap_or(0)] += 1.0;
    }
    normalize(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_normalized() {
        let v = normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn hash_vectors_bring_shared_words_together() {
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let q = hash_vector("ownership en Rust");
        let near = hash_vector("le ownership de Rust garantit");
        let far = hash_vector("recette de la tarte tatin");
        assert!(dot(&q, &near) > dot(&q, &far));
    }

    #[test]
    fn claude_cannot_embed() {
        let providers = crate::config::Config::default().resolve_providers(|_| Some("sk".into()));
        let claude = providers.iter().find(|p| p.id == "claude").expect("preset");
        assert!(OpenAiEmbedder::new(claude, "x", Duration::from_secs(1)).is_err());
    }
}
