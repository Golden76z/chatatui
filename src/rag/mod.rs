//! Retrieval-augmented generation over the user's documents.
//!
//! - [`extract`]: text out of Markdown, text, code, PDF, .docx and .odt files;
//! - [`chunk`]: passages of about `chunk_tokens` tokens, each with its location;
//! - [`embed`]: vectors from an OpenAI-compatible `/v1/embeddings` endpoint;
//! - [`store`]: collections, documents and passages in the SQLite database;
//! - [`indexer`]: the background job behind `/index`;
//! - [`retrieve`]: the context provider that searches a collection for each reply.

pub mod chunk;
pub mod embed;
pub mod extract;
pub mod indexer;
pub mod retrieve;
pub mod store;

use serde::Deserialize;

/// `[rag]` section of the configuration.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagConfig {
    /// Provider computing the embeddings (must speak the OpenAI API: Ollama, LM Studio,
    /// llama.cpp, OpenAI…).
    pub embedding_provider: String,
    pub embedding_model: String,
    /// Target size of a passage, in tokens.
    pub chunk_tokens: usize,
    /// Most passages given to the model per reply.
    pub top_k: usize,
    /// Token budget of the passages given per reply.
    pub context_tokens: u64,
    /// Passages less similar to the question than this (cosine, 0–1) are left out,
    /// unless they contain its keywords.
    pub min_score: f32,
    /// Combine the similarity search with a keyword search (names, codes, rare terms).
    pub keyword_search: bool,
    /// Gitignore-style patterns never indexed (`*.min.js`, `node_modules/`, …).
    pub exclude: Vec<String>,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            embedding_provider: "ollama".into(),
            embedding_model: "bge-m3".into(),
            chunk_tokens: 800,
            top_k: 5,
            context_tokens: 3_000,
            min_score: 0.3,
            keyword_search: true,
            exclude: Vec::new(),
        }
    }
}

/// 64-bit FNV-1a hash, to notice content changes cheaply.
pub fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_differ_with_content() {
        assert_eq!(fingerprint(b"abc"), fingerprint(b"abc"));
        assert_ne!(fingerprint(b"abc"), fingerprint(b"abd"));
    }
}
