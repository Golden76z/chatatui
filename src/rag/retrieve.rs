//! Retrieval: the [`ContextProvider`] that searches the conversation's collection.
//!
//! The question (plus the previous one when it is short, for follow-ups such as « et la
//! deuxième ? ») is embedded with the collection's model and compared to every passage
//! (cosine similarity; vectors are unit length, so a dot product). The best passages above
//! `min_score` are kept, within `top_k` and a token budget. Passages are loaded from the
//! database once and kept until the collection changes.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex, PoisonError},
};

use async_trait::async_trait;

use super::{
    RagConfig,
    embed::Embedder,
    store::{self, StoredChunk},
};
use crate::{
    context::{Context, ContextChunk, ContextError, ContextProvider, ContextQuery},
    state::{Message, Role},
    storage::Store,
    tokens,
};

/// Below this many words, the previous question is added to the search.
const SHORT_QUESTION_WORDS: usize = 6;

/// How many passages are kept and how much of the prompt they may use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    pub top_k: usize,
    pub context_tokens: u64,
    pub min_score: f32,
}

impl From<&RagConfig> for Selection {
    fn from(config: &RagConfig) -> Self {
        Self {
            top_k: config.top_k,
            context_tokens: config.context_tokens,
            min_score: config.min_score,
        }
    }
}

/// Passages of one collection, as last loaded.
struct Loaded {
    collection: String,
    version: store::ChunksVersion,
    chunks: Arc<Vec<StoredChunk>>,
}

/// Searches the collection chosen with `/rag`; adds nothing when there is none.
pub struct RagContext {
    embedder: Result<Arc<dyn Embedder>, String>,
    database: Result<PathBuf, String>,
    selection: Selection,
    cache: Mutex<Option<Loaded>>,
}

impl std::fmt::Debug for RagContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagContext")
            .field("embedder", &self.embedder.as_ref().map(|e| e.model()))
            .field("database", &self.database)
            .field("selection", &self.selection)
            .finish_non_exhaustive()
    }
}

impl RagContext {
    /// `embedder` and `database` carry the reason when they are unavailable; it is shown
    /// only if a conversation actually uses a collection.
    pub fn new(
        embedder: Result<Arc<dyn Embedder>, String>,
        database: Result<PathBuf, String>,
        selection: Selection,
    ) -> Self {
        Self {
            embedder,
            database,
            selection,
            cache: Mutex::new(None),
        }
    }

    /// Passages of `collection` and its embedding model, from the cache when unchanged.
    async fn chunks(
        &self,
        collection: &str,
    ) -> Result<(String, Arc<Vec<StoredChunk>>), ContextError> {
        let database = self.database.clone().map_err(ContextError)?;
        let name = collection.to_owned();
        let cached = {
            let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            cache
                .as_ref()
                .filter(|c| c.collection == name)
                .map(|c| (c.version, Arc::clone(&c.chunks)))
        };
        let loaded = tokio::task::spawn_blocking(move || {
            let conn = Store::open(&database)?.into_connection();
            let Some((model, version)) = store::collection_state(&conn, &name)? else {
                return Ok(None);
            };
            let chunks = match cached {
                Some((cached_version, chunks)) if cached_version == version => chunks,
                _ => Arc::new(store::load_chunks(&conn, &name)?),
            };
            Ok::<_, crate::storage::StoreError>(Some((model, version, chunks)))
        })
        .await
        .map_err(|e| ContextError(format!("recherche interrompue ({e})")))?
        .map_err(|e| ContextError(format!("index : {e}")))?;
        let Some((model, version, chunks)) = loaded else {
            return Err(ContextError(format!(
                "collection « {collection} » introuvable (voir /collections)"
            )));
        };
        *self.cache.lock().unwrap_or_else(PoisonError::into_inner) = Some(Loaded {
            collection: collection.to_owned(),
            version,
            chunks: Arc::clone(&chunks),
        });
        Ok((model, chunks))
    }
}

#[async_trait]
impl ContextProvider for RagContext {
    async fn provide(&self, query: ContextQuery<'_>) -> Result<Context, ContextError> {
        let Some(collection) = query.collection else {
            return Ok(Context::default());
        };
        let question = search_text(query.history);
        if question.trim().is_empty() {
            return Ok(Context::default());
        }
        let embedder = self
            .embedder
            .as_ref()
            .map_err(|e| ContextError(e.clone()))?;
        let (model, chunks) = self.chunks(collection).await?;
        if model != embedder.model() {
            return Err(ContextError(format!(
                "« {collection} » a été indexée avec {model}, le modèle d'embedding \
                 configuré est {} : relancez /index",
                embedder.model()
            )));
        }
        let vector = embedder
            .embed(&[question])
            .await
            .map_err(|e| ContextError(format!("embeddings : {e}")))?
            .pop()
            .ok_or_else(|| ContextError("embeddings : réponse vide".into()))?;
        Ok(Context {
            chunks: select(&chunks, &vector, self.selection),
        })
    }
}

/// Text to search for: the last user message, preceded by the previous one when short.
pub fn search_text(history: &[Message]) -> String {
    let mut questions = history
        .iter()
        .rev()
        .filter(|m| m.role == Role::User && !m.content.trim().is_empty());
    let Some(last) = questions.next() else {
        return String::new();
    };
    let last = last.content.trim();
    match questions.next() {
        Some(previous) if last.split_whitespace().count() < SHORT_QUESTION_WORDS => {
            format!("{}\n{last}", previous.content.trim())
        }
        _ => last.to_owned(),
    }
}

/// The passages most similar to `query`, best first.
pub fn select(chunks: &[StoredChunk], query: &[f32], selection: Selection) -> Vec<ContextChunk> {
    let mut scored: Vec<(f32, &StoredChunk)> = chunks
        .iter()
        .filter(|c| c.vector.len() == query.len())
        .map(|c| (dot(&c.vector, query), c))
        .filter(|(score, _)| *score >= selection.min_score)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut picked: Vec<ContextChunk> = Vec::new();
    let mut used = 0;
    for (_, chunk) in scored {
        if picked.len() >= selection.top_k {
            break;
        }
        // Identical passages (a file copied twice) are given once.
        if picked.iter().any(|p| p.text == chunk.text) {
            continue;
        }
        let cost = tokens::estimate(&chunk.text);
        if used + cost > selection.context_tokens {
            continue;
        }
        used += cost;
        picked.push(ContextChunk {
            source: chunk.path.clone(),
            location: chunk.location.clone(),
            text: chunk.text.clone(),
        });
    }
    picked
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        rag::{
            embed::{HashEmbedder, hash_vector},
            indexer::{self, IndexRequest},
        },
        state::{Conversation, MessageStatus},
    };
    use tokio_util::sync::CancellationToken;

    const WIDE: Selection = Selection {
        top_k: 3,
        context_tokens: 10_000,
        min_score: 0.0,
    };

    fn chunk(path: &str, text: &str) -> StoredChunk {
        StoredChunk {
            path: path.into(),
            location: "L1-2".into(),
            text: text.into(),
            vector: hash_vector(text),
        }
    }

    #[test]
    fn best_passages_first_within_limits() {
        let chunks = vec![
            chunk("a.md", "les traits définissent un comportement partagé"),
            chunk("b.md", "ownership emprunt références durée de vie"),
            chunk("c.md", "ownership emprunt références"),
            chunk("d.md", "ownership emprunt références"),
        ];
        let query = hash_vector("ownership emprunt références");
        let picked = select(&chunks, &query, WIDE);
        let sources: Vec<&str> = picked.iter().map(|c| c.source.as_str()).collect();
        assert_eq!(
            sources[0], "c.md",
            "exact match first, its duplicate skipped"
        );
        assert!(!sources.contains(&"d.md"));
        assert_eq!(picked.len(), 3);
        assert_eq!(picked[0].location, "L1-2");

        let strict = Selection {
            min_score: 0.5,
            ..WIDE
        };
        assert!(
            select(&chunks, &query, strict)
                .iter()
                .all(|c| c.source != "a.md"),
            "unrelated passages are left out"
        );
        let tight = Selection {
            context_tokens: tokens::estimate("ownership emprunt références"),
            ..WIDE
        };
        assert_eq!(select(&chunks, &query, tight).len(), 1, "token budget");
    }

    #[test]
    fn short_follow_ups_include_the_previous_question() {
        let mut c = Conversation::new();
        c.push(
            Role::User,
            "Que dit le cours sur l'ownership ?",
            MessageStatus::Complete,
        );
        c.push(Role::Assistant, "Il dit…", MessageStatus::Complete);
        c.push(Role::User, "Et les traits ?", MessageStatus::Complete);
        assert_eq!(
            search_text(c.messages()),
            "Que dit le cours sur l'ownership ?\nEt les traits ?"
        );
        c.push(
            Role::User,
            "Explique en détail la différence entre Box et Rc",
            MessageStatus::Complete,
        );
        assert_eq!(
            search_text(c.messages()),
            "Explique en détail la différence entre Box et Rc"
        );
    }

    fn question(text: &str) -> Vec<Message> {
        let mut c = Conversation::new();
        c.push(Role::User, text, MessageStatus::Complete);
        c.messages().to_vec()
    }

    async fn indexed(dir: &std::path::Path) -> PathBuf {
        let docs = dir.join("docs");
        std::fs::create_dir(&docs).expect("mkdir");
        std::fs::write(
            docs.join("ownership.md"),
            "# Ownership\n\nChaque valeur a un propriétaire unique.",
        )
        .expect("write");
        std::fs::write(
            docs.join("traits.md"),
            "# Traits\n\nUn trait décrit un comportement commun.",
        )
        .expect("write");
        let db = dir.join("db.sqlite");
        indexer::run(
            IndexRequest {
                collection: "cours".into(),
                root: docs,
                chunk_tokens: 200,
            },
            db.clone(),
            Arc::new(HashEmbedder::default()),
            CancellationToken::new(),
            |_| {},
        )
        .await;
        db
    }

    #[tokio::test]
    async fn searches_the_indexed_collection() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = indexed(dir.path()).await;
        let embedder = Arc::new(HashEmbedder::default());
        let rag = RagContext::new(Ok(embedder.clone()), Ok(db), Selection { top_k: 1, ..WIDE });
        let history = question("Qu'est-ce qu'un trait et un comportement commun ?");

        let none = rag
            .provide(ContextQuery {
                collection: None,
                history: &history,
            })
            .await
            .expect("no collection");
        assert!(none.is_empty());
        assert_eq!(embedder.calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        let query = ContextQuery {
            collection: Some("cours"),
            history: &history,
        };
        let found = rag.provide(query).await.expect("search");
        assert_eq!(found.chunks.len(), 1);
        assert_eq!(found.chunks[0].source, "traits.md");
        assert_eq!(found.chunks[0].location, "§ Traits");
        // Second search: passages come from the cache.
        assert_eq!(rag.provide(query).await.expect("search"), found);

        let missing = rag
            .provide(ContextQuery {
                collection: Some("autre"),
                history: &history,
            })
            .await;
        assert_eq!(
            missing,
            Err(ContextError(
                "collection « autre » introuvable (voir /collections)".into()
            ))
        );
    }

    #[tokio::test]
    async fn unavailable_embedder_is_reported_only_when_used() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = indexed(dir.path()).await;
        let rag = RagContext::new(Err("clé manquante".into()), Ok(db), WIDE);
        let history = question("trait");
        let none = ContextQuery {
            collection: None,
            history: &history,
        };
        assert!(rag.provide(none).await.expect("unused").is_empty());
        let used = ContextQuery {
            collection: Some("cours"),
            history: &history,
        };
        assert_eq!(
            rag.provide(used).await,
            Err(ContextError("clé manquante".into()))
        );
    }
}
