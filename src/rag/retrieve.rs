//! Retrieval: the [`ContextProvider`] that searches the conversation's collection.
//!
//! The question (plus the previous one when it is short, for follow-ups such as « et la
//! deuxième ? ») is searched two ways:
//! - by meaning: embedded with the collection's model and compared to every passage
//!   (cosine similarity; vectors are unit length, so a dot product);
//! - by keywords: its words in the FTS5 index (BM25), which catches names, codes and rare
//!   terms that embeddings blur.
//!
//! Both rankings are merged by reciprocal rank fusion. Passages below `min_score` that
//! match no keyword are dropped, then the best are kept within `top_k` and a token budget.
//! Passages are loaded from the database once and kept until the collection changes.

use std::{
    collections::HashMap,
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
/// Candidates taken from each ranking before fusion.
const CANDIDATES: usize = 50;
/// Reciprocal rank fusion constant (the usual value).
const RRF_K: f32 = 60.0;
/// Words ignored by the keyword search.
const STOP_WORDS: &[&str] = &[
    "les", "des", "une", "est", "que", "qui", "quoi", "dans", "pour", "par", "sur", "avec", "sans",
    "pas", "plus", "mais", "ont", "sont", "elle", "ils", "elles", "nous", "vous", "leur", "leurs",
    "cette", "ces", "son", "ses", "aux", "comment", "quel", "quelle", "quels", "quelles", "dit",
    "fait", "faire", "peux", "peut", "tu", "moi", "the", "and", "for", "with", "what", "which",
    "how", "does", "this", "that", "are", "from", "about",
];

/// How many passages are kept and how much of the prompt they may use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    pub top_k: usize,
    pub context_tokens: u64,
    pub min_score: f32,
    pub keywords: bool,
}

impl From<&RagConfig> for Selection {
    fn from(config: &RagConfig) -> Self {
        Self {
            top_k: config.top_k,
            context_tokens: config.context_tokens,
            min_score: config.min_score,
            keywords: config.keyword_search,
        }
    }
}

/// Passages of one collection, as last loaded.
struct Loaded {
    version: store::ChunksVersion,
    chunks: Arc<Vec<StoredChunk>>,
}

/// Searches the collections chosen with `/rag` (comma-separated); adds nothing when there
/// is none.
pub struct RagContext {
    embedder: Result<Arc<dyn Embedder>, String>,
    database: Result<PathBuf, String>,
    selection: Selection,
    /// By collection name.
    cache: Mutex<HashMap<String, Loaded>>,
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
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Ids of the passages matching the question's keywords, best first.
    async fn keyword_hits(
        &self,
        collections: &[&str],
        question: &str,
    ) -> Result<Vec<i64>, ContextError> {
        let Some(query) = fts_query(question) else {
            return Ok(Vec::new());
        };
        let database = self.database.clone().map_err(ContextError)?;
        let names: Vec<String> = collections.iter().map(|c| (*c).to_owned()).collect();
        tokio::task::spawn_blocking(move || {
            let conn = Store::open(&database)?.into_connection();
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            store::keyword_search(&conn, &names, &query, CANDIDATES)
        })
        .await
        .map_err(|e| ContextError(format!("recherche interrompue ({e})")))?
        .map_err(|e| ContextError(format!("index : {e}")))
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
            cache.get(&name).map(|c| (c.version, Arc::clone(&c.chunks)))
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
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                collection.to_owned(),
                Loaded {
                    version,
                    chunks: Arc::clone(&chunks),
                },
            );
        Ok((model, chunks))
    }
}

#[async_trait]
impl ContextProvider for RagContext {
    async fn provide(&self, query: ContextQuery<'_>) -> Result<Context, ContextError> {
        let Some(collections) = query.collection else {
            return Ok(Context::default());
        };
        let names = collection_names(collections);
        if names.is_empty() {
            return Ok(Context::default());
        }
        let question = search_text(query.history);
        if question.trim().is_empty() {
            return Ok(Context::default());
        }
        let embedder = self
            .embedder
            .as_ref()
            .map_err(|e| ContextError(e.clone()))?;
        let mut groups = Vec::with_capacity(names.len());
        for name in &names {
            let (model, chunks) = self.chunks(name).await?;
            if model != embedder.model() {
                return Err(ContextError(format!(
                    "« {name} » a été indexée avec {model}, le modèle d'embedding \
                     configuré est {} : relancez /index {name}",
                    embedder.model()
                )));
            }
            groups.push((*name, chunks));
        }
        let vector = embedder
            .embed(std::slice::from_ref(&question))
            .await
            .map_err(|e| ContextError(format!("embeddings : {e}")))?
            .pop()
            .ok_or_else(|| ContextError("embeddings : réponse vide".into()))?;
        let keyword_hits = if self.selection.keywords {
            self.keyword_hits(&names, &question).await?
        } else {
            Vec::new()
        };
        // With several collections, each source says which one it comes from.
        let several = groups.len() > 1;
        let groups: Vec<(Option<&str>, &[StoredChunk])> = groups
            .iter()
            .map(|(name, chunks)| (several.then_some(*name), chunks.as_slice()))
            .collect();
        Ok(Context {
            chunks: select_in(&groups, &vector, &keyword_hits, self.selection),
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

/// FTS5 query for the question's words: each quoted, as a prefix, joined by `OR`
/// (a final `s` is dropped so that plurals match). `None` when no word is left.
pub fn fts_query(question: &str) -> Option<String> {
    let mut words: Vec<String> = Vec::new();
    // Short words count only when they look like a code or an acronym (`42`, `XK`).
    let meaningful = |w: &&str| {
        let length = w.chars().count();
        length >= 3
            || (length == 2
                && (w.chars().any(|c| c.is_ascii_digit()) || w.chars().all(char::is_uppercase)))
    };
    for word in question
        .split(|c: char| !c.is_alphanumeric())
        .filter(meaningful)
        .map(str::to_lowercase)
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
    {
        let word = match word.strip_suffix('s') {
            Some(stem) if stem.chars().count() >= 4 => stem.to_owned(),
            _ => word,
        };
        if !words.contains(&word) {
            words.push(word);
        }
    }
    if words.is_empty() {
        return None;
    }
    Some(
        words
            .iter()
            .map(|w| format!("\"{w}\"*"))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

/// The names in a `/rag` value (`cours,tp`), without duplicates.
pub fn collection_names(value: &str) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for name in value.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// The best passages of one collection for the question: `query` is its vector,
/// `keyword_hits` the ids of the passages matching its words (best first).
pub fn select(
    chunks: &[StoredChunk],
    query: &[f32],
    keyword_hits: &[i64],
    selection: Selection,
) -> Vec<ContextChunk> {
    select_in(&[(None, chunks)], query, keyword_hits, selection)
}

/// [`select`] over several collections; each source is prefixed with its collection's
/// name when given (`cours › plan.docx`).
pub fn select_in(
    groups: &[(Option<&str>, &[StoredChunk])],
    query: &[f32],
    keyword_hits: &[i64],
    selection: Selection,
) -> Vec<ContextChunk> {
    let mut by_meaning: Vec<(f32, Option<&str>, &StoredChunk)> = groups
        .iter()
        .flat_map(|(prefix, chunks)| chunks.iter().map(move |c| (*prefix, c)))
        .filter(|(_, c)| c.vector.len() == query.len())
        .map(|(prefix, c)| (dot(&c.vector, query), prefix, c))
        .collect();
    by_meaning.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Reciprocal rank fusion of the two rankings.
    let mut fused: Vec<(f32, Option<&str>, &StoredChunk)> = Vec::new();
    let keyword_rank = |id: i64| keyword_hits.iter().take(CANDIDATES).position(|k| *k == id);
    for (rank, (score, prefix, chunk)) in by_meaning.iter().enumerate() {
        let keyword = keyword_rank(chunk.id);
        let in_meaning = rank < CANDIDATES && *score >= selection.min_score;
        if !in_meaning && keyword.is_none() {
            continue;
        }
        let mut fusion = 0.0;
        if in_meaning {
            fusion += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        if let Some(position) = keyword {
            fusion += 1.0 / (RRF_K + position as f32 + 1.0);
        }
        fused.push((fusion, *prefix, chunk));
    }
    fused.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut picked: Vec<ContextChunk> = Vec::new();
    let mut used = 0;
    for (_, prefix, chunk) in fused {
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
            source: match prefix {
                Some(collection) => format!("{collection} › {}", chunk.path),
                None => chunk.path.clone(),
            },
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
        keywords: true,
    };

    fn chunk(path: &str, text: &str) -> StoredChunk {
        StoredChunk {
            id: i64::from(path.as_bytes()[0]),
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
        let picked = select(&chunks, &query, &[], WIDE);
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
            select(&chunks, &query, &[], strict)
                .iter()
                .all(|c| c.source != "a.md"),
            "unrelated passages are left out"
        );
        let tight = Selection {
            context_tokens: tokens::estimate("ownership emprunt références"),
            ..WIDE
        };
        assert_eq!(select(&chunks, &query, &[], tight).len(), 1, "token budget");
    }

    #[test]
    fn keyword_hits_rescue_passages_the_vectors_miss() {
        let chunks = vec![
            chunk("a.md", "ownership emprunt références"),
            chunk("b.md", "le module XK-42 gère les tickets"),
        ];
        let query = hash_vector("ownership emprunt références");
        let strict = Selection {
            min_score: 0.9,
            ..WIDE
        };
        let sources = |picked: Vec<ContextChunk>| -> Vec<String> {
            picked.into_iter().map(|c| c.source).collect()
        };
        assert_eq!(sources(select(&chunks, &query, &[], strict)), vec!["a.md"]);
        // b.md matches a keyword: kept despite its low similarity.
        let b = i64::from(b'b');
        assert_eq!(
            sources(select(&chunks, &query, &[b], strict)),
            vec!["a.md", "b.md"]
        );
        // Ranked first by both searches wins over first by one.
        let hits = [b, i64::from(b'a')];
        assert_eq!(sources(select(&chunks, &query, &hits, WIDE))[0], "a.md");
    }

    #[test]
    fn fts_query_keeps_meaningful_words() {
        assert_eq!(
            fts_query("Que dit le cours sur les traits et XK-42 ?").as_deref(),
            Some("\"cour\"* OR \"trait\"* OR \"xk\"* OR \"42\"*")
        );
        assert_eq!(fts_query("et le ?"), None);
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
                ..IndexRequest::default()
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

    #[tokio::test]
    async fn several_collections_are_searched_together() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = indexed(dir.path()).await;
        let tp = dir.path().join("tp");
        std::fs::create_dir(&tp).expect("mkdir");
        std::fs::write(
            tp.join("tp1.md"),
            "# TP 1\n\nImplémentez un trait Forme avec un comportement commun.",
        )
        .expect("write");
        indexer::run(
            IndexRequest {
                collection: "tp".into(),
                root: tp,
                chunk_tokens: 200,
                ..IndexRequest::default()
            },
            db.clone(),
            Arc::new(HashEmbedder::default()),
            CancellationToken::new(),
            |_| {},
        )
        .await;
        let rag = RagContext::new(
            Ok(Arc::new(HashEmbedder::default())),
            Ok(db),
            Selection { top_k: 2, ..WIDE },
        );
        let history = question("un trait et un comportement commun");
        let found = rag
            .provide(ContextQuery {
                collection: Some("cours, tp,cours"),
                history: &history,
            })
            .await
            .expect("search");
        let mut sources: Vec<&str> = found.chunks.iter().map(|c| c.source.as_str()).collect();
        sources.sort_unstable();
        assert_eq!(sources, vec!["cours › traits.md", "tp › tp1.md"]);

        let missing = rag
            .provide(ContextQuery {
                collection: Some("cours,autre"),
                history: &history,
            })
            .await;
        assert!(missing.is_err(), "every collection must exist");
        assert_eq!(collection_names(" a, b,a ,"), vec!["a", "b"]);
    }
}
