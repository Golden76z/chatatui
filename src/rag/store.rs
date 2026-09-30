//! Collections, documents and passages in SQLite (tables `rag_*`, schemas v4 and v6).
//!
//! `rag_fts` is an FTS5 index over the passages' text, kept in sync by triggers, for the
//! keyword half of hybrid search.
//!
//! Functions take a plain connection so that both the storage worker (listing) and the
//! indexer (writing, on its own connection) can use them. Vectors are stored as
//! little-endian `f32` blobs.

use std::collections::HashMap;

use rusqlite::{Connection, OptionalExtension, params};

use super::chunk::Passage;
use crate::storage::StoreError;

/// A collection row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Collection {
    pub id: i64,
    pub name: String,
    pub root: String,
    pub embedding_model: String,
}

/// A collection for display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollectionSummary {
    pub name: String,
    pub root: String,
    pub embedding_model: String,
    /// File extensions indexed (empty: every supported type).
    pub types: Vec<String>,
    pub documents: u64,
    pub chunks: u64,
    /// Unix seconds.
    pub updated_at: i64,
}

/// What is known about an indexed file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentState {
    pub id: i64,
    pub size: u64,
    pub mtime: i64,
    pub hash: u64,
    /// Why the file could not be indexed (it then has no passages).
    pub skipped: Option<String>,
}

/// A passage loaded for search.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredChunk {
    /// Row id (also the keyword index's rowid).
    pub id: i64,
    /// Path relative to the collection root.
    pub path: String,
    pub location: String,
    pub text: String,
    pub vector: Vec<f32>,
}

fn to_i64(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn to_u64(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

/// Encodes a vector as a blob.
pub fn encode_vector(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Decodes a blob written by [`encode_vector`].
pub fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Returns the collection `name`, creating it if needed. If it was indexed with another
/// embedding model, its documents are dropped (vectors of different models do not mix)
/// and `true` is returned.
pub fn open_collection(
    conn: &Connection,
    name: &str,
    root: &str,
    model: &str,
    now: i64,
) -> Result<(Collection, bool), StoreError> {
    let existing = conn
        .query_row(
            "SELECT id, embedding_model FROM rag_collections WHERE name = ?1",
            [name],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()?;
    let (id, reset) = match existing {
        Some((id, old_model)) => {
            let reset = old_model != model;
            if reset {
                conn.execute("DELETE FROM rag_documents WHERE collection_id = ?1", [id])?;
            }
            conn.execute(
                "UPDATE rag_collections SET root = ?2, embedding_model = ?3,
                     dims = CASE WHEN ?4 THEN 0 ELSE dims END
                 WHERE id = ?1",
                params![id, root, model, reset],
            )?;
            (id, reset)
        }
        None => {
            conn.execute(
                "INSERT INTO rag_collections (name, root, embedding_model, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)",
                params![name, root, model, now],
            )?;
            (conn.last_insert_rowid(), false)
        }
    };
    Ok((
        Collection {
            id,
            name: name.to_owned(),
            root: root.to_owned(),
            embedding_model: model.to_owned(),
        },
        reset,
    ))
}

/// Indexed files of a collection, by relative path.
pub fn documents(
    conn: &Connection,
    collection_id: i64,
) -> Result<HashMap<String, DocumentState>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT id, path, size, mtime, hash, skipped FROM rag_documents
         WHERE collection_id = ?1",
    )?;
    let rows = statement.query_map([collection_id], |r| {
        Ok((
            r.get::<_, String>(1)?,
            DocumentState {
                id: r.get(0)?,
                size: to_u64(r.get(2)?),
                mtime: r.get(3)?,
                hash: to_u64(r.get(4)?),
                skipped: r.get(5)?,
            },
        ))
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Replaces a file's passages (in one transaction).
#[allow(clippy::too_many_arguments)]
pub fn write_document(
    conn: &mut Connection,
    collection_id: i64,
    path: &str,
    size: u64,
    mtime: i64,
    hash: u64,
    passages: &[Passage],
    vectors: &[Vec<f32>],
    now: i64,
) -> Result<(), StoreError> {
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM rag_documents WHERE collection_id = ?1 AND path = ?2",
        params![collection_id, path],
    )?;
    tx.execute(
        "INSERT INTO rag_documents (collection_id, path, size, mtime, hash, indexed_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![collection_id, path, to_i64(size), mtime, to_i64(hash), now],
    )?;
    let document_id = tx.last_insert_rowid();
    {
        let mut insert = tx.prepare(
            "INSERT INTO rag_chunks (document_id, ordinal, location, text, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for (passage, vector) in passages.iter().zip(vectors) {
            insert.execute(params![
                document_id,
                i64::try_from(passage.ordinal).unwrap_or(i64::MAX),
                passage.location,
                passage.text,
                encode_vector(vector),
            ])?;
        }
    }
    if let Some(dims) = vectors.first().map(Vec::len) {
        tx.execute(
            "UPDATE rag_collections SET dims = ?2 WHERE id = ?1 AND dims = 0",
            params![collection_id, i64::try_from(dims).unwrap_or(0)],
        )?;
    }
    tx.commit()?;
    Ok(())
}

/// Records a file that could not be indexed, so that it is not retried until it changes.
pub fn write_skipped(
    conn: &mut Connection,
    collection_id: i64,
    path: &str,
    size: u64,
    mtime: i64,
    reason: &str,
    now: i64,
) -> Result<(), StoreError> {
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM rag_documents WHERE collection_id = ?1 AND path = ?2",
        params![collection_id, path],
    )?;
    tx.execute(
        "INSERT INTO rag_documents (collection_id, path, size, mtime, hash, indexed_at, skipped)
         VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6)",
        params![collection_id, path, to_i64(size), mtime, now, reason],
    )?;
    tx.commit()?;
    Ok(())
}

/// Records a new modification time for an unchanged file.
pub fn touch_document(conn: &Connection, id: i64, mtime: i64) -> Result<(), StoreError> {
    conn.execute(
        "UPDATE rag_documents SET mtime = ?2 WHERE id = ?1",
        params![id, mtime],
    )?;
    Ok(())
}

/// Removes a file and its passages.
pub fn remove_document(conn: &Connection, id: i64) -> Result<(), StoreError> {
    conn.execute("DELETE FROM rag_documents WHERE id = ?1", [id])?;
    Ok(())
}

/// Marks the end of an indexing run.
pub fn finish_collection(conn: &Connection, id: i64, now: i64) -> Result<(), StoreError> {
    conn.execute(
        "UPDATE rag_collections SET updated_at = ?2 WHERE id = ?1",
        params![id, now],
    )?;
    Ok(())
}

/// All collections with their sizes, by name.
pub fn list_collections(conn: &Connection) -> Result<Vec<CollectionSummary>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT c.name, c.root, c.embedding_model, c.updated_at, c.types,
                (SELECT COUNT(*) FROM rag_documents d
                  WHERE d.collection_id = c.id AND d.skipped IS NULL),
                (SELECT COUNT(*) FROM rag_chunks k JOIN rag_documents d ON k.document_id = d.id
                  WHERE d.collection_id = c.id)
         FROM rag_collections c ORDER BY c.name",
    )?;
    let rows = statement.query_map([], |r| {
        Ok(CollectionSummary {
            name: r.get(0)?,
            root: r.get(1)?,
            embedding_model: r.get(2)?,
            updated_at: r.get(3)?,
            types: decode_types(&r.get::<_, String>(4)?),
            documents: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
            chunks: u64::try_from(r.get::<_, i64>(6)?).unwrap_or(0),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Changes whenever the passages of a collection do: (count, highest id).
pub type ChunksVersion = (u64, i64);

/// Ids of the passages of `collections` matching the FTS5 `query`, best first (BM25).
pub fn keyword_search(
    conn: &Connection,
    collections: &[&str],
    query: &str,
    limit: usize,
) -> Result<Vec<i64>, StoreError> {
    let names = serde_json::to_string(collections).unwrap_or_else(|_| "[]".into());
    let mut statement = conn.prepare(
        "SELECT f.rowid FROM rag_fts f
         JOIN rag_chunks k ON k.id = f.rowid
         JOIN rag_documents d ON d.id = k.document_id
         JOIN rag_collections c ON c.id = d.collection_id
         WHERE rag_fts MATCH ?1 AND c.name IN (SELECT value FROM json_each(?2))
         ORDER BY bm25(rag_fts)
         LIMIT ?3",
    )?;
    let rows = statement.query_map(
        params![query, names, i64::try_from(limit).unwrap_or(i64::MAX)],
        |r| r.get(0),
    )?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Where a collection's files come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollectionSource {
    pub id: i64,
    pub name: String,
    pub root: String,
    /// File extensions indexed (empty: every supported type).
    pub types: Vec<String>,
}

/// Folder and file types of a collection (`None`: no such collection).
pub fn collection_source(
    conn: &Connection,
    name: &str,
) -> Result<Option<CollectionSource>, StoreError> {
    Ok(conn
        .query_row(
            "SELECT id, name, root, types FROM rag_collections WHERE name = ?1",
            [name],
            |r| {
                Ok(CollectionSource {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    root: r.get(2)?,
                    types: decode_types(&r.get::<_, String>(3)?),
                })
            },
        )
        .optional()?)
}

/// Every collection's source, by name.
pub fn collection_sources(conn: &Connection) -> Result<Vec<CollectionSource>, StoreError> {
    let mut statement =
        conn.prepare("SELECT id, name, root, types FROM rag_collections ORDER BY name")?;
    let rows = statement.query_map([], |r| {
        Ok(CollectionSource {
            id: r.get(0)?,
            name: r.get(1)?,
            root: r.get(2)?,
            types: decode_types(&r.get::<_, String>(3)?),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Records the file types a collection is limited to (empty: all).
pub fn set_types(conn: &Connection, id: i64, types: &[String]) -> Result<(), StoreError> {
    conn.execute(
        "UPDATE rag_collections SET types = ?2 WHERE id = ?1",
        params![id, types.join(",")],
    )?;
    Ok(())
}

/// Deletes a collection and its passages; conversations using it stop searching it.
/// Returns `false` if there was no such collection.
pub fn delete_collection(conn: &mut Connection, name: &str) -> Result<bool, StoreError> {
    let tx = conn.transaction()?;
    let deleted = tx.execute("DELETE FROM rag_collections WHERE name = ?1", [name])?;
    // `rag_collection` may list several collections (`cours,tp`).
    let using: Vec<(String, String)> = {
        let mut statement = tx.prepare(
            "SELECT id, rag_collection FROM conversations WHERE rag_collection IS NOT NULL",
        )?;
        let rows = statement.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        rows.collect::<Result<_, _>>()?
    };
    for (id, value) in using {
        let names = super::retrieve::collection_names(&value);
        if names.contains(&name) {
            let rest: Vec<&str> = names.into_iter().filter(|n| *n != name).collect();
            let rest = (!rest.is_empty()).then(|| rest.join(","));
            tx.execute(
                "UPDATE conversations SET rag_collection = ?2 WHERE id = ?1",
                params![id, rest],
            )?;
        }
    }
    tx.commit()?;
    Ok(deleted > 0)
}

fn decode_types(types: &str) -> Vec<String> {
    types
        .split(',')
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Embedding model of a collection and a value that changes whenever its passages do
/// (`None`: no such collection).
pub fn collection_state(
    conn: &Connection,
    name: &str,
) -> Result<Option<(String, ChunksVersion)>, StoreError> {
    Ok(conn
        .query_row(
            "SELECT c.embedding_model,
                    (SELECT COUNT(*) FROM rag_chunks k JOIN rag_documents d
                       ON k.document_id = d.id WHERE d.collection_id = c.id),
                    (SELECT COALESCE(MAX(k.id), 0) FROM rag_chunks k JOIN rag_documents d
                       ON k.document_id = d.id WHERE d.collection_id = c.id)
             FROM rag_collections c WHERE c.name = ?1",
            [name],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    (
                        u64::try_from(r.get::<_, i64>(1)?).unwrap_or(0),
                        r.get::<_, i64>(2)?,
                    ),
                ))
            },
        )
        .optional()?)
}

/// Every passage of a collection, with its vector (for search).
pub fn load_chunks(conn: &Connection, collection: &str) -> Result<Vec<StoredChunk>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT k.id, d.path, k.location, k.text, k.embedding
         FROM rag_chunks k
         JOIN rag_documents d ON k.document_id = d.id
         JOIN rag_collections c ON d.collection_id = c.id
         WHERE c.name = ?1
         ORDER BY d.path, k.ordinal",
    )?;
    let rows = statement.query_map([collection], |r| {
        Ok(StoredChunk {
            id: r.get(0)?,
            path: r.get(1)?,
            location: r.get(2)?,
            text: r.get(3)?,
            vector: decode_vector(&r.get::<_, Vec<u8>>(4)?),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Store;

    fn conn() -> Connection {
        Store::open_in_memory().expect("store").into_connection()
    }

    fn passage(ordinal: usize, text: &str) -> Passage {
        Passage {
            ordinal,
            location: format!("L{ordinal}"),
            text: text.to_owned(),
        }
    }

    #[test]
    fn vectors_round_trip() {
        let v = vec![0.5, -1.25, 3.0];
        assert_eq!(decode_vector(&encode_vector(&v)), v);
    }

    #[test]
    fn documents_are_written_listed_and_replaced() {
        let mut conn = conn();
        let (collection, reset) =
            open_collection(&conn, "cours", "/home/x/cours", "m", 10).expect("collection");
        assert!(!reset);
        write_document(
            &mut conn,
            collection.id,
            "a.md",
            12,
            100,
            u64::MAX,
            &[passage(0, "un"), passage(1, "deux")],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            10,
        )
        .expect("write");
        let docs = documents(&conn, collection.id).expect("docs");
        assert_eq!(
            docs["a.md"].hash,
            u64::MAX,
            "u64 hashes survive i64 storage"
        );

        // Rewriting replaces the passages.
        write_document(
            &mut conn,
            collection.id,
            "a.md",
            3,
            200,
            1,
            &[passage(0, "trois")],
            &[vec![1.0, 0.0]],
            11,
        )
        .expect("rewrite");
        let chunks = load_chunks(&conn, "cours").expect("chunks");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "trois");
        assert_eq!(chunks[0].vector, vec![1.0, 0.0]);

        let summary = &list_collections(&conn).expect("list")[0];
        assert_eq!((summary.documents, summary.chunks), (1, 1));
        assert_eq!(summary.root, "/home/x/cours");

        let id = documents(&conn, collection.id).expect("docs")["a.md"].id;
        remove_document(&conn, id).expect("remove");
        assert!(
            load_chunks(&conn, "cours").expect("chunks").is_empty(),
            "cascade"
        );
    }

    #[test]
    fn changing_the_model_resets_the_collection() {
        let mut conn = conn();
        let (collection, _) = open_collection(&conn, "c", "/r", "model-a", 1).expect("open");
        write_document(
            &mut conn,
            collection.id,
            "a.md",
            1,
            1,
            1,
            &[passage(0, "x")],
            &[vec![1.0]],
            1,
        )
        .expect("write");
        let (again, reset) = open_collection(&conn, "c", "/r", "model-a", 2).expect("reopen");
        assert!(!reset);
        assert_eq!(again.id, collection.id);
        let (_, reset) = open_collection(&conn, "c", "/r", "model-b", 3).expect("new model");
        assert!(reset);
        assert!(documents(&conn, collection.id).expect("docs").is_empty());
    }

    #[test]
    fn keyword_index_follows_writes_and_deletes() {
        let mut conn = conn();
        let (collection, _) = open_collection(&conn, "cours", "/r", "m", 1).expect("open");
        let (other, _) = open_collection(&conn, "autre", "/o", "m", 1).expect("open");
        let write = |conn: &mut Connection, id: i64, path: &str, text: &str| {
            write_document(
                conn,
                id,
                path,
                1,
                1,
                1,
                &[passage(0, text)],
                &[vec![1.0]],
                1,
            )
            .expect("write");
        };
        write(
            &mut conn,
            collection.id,
            "a.md",
            "Le module XK-42 gère les tickets",
        );
        write(&mut conn, collection.id, "b.md", "Les élèves rendent le TP");
        write(&mut conn, other.id, "c.md", "XK-42 aussi ici");
        let hits =
            |conn: &Connection, q: &str| keyword_search(conn, &["cours"], q, 10).expect("search");
        assert_eq!(
            hits(&conn, "\"xk\"* OR \"42\"*").len(),
            1,
            "only this collection"
        );
        assert_eq!(hits(&conn, "\"eleve\"*").len(), 1, "accents are ignored");

        // Rewriting a file replaces its passages in the keyword index too.
        write(&mut conn, collection.id, "a.md", "Plus rien à voir");
        assert!(hits(&conn, "\"xk\"*").is_empty());
        assert_eq!(hits(&conn, "\"voir\"*").len(), 1);

        // Deleting the collection removes everything and frees its conversations.
        conn.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at, rag_collection)
             VALUES ('c1', 't', 'm', 0, 0, 'cours')",
            [],
        )
        .expect("conversation");
        assert!(delete_collection(&mut conn, "cours").expect("delete"));
        assert!(!delete_collection(&mut conn, "cours").expect("delete"));
        assert!(hits(&conn, "\"voir\"*").is_empty());
        let rag: Option<String> = conn
            .query_row("SELECT rag_collection FROM conversations", [], |r| r.get(0))
            .expect("row");
        assert_eq!(rag, None);
        let names: Vec<String> = list_collections(&conn)
            .expect("list")
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, vec!["autre"]);
    }

    #[test]
    fn skipped_files_are_remembered_but_not_counted() {
        let mut conn = conn();
        let (collection, _) = open_collection(&conn, "c", "/r", "m", 1).expect("open");
        write_skipped(
            &mut conn,
            collection.id,
            "scan.pdf",
            10,
            5,
            "PDF sans texte",
            1,
        )
        .expect("skip");
        let docs = documents(&conn, collection.id).expect("docs");
        assert_eq!(docs["scan.pdf"].skipped.as_deref(), Some("PDF sans texte"));
        assert_eq!(list_collections(&conn).expect("list")[0].documents, 0);

        set_types(&conn, collection.id, &["pdf".into(), "md".into()]).expect("types");
        let source = collection_source(&conn, "c")
            .expect("source")
            .expect("exists");
        assert_eq!(source.types, vec!["pdf", "md"]);
        assert_eq!(
            list_collections(&conn).expect("list")[0].types,
            vec!["pdf", "md"]
        );
    }

    #[test]
    fn deleting_a_collection_keeps_the_others_in_a_conversation() {
        let mut conn = conn();
        open_collection(&conn, "cours", "/r", "m", 1).expect("open");
        conn.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at, rag_collection)
             VALUES ('c1', 't', 'm', 0, 0, 'cours,tp'), ('c2', 't', 'm', 0, 0, 'tp')",
            [],
        )
        .expect("conversations");
        assert!(delete_collection(&mut conn, "cours").expect("delete"));
        let rag = |id: &str| -> Option<String> {
            conn.query_row(
                "SELECT rag_collection FROM conversations WHERE id = ?1",
                [id],
                |r| r.get(0),
            )
            .expect("row")
        };
        assert_eq!(rag("c1").as_deref(), Some("tp"));
        assert_eq!(rag("c2").as_deref(), Some("tp"));
    }
}
