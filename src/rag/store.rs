//! Collections, documents and passages in SQLite (tables `rag_*`, schema v4).
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
    pub documents: u64,
    pub chunks: u64,
    /// Unix seconds.
    pub updated_at: i64,
}

/// What is known about an indexed file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocumentState {
    pub id: i64,
    pub size: u64,
    pub mtime: i64,
    pub hash: u64,
}

/// A passage loaded for search.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredChunk {
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
        "SELECT id, path, size, mtime, hash FROM rag_documents WHERE collection_id = ?1",
    )?;
    let rows = statement.query_map([collection_id], |r| {
        Ok((
            r.get::<_, String>(1)?,
            DocumentState {
                id: r.get(0)?,
                size: to_u64(r.get(2)?),
                mtime: r.get(3)?,
                hash: to_u64(r.get(4)?),
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
        "SELECT c.name, c.root, c.embedding_model, c.updated_at,
                (SELECT COUNT(*) FROM rag_documents d WHERE d.collection_id = c.id),
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
            documents: u64::try_from(r.get::<_, i64>(4)?).unwrap_or(0),
            chunks: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Changes whenever the passages of a collection do: (count, highest id).
pub type ChunksVersion = (u64, i64);

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
        "SELECT d.path, k.location, k.text, k.embedding
         FROM rag_chunks k
         JOIN rag_documents d ON k.document_id = d.id
         JOIN rag_collections c ON d.collection_id = c.id
         WHERE c.name = ?1
         ORDER BY d.path, k.ordinal",
    )?;
    let rows = statement.query_map([collection], |r| {
        Ok(StoredChunk {
            path: r.get(0)?,
            location: r.get(1)?,
            text: r.get(2)?,
            vector: decode_vector(&r.get::<_, Vec<u8>>(3)?),
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
}
