//! Database schema and migrations, tracked with `PRAGMA user_version`.
//!
//! Each entry of [`MIGRATIONS`] upgrades the schema by one version. Never edit a released
//! migration: append a new one.

use rusqlite::Connection;

/// SQL scripts, in order. `MIGRATIONS[i]` upgrades from version `i` to `i + 1`.
const MIGRATIONS: &[&str] = &[
    // v1: conversations and their messages.
    r#"
    CREATE TABLE conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        model       TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL
    );
    CREATE INDEX conversations_by_update ON conversations (updated_at DESC);

    CREATE TABLE messages (
        conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
        seq             INTEGER NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        status          TEXT NOT NULL,
        error           TEXT,
        created_at      INTEGER NOT NULL,
        PRIMARY KEY (conversation_id, seq)
    );
    "#,
    // v2: provider of each conversation ('' for conversations created before providers).
    r#"
    ALTER TABLE conversations ADD COLUMN provider TEXT NOT NULL DEFAULT '';
    "#,
    // v3: where the model's context starts (/clear, /compact) and attachment origins.
    r#"
    ALTER TABLE conversations ADD COLUMN context_start INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE messages ADD COLUMN source TEXT;
    "#,
    // v4: document collections for retrieval (RAG), and the collection a conversation uses.
    r#"
    CREATE TABLE rag_collections (
        id              INTEGER PRIMARY KEY,
        name            TEXT NOT NULL UNIQUE,
        root            TEXT NOT NULL,
        embedding_model TEXT NOT NULL,
        dims            INTEGER NOT NULL DEFAULT 0,
        created_at      INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL
    );
    CREATE TABLE rag_documents (
        id            INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES rag_collections (id) ON DELETE CASCADE,
        path          TEXT NOT NULL,
        size          INTEGER NOT NULL,
        mtime         INTEGER NOT NULL,
        hash          INTEGER NOT NULL,
        indexed_at    INTEGER NOT NULL,
        UNIQUE (collection_id, path)
    );
    CREATE TABLE rag_chunks (
        id          INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES rag_documents (id) ON DELETE CASCADE,
        ordinal     INTEGER NOT NULL,
        location    TEXT NOT NULL,
        text        TEXT NOT NULL,
        embedding   BLOB NOT NULL
    );
    CREATE INDEX rag_chunks_by_document ON rag_chunks (document_id);
    ALTER TABLE conversations ADD COLUMN rag_collection TEXT;
    "#,
    // v5: passages cited by assistant replies (JSON list).
    r#"
    ALTER TABLE messages ADD COLUMN citations TEXT;
    "#,
    // v6: keyword index of the passages (hybrid search), per-collection file types, and
    // files that could not be indexed (with the reason, so they are not retried).
    r#"
    CREATE VIRTUAL TABLE rag_fts USING fts5(
        text,
        content = 'rag_chunks',
        content_rowid = 'id',
        tokenize = 'unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER rag_chunks_fts_insert AFTER INSERT ON rag_chunks BEGIN
        INSERT INTO rag_fts (rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER rag_chunks_fts_delete AFTER DELETE ON rag_chunks BEGIN
        INSERT INTO rag_fts (rag_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;
    INSERT INTO rag_fts (rag_fts) VALUES ('rebuild');
    ALTER TABLE rag_collections ADD COLUMN types TEXT NOT NULL DEFAULT '';
    ALTER TABLE rag_documents ADD COLUMN skipped TEXT;
    "#,
    // v7: full-text index of the messages, to search the history.
    r#"
    CREATE VIRTUAL TABLE messages_fts USING fts5(
        content,
        content = 'messages',
        content_rowid = 'rowid',
        tokenize = 'unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, content) VALUES (new.rowid, new.content);
    END;
    CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts (messages_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    END;
    CREATE TRIGGER messages_fts_update AFTER UPDATE OF content ON messages BEGIN
        INSERT INTO messages_fts (messages_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
        INSERT INTO messages_fts (rowid, content) VALUES (new.rowid, new.content);
    END;
    INSERT INTO messages_fts (messages_fts) VALUES ('rebuild');
    "#,
    // v8: named system prompt of a conversation (`/persona`).
    r#"
    ALTER TABLE conversations ADD COLUMN persona TEXT;
    "#,
    // v9: attached images, kept out of `content` (and of the full-text index).
    r#"
    ALTER TABLE messages ADD COLUMN image_type TEXT;
    ALTER TABLE messages ADD COLUMN image_data TEXT;
    "#,
    // v10: versions replaced by /edit and /retry, kept to switch back to them.
    r#"
    CREATE TABLE message_tails (
        id              INTEGER PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
        after_seq       INTEGER,
        number          INTEGER NOT NULL,
        messages        TEXT NOT NULL,
        created_at      INTEGER NOT NULL
    );
    CREATE INDEX message_tails_by_conversation ON message_tails (conversation_id);
    "#,
];

/// Latest schema version.
pub const LATEST_VERSION: usize = MIGRATIONS.len();

/// Current schema version of the database.
pub fn version(conn: &Connection) -> rusqlite::Result<usize> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    Ok(usize::try_from(version).unwrap_or(0))
}

/// Applies the missing migrations, each in its own transaction.
///
/// Several connections may open the database at once (the storage worker, the indexer,
/// the startup check, another chatatui): each step takes the write lock first and reads
/// the version inside its transaction, so a migration is never applied twice.
pub fn migrate(conn: &mut Connection) -> rusqlite::Result<()> {
    loop {
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let current = version(&tx)?;
        let Some(script) = MIGRATIONS.get(current) else {
            return tx.commit();
        };
        tx.execute_batch(script)?;
        let next = i64::try_from(current + 1).unwrap_or(i64::MAX);
        tx.pragma_update(None, "user_version", next)?;
        tx.commit()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_a_new_database_to_the_latest_version() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        assert_eq!(version(&conn).expect("version"), 0);
        migrate(&mut conn).expect("migrate");
        assert_eq!(version(&conn).expect("version"), LATEST_VERSION);
    }

    #[test]
    fn v1_databases_are_upgraded_in_place() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        let tx = conn.transaction().expect("tx");
        tx.execute_batch(MIGRATIONS[0]).expect("v1");
        tx.execute(
            "INSERT INTO conversations VALUES ('c', 'titre', 'llama3.2', 1, 1)",
            [],
        )
        .expect("old row");
        tx.pragma_update(None, "user_version", 1).expect("version");
        tx.commit().expect("commit");

        migrate(&mut conn).expect("migrate");
        let provider: String = conn
            .query_row(
                "SELECT provider FROM conversations WHERE id = 'c'",
                [],
                |r| r.get(0),
            )
            .expect("column added");
        assert_eq!(provider, "");
        assert_eq!(version(&conn).expect("version"), LATEST_VERSION);
    }

    #[test]
    fn migrating_twice_is_a_no_op() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        migrate(&mut conn).expect("first");
        migrate(&mut conn).expect("second");
        assert_eq!(version(&conn).expect("version"), LATEST_VERSION);
    }

    #[test]
    fn concurrent_openers_migrate_once() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("db.sqlite");
        let openers: Vec<_> = (0..6)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || crate::storage::Store::open(&path).map(|_| ()))
            })
            .collect();
        for opener in openers {
            let opened = opener.join().expect("no panic");
            assert!(opened.is_ok(), "{opened:?}");
        }
        let conn = Connection::open(&path).expect("open");
        assert_eq!(version(&conn).expect("version"), LATEST_VERSION);
    }
}
