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
];

/// Latest schema version.
pub const LATEST_VERSION: usize = MIGRATIONS.len();

/// Current schema version of the database.
pub fn version(conn: &Connection) -> rusqlite::Result<usize> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    Ok(usize::try_from(version).unwrap_or(0))
}

/// Applies the missing migrations, each in its own transaction.
pub fn migrate(conn: &mut Connection) -> rusqlite::Result<()> {
    let current = version(conn)?;
    for (index, script) in MIGRATIONS.iter().enumerate().skip(current) {
        let tx = conn.transaction()?;
        tx.execute_batch(script)?;
        let next = i64::try_from(index + 1).unwrap_or(i64::MAX);
        tx.pragma_update(None, "user_version", next)?;
        tx.commit()?;
    }
    Ok(())
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
}
