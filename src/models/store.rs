//! The inventory of downloaded models (table `local_models`, schema v11).
//!
//! Like [`crate::rag::store`], these are free functions over a plain `Connection`, so both
//! the storage worker (listing, deleting) and the download job (saving, on its own
//! connection) can call them.

use rusqlite::{Connection, OptionalExtension, params};

use crate::storage::StoreError;

/// A model on this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModel {
    /// `owner/name` on HuggingFace.
    pub repo: String,
    /// Branch or commit the file came from.
    pub revision: String,
    /// File name inside the repository.
    pub file: String,
    /// Absolute path on disk.
    pub path: String,
    pub bytes: u64,
    /// Verified sha256; `None` means the Hub exposed none, so nothing was checked.
    pub sha256: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub context_length: Option<u64>,
    pub parameters: Option<u64>,
    /// Unix seconds.
    pub downloaded_at: i64,
}

impl LocalModel {
    /// How the model is named in the UI: `owner/name · Q4_K_M`.
    pub fn label(&self) -> String {
        match &self.quantization {
            Some(quantization) => format!("{} · {quantization}", self.repo),
            None => self.repo.clone(),
        }
    }
}

// rusqlite only implements `ToSql`/`FromSql` for `u64` behind the `fallible_uint` feature,
// which this crate does not enable (see `src/rag/store.rs` for the same pattern). SQLite
// integers are 64-bit two's complement, so round-tripping through `i64` is lossless for
// every `u64` value; `bytes`, `context_length` and `parameters` never need to be compared
// or ordered as signed, only stored and read back.
fn to_i64(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn to_u64(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

/// Inserts the model, replacing any row for the same `(repo, file)`.
pub fn save(conn: &Connection, model: &LocalModel) -> Result<(), StoreError> {
    conn.execute(
        "INSERT INTO local_models
             (repo, revision, file, path, bytes, sha256, architecture, quantization,
              context_length, parameters, downloaded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
         ON CONFLICT (repo, file) DO UPDATE SET
             revision = excluded.revision,
             path = excluded.path,
             bytes = excluded.bytes,
             sha256 = excluded.sha256,
             architecture = excluded.architecture,
             quantization = excluded.quantization,
             context_length = excluded.context_length,
             parameters = excluded.parameters,
             downloaded_at = excluded.downloaded_at",
        params![
            model.repo,
            model.revision,
            model.file,
            model.path,
            to_i64(model.bytes),
            model.sha256,
            model.architecture,
            model.quantization,
            model.context_length.map(to_i64),
            model.parameters.map(to_i64),
            model.downloaded_at,
        ],
    )?;
    Ok(())
}

/// Every model, most recently downloaded first.
pub fn list(conn: &Connection) -> Result<Vec<LocalModel>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT repo, revision, file, path, bytes, sha256, architecture, quantization,
                context_length, parameters, downloaded_at
           FROM local_models
          ORDER BY downloaded_at DESC, repo, file",
    )?;
    let rows = statement.query_map([], row_to_model)?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

/// Removes a model's row, returning what it held (`None` when there was none).
pub fn delete(conn: &Connection, repo: &str, file: &str) -> Result<Option<LocalModel>, StoreError> {
    let found = conn
        .query_row(
            "SELECT repo, revision, file, path, bytes, sha256, architecture, quantization,
                    context_length, parameters, downloaded_at
               FROM local_models WHERE repo = ?1 AND file = ?2",
            params![repo, file],
            row_to_model,
        )
        .optional()?;
    if found.is_some() {
        conn.execute(
            "DELETE FROM local_models WHERE repo = ?1 AND file = ?2",
            params![repo, file],
        )?;
    }
    Ok(found)
}

fn row_to_model(row: &rusqlite::Row<'_>) -> rusqlite::Result<LocalModel> {
    Ok(LocalModel {
        repo: row.get(0)?,
        revision: row.get(1)?,
        file: row.get(2)?,
        path: row.get(3)?,
        bytes: to_u64(row.get(4)?),
        sha256: row.get(5)?,
        architecture: row.get(6)?,
        quantization: row.get(7)?,
        context_length: row.get::<_, Option<i64>>(8)?.map(to_u64),
        parameters: row.get::<_, Option<i64>>(9)?.map(to_u64),
        downloaded_at: row.get(10)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connection() -> Connection {
        // `storage::schema` is private; go through `Store` like `rag::store`'s tests do.
        crate::storage::Store::open_in_memory()
            .expect("store")
            .into_connection()
    }

    fn model(file: &str) -> LocalModel {
        LocalModel {
            repo: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_owned(),
            revision: "main".to_owned(),
            file: file.to_owned(),
            path: format!("/models/bartowski/Qwen2.5-7B-Instruct-GGUF/{file}"),
            bytes: 4_431_401_088,
            sha256: Some("abc".to_owned()),
            architecture: Some("qwen2".to_owned()),
            quantization: Some("Q4_K_M".to_owned()),
            context_length: Some(32768),
            parameters: Some(7_615_616_512),
            downloaded_at: 1_760_000_000,
        }
    }

    #[test]
    fn saves_and_lists_a_model() {
        let conn = connection();

        save(&conn, &model("a-Q4_K_M.gguf")).expect("saved");
        let models = list(&conn).expect("listed");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0], model("a-Q4_K_M.gguf"));
    }

    #[test]
    fn lists_the_most_recent_first() {
        let conn = connection();
        let mut older = model("old.gguf");
        older.downloaded_at = 1_700_000_000;
        save(&conn, &older).expect("saved");
        save(&conn, &model("new.gguf")).expect("saved");

        let models = list(&conn).expect("listed");

        assert_eq!(models[0].file, "new.gguf");
        assert_eq!(models[1].file, "old.gguf");
    }

    #[test]
    fn downloading_again_replaces_the_row_rather_than_adding_one() {
        let conn = connection();
        save(&conn, &model("a.gguf")).expect("saved");
        let mut again = model("a.gguf");
        again.revision = "d34db33f".to_owned();
        again.sha256 = None;
        again.bytes = 999;

        save(&conn, &again).expect("saved again");
        let models = list(&conn).expect("listed");

        assert_eq!(models.len(), 1, "one file, one row");
        assert_eq!(models[0].revision, "d34db33f");
        assert_eq!(models[0].sha256, None);
        assert_eq!(models[0].bytes, 999);
    }

    #[test]
    fn deletes_a_model_and_returns_what_it_held() {
        let conn = connection();
        save(&conn, &model("a.gguf")).expect("saved");

        let removed =
            delete(&conn, "bartowski/Qwen2.5-7B-Instruct-GGUF", "a.gguf").expect("deleted");

        assert_eq!(removed.as_ref().map(|m| m.file.as_str()), Some("a.gguf"));
        assert!(list(&conn).expect("listed").is_empty());
    }

    #[test]
    fn deleting_an_unknown_model_is_not_an_error() {
        let conn = connection();

        let removed = delete(&conn, "owner/name", "absent.gguf").expect("no error");

        assert_eq!(removed, None);
    }

    #[test]
    fn a_model_whose_metadata_is_unknown_round_trips_as_none() {
        let conn = connection();
        let bare = LocalModel {
            sha256: None,
            architecture: None,
            quantization: None,
            context_length: None,
            parameters: None,
            ..model("bare.gguf")
        };

        save(&conn, &bare).expect("saved");

        assert_eq!(list(&conn).expect("listed")[0], bare);
    }
}
