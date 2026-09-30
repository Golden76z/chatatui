//! The background job behind `/index`: walks a folder, extracts and splits the supported
//! files, embeds the passages and stores them.
//!
//! Incremental: a file whose size and modification time did not change is skipped, one
//! whose content hash did not change only gets its time updated, and files that
//! disappeared are removed from the index. `.gitignore` rules and hidden files are
//! respected. Blocking work (disk, parsing, SQLite) runs on blocking threads; the job can
//! be cancelled between batches.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::Connection;
use tokio_util::sync::CancellationToken;

use super::{
    chunk::{self, Passage},
    embed::Embedder,
    extract::{self, FileKind},
    fingerprint,
    store::{self, Collection},
};
use crate::storage::{Store, StoreError};

/// Passages embedded per request.
const BATCH: usize = 16;
/// Largest text or code file indexed.
const MAX_TEXT_BYTES: u64 = 2 * 1024 * 1024;
/// Largest PDF / office document indexed.
const MAX_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;

/// What to index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexRequest {
    /// Collection name.
    pub collection: String,
    pub root: PathBuf,
    pub chunk_tokens: usize,
}

/// Outcome of a completed run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexReport {
    pub collection: String,
    /// Supported files found.
    pub files: usize,
    pub added: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub removed: usize,
    /// Files that could not be indexed: `(path, reason)`.
    pub skipped: Vec<(String, String)>,
    /// Passages written during this run.
    pub passages: usize,
    /// The collection was emptied first because the embedding model changed.
    pub reset: bool,
}

/// Progress of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexEvent {
    /// Files are being listed.
    Scanning {
        collection: String,
    },
    /// `done` of `total` files processed; `current` is being processed.
    Progress {
        collection: String,
        done: usize,
        total: usize,
        current: String,
    },
    Finished(IndexReport),
    /// The run stopped on an error (user-facing message).
    Failed {
        collection: String,
        error: String,
    },
    Cancelled {
        collection: String,
    },
}

/// A supported file found in the folder.
#[derive(Clone, Debug)]
struct Candidate {
    path: PathBuf,
    /// Relative to the root, with `/` separators.
    relative: String,
    kind: FileKind,
    size: u64,
    mtime: i64,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// Lists the supported files under `root`, following `.gitignore` rules.
fn scan(root: &Path) -> Vec<Candidate> {
    let mut files: Vec<Candidate> = ignore::WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(false)
        .require_git(false)
        .follow_links(false)
        .build()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_some_and(|t| t.is_file()))
        .filter_map(|entry| {
            let path = entry.into_path();
            let kind = FileKind::of(&path)?;
            let metadata = std::fs::metadata(&path).ok()?;
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0));
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            Some(Candidate {
                path,
                relative,
                kind,
                size: metadata.len(),
                mtime,
            })
        })
        .collect();
    files.sort_by(|a, b| a.relative.cmp(&b.relative));
    files
}

/// Runs `f` on a blocking thread with the database connection.
async fn with_db<T: Send + 'static>(
    db: &Arc<Mutex<Connection>>,
    f: impl FnOnce(&mut Connection) -> Result<T, StoreError> + Send + 'static,
) -> Result<T, String> {
    let db = db.clone();
    tokio::task::spawn_blocking(move || {
        let mut conn = db.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut conn)
    })
    .await
    .map_err(|e| format!("tâche interrompue ({e})"))?
    .map_err(|e| format!("base de données : {e}"))
}

/// Reads, extracts and splits one file (blocking).
fn prepare(candidate: &Candidate, chunk_tokens: usize) -> Result<(u64, Vec<Passage>), String> {
    let limit = match candidate.kind {
        FileKind::Pdf | FileKind::Docx | FileKind::Odt => MAX_DOCUMENT_BYTES,
        FileKind::Markdown | FileKind::Text | FileKind::Code => MAX_TEXT_BYTES,
    };
    if candidate.size > limit {
        return Err(format!("trop gros ({} Mo)", candidate.size / (1024 * 1024)));
    }
    let bytes = std::fs::read(&candidate.path).map_err(|e| format!("illisible ({e})"))?;
    let hash = fingerprint(&bytes);
    let extracted = extract::extract(candidate.kind, &bytes)?;
    Ok((hash, chunk::split(&extracted, chunk_tokens)))
}

/// Indexes `request.root` into the database at `db_path`, reporting progress.
pub async fn run(
    request: IndexRequest,
    db_path: PathBuf,
    embedder: Arc<dyn Embedder>,
    cancel: CancellationToken,
    report: impl Fn(IndexEvent) + Send + Sync,
) {
    let collection = request.collection.clone();
    let outcome = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(None),
        result = index(&request, db_path, embedder.as_ref(), &report) => result.map_err(Some),
    };
    report(match outcome {
        Ok(summary) => IndexEvent::Finished(summary),
        Err(None) => IndexEvent::Cancelled { collection },
        Err(Some(error)) => IndexEvent::Failed { collection, error },
    });
}

async fn index(
    request: &IndexRequest,
    db_path: PathBuf,
    embedder: &dyn Embedder,
    report: &(impl Fn(IndexEvent) + Send + Sync),
) -> Result<IndexReport, String> {
    let root = request.root.clone();
    if !root.is_dir() {
        return Err(format!("{} n'est pas un dossier", root.display()));
    }
    report(IndexEvent::Scanning {
        collection: request.collection.clone(),
    });

    let db = tokio::task::spawn_blocking(move || Store::open(&db_path).map(Store::into_connection))
        .await
        .map_err(|e| format!("tâche interrompue ({e})"))?
        .map_err(|e| format!("base de données : {e}"))?;
    let db = Arc::new(Mutex::new(db));

    let name = request.collection.clone();
    let root_text = root.to_string_lossy().into_owned();
    let model = embedder.model().to_owned();
    let (collection, reset): (Collection, bool) = with_db(&db, move |conn| {
        store::open_collection(conn, &name, &root_text, &model, now())
    })
    .await?;
    let collection_id = collection.id;
    let known = with_db(&db, move |conn| store::documents(conn, collection_id)).await?;

    let scan_root = root.clone();
    let files = tokio::task::spawn_blocking(move || scan(&scan_root))
        .await
        .map_err(|e| format!("tâche interrompue ({e})"))?;

    let mut summary = IndexReport {
        collection: request.collection.clone(),
        files: files.len(),
        reset,
        ..IndexReport::default()
    };
    let mut seen: HashSet<String> = HashSet::new();
    let total = files.len();

    for (done, candidate) in files.into_iter().enumerate() {
        report(IndexEvent::Progress {
            collection: request.collection.clone(),
            done,
            total,
            current: candidate.relative.clone(),
        });
        seen.insert(candidate.relative.clone());
        let previous = known.get(&candidate.relative).copied();
        if previous.is_some_and(|p| p.size == candidate.size && p.mtime == candidate.mtime) {
            summary.unchanged += 1;
            continue;
        }

        let chunk_tokens = request.chunk_tokens;
        let prepared = {
            let candidate = candidate.clone();
            tokio::task::spawn_blocking(move || prepare(&candidate, chunk_tokens))
                .await
                .map_err(|e| format!("tâche interrompue ({e})"))?
        };
        let (hash, passages) = match prepared {
            Ok(prepared) => prepared,
            Err(reason) => {
                summary.skipped.push((candidate.relative.clone(), reason));
                if let Some(previous) = previous {
                    with_db(&db, move |conn| store::remove_document(conn, previous.id)).await?;
                }
                continue;
            }
        };
        if let Some(previous) = previous.filter(|p| p.hash == hash) {
            let mtime = candidate.mtime;
            with_db(&db, move |conn| {
                store::touch_document(conn, previous.id, mtime)
            })
            .await?;
            summary.unchanged += 1;
            continue;
        }

        let file_name = candidate
            .relative
            .rsplit('/')
            .next()
            .unwrap_or(&candidate.relative)
            .to_owned();
        let mut vectors = Vec::with_capacity(passages.len());
        for batch in passages.chunks(BATCH) {
            let texts: Vec<String> = batch.iter().map(|p| p.embedding_text(&file_name)).collect();
            let embedded = embedder
                .embed(&texts)
                .await
                .map_err(|e| format!("embeddings : {e}"))?;
            vectors.extend(embedded);
        }

        summary.passages += passages.len();
        if previous.is_some() {
            summary.updated += 1;
        } else {
            summary.added += 1;
        }
        let relative = candidate.relative.clone();
        let (size, mtime) = (candidate.size, candidate.mtime);
        with_db(&db, move |conn| {
            store::write_document(
                conn,
                collection_id,
                &relative,
                size,
                mtime,
                hash,
                &passages,
                &vectors,
                now(),
            )
        })
        .await?;
    }

    // Files that disappeared (or became ignored).
    for (path, state) in known {
        if !seen.contains(&path) {
            with_db(&db, move |conn| store::remove_document(conn, state.id)).await?;
            summary.removed += 1;
        }
    }
    with_db(&db, move |conn| {
        store::finish_collection(conn, collection_id, now())
    })
    .await?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::atomic::Ordering};

    use super::*;
    use crate::rag::embed::HashEmbedder;

    struct Setup {
        _dir: tempfile::TempDir,
        root: PathBuf,
        db: PathBuf,
    }

    fn setup() -> Setup {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("docs");
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::create_dir_all(root.join("target")).expect("mkdir");
        let fixtures = format!("{}/tests/fixtures/rag", env!("CARGO_MANIFEST_DIR"));
        for name in ["cours.pdf", "plan.docx", "plan.odt", "scan.pdf"] {
            fs::copy(format!("{fixtures}/{name}"), root.join(name)).expect("copy");
        }
        fs::write(
            root.join("notes.md"),
            "# Notes\n\nLes closures capturent leur environnement.\n",
        )
        .expect("write");
        fs::write(
            root.join("src/main.rs"),
            "fn main() {\n    println!(\"bonjour\");\n}\n",
        )
        .expect("write");
        fs::write(root.join("target/build.rs"), "ignored").expect("write");
        fs::write(root.join(".gitignore"), "target/\n").expect("write");
        fs::write(root.join(".secret.md"), "caché").expect("write");
        fs::write(root.join("photo.jpg"), [0xff, 0xd8]).expect("write");
        Setup {
            db: dir.path().join("chatatui.db"),
            root,
            _dir: dir,
        }
    }

    async fn index_once(setup: &Setup, embedder: Arc<HashEmbedder>) -> Vec<IndexEvent> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        run(
            IndexRequest {
                collection: "docs".into(),
                root: setup.root.clone(),
                chunk_tokens: 800,
            },
            setup.db.clone(),
            embedder,
            CancellationToken::new(),
            move |e| sink.lock().unwrap_or_else(PoisonError::into_inner).push(e),
        )
        .await;
        events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn finished(events: &[IndexEvent]) -> IndexReport {
        match events.last() {
            Some(IndexEvent::Finished(report)) => report.clone(),
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn indexes_supported_files_and_skips_the_rest() {
        let setup = setup();
        let events = index_once(&setup, Arc::new(HashEmbedder::default())).await;
        let report = finished(&events);
        assert_eq!(
            report.files, 6,
            "pdf×2, docx, odt, md, rs; not hidden/ignored/jpg"
        );
        assert_eq!(report.added, 5);
        assert_eq!(
            report.skipped,
            vec![("scan.pdf".to_owned(), "PDF sans texte (scan ?)".to_owned())]
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, IndexEvent::Progress { total: 6, .. }))
        );

        let conn = Store::open(&setup.db).expect("db").into_connection();
        let chunks = store::load_chunks(&conn, "docs").expect("chunks");
        let pdf_pages: Vec<&str> = chunks
            .iter()
            .filter(|c| c.path == "cours.pdf")
            .map(|c| c.location.as_str())
            .collect();
        assert_eq!(pdf_pages, ["p. 1", "p. 2"]);
        assert!(
            chunks
                .iter()
                .any(|c| c.path == "src/main.rs" && c.location == "L1-3")
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.path == "plan.docx" && c.location == "§ Séance 1 : ownership")
        );
        assert!(!chunks.iter().any(|c| c.path.starts_with("target/")));
    }

    #[tokio::test]
    async fn second_run_only_reindexes_what_changed() {
        let setup = setup();
        index_once(&setup, Arc::new(HashEmbedder::default())).await;

        let embedder = Arc::new(HashEmbedder::default());
        let report = finished(&index_once(&setup, embedder.clone()).await);
        assert_eq!((report.added, report.updated, report.removed), (0, 0, 0));
        assert_eq!(report.unchanged, 5);
        assert_eq!(
            embedder.calls.load(Ordering::SeqCst),
            0,
            "nothing embedded again"
        );

        // Modify one file (size changes), delete another.
        fs::write(
            setup.root.join("notes.md"),
            "# Notes\n\nNouveau contenu plus long.\n",
        )
        .expect("write");
        fs::remove_file(setup.root.join("plan.odt")).expect("remove");
        let report = finished(&index_once(&setup, Arc::new(HashEmbedder::default())).await);
        assert_eq!(
            (report.updated, report.removed, report.unchanged),
            (1, 1, 3)
        );
        let conn = Store::open(&setup.db).expect("db").into_connection();
        let chunks = store::load_chunks(&conn, "docs").expect("chunks");
        assert!(chunks.iter().any(|c| c.text.contains("Nouveau contenu")));
        assert!(!chunks.iter().any(|c| c.path == "plan.odt"));
    }

    #[tokio::test]
    async fn missing_folder_fails_cleanly() {
        let setup = setup();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        run(
            IndexRequest {
                collection: "x".into(),
                root: setup.root.join("nope"),
                chunk_tokens: 800,
            },
            setup.db.clone(),
            Arc::new(HashEmbedder::default()),
            CancellationToken::new(),
            move |e| sink.lock().unwrap_or_else(PoisonError::into_inner).push(e),
        )
        .await;
        let events = events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        assert!(
            matches!(&events[..], [IndexEvent::Failed { error, .. }] if error.ends_with("n'est pas un dossier"))
        );
    }

    #[tokio::test]
    async fn cancelled_run_reports_it() {
        let setup = setup();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        run(
            IndexRequest {
                collection: "docs".into(),
                root: setup.root.clone(),
                chunk_tokens: 800,
            },
            setup.db.clone(),
            Arc::new(HashEmbedder::default()),
            cancel,
            move |e| sink.lock().unwrap_or_else(PoisonError::into_inner).push(e),
        )
        .await;
        let events = events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        assert_eq!(
            events.last(),
            Some(&IndexEvent::Cancelled {
                collection: "docs".into()
            })
        );
    }

    #[tokio::test]
    async fn embedding_errors_stop_the_run() {
        struct Broken;
        #[async_trait::async_trait]
        impl Embedder for Broken {
            fn model(&self) -> &str {
                "broken"
            }
            async fn embed(&self, _: &[String]) -> Result<Vec<Vec<f32>>, crate::llm::LlmError> {
                Err(crate::llm::LlmError::Http {
                    status: 404,
                    message: "model \"bge-m3\" not found, try pulling it first".into(),
                })
            }
        }
        let setup = setup();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        run(
            IndexRequest {
                collection: "docs".into(),
                root: setup.root.clone(),
                chunk_tokens: 800,
            },
            setup.db.clone(),
            Arc::new(Broken),
            CancellationToken::new(),
            move |e| sink.lock().unwrap_or_else(PoisonError::into_inner).push(e),
        )
        .await;
        let events = events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        assert!(matches!(
            events.last(),
            Some(IndexEvent::Failed { error, .. }) if error.contains("not found, try pulling it first")
        ));
    }
}
