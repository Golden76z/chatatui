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
    extract::{self, Extracted, FileKind, NO_TEXT_PDF},
    fingerprint,
    ocr::Ocr,
    store::{self, Collection},
};
use crate::storage::{Store, StoreError};

/// Passages embedded per request.
const BATCH: usize = 16;
/// Largest text or code file indexed.
const MAX_TEXT_BYTES: u64 = 2 * 1024 * 1024;
/// Largest PDF / office document indexed.
const MAX_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;

/// Name of the ignore file read in indexed folders, on top of `.gitignore`.
pub const IGNORE_FILE: &str = ".chatatuiignore";

/// What to index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexRequest {
    /// Collection name.
    pub collection: String,
    /// Folder to index. A bare collection name (`cours`) that is not a folder re-indexes
    /// that collection's folder.
    pub root: PathBuf,
    pub chunk_tokens: usize,
    /// File types to index: extensions, or `code` for every source file. `None` keeps
    /// the collection's current choice; an empty list means every supported type.
    pub types: Option<Vec<String>>,
    /// Gitignore-style patterns left out (`[rag] exclude`).
    pub exclude: Vec<String>,
    /// OCR for scanned PDFs, when available.
    pub ocr: Option<Ocr>,
}

/// Which files of a folder are indexed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScanFilter {
    pub types: Vec<String>,
    pub exclude: Vec<String>,
}

impl ScanFilter {
    fn accepts(&self, path: &Path, kind: FileKind) -> bool {
        if self.types.is_empty() {
            return true;
        }
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        self.types
            .iter()
            .any(|t| *t == extension || (t == "code" && kind == FileKind::Code))
    }
}

/// How a collection's folder differs from its index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Staleness {
    pub collection: String,
    pub added: usize,
    pub modified: usize,
    pub removed: usize,
    /// The folder no longer exists.
    pub missing_root: bool,
}

impl Staleness {
    /// `true` when the index is not up to date.
    pub fn is_stale(&self) -> bool {
        self.missing_root || self.added + self.modified + self.removed > 0
    }
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

/// Lists the supported files under `root` accepted by `filter`, following `.gitignore`
/// and `.chatatuiignore` rules.
fn scan(root: &Path, filter: &ScanFilter) -> Vec<Candidate> {
    let mut walker = ignore::WalkBuilder::new(root);
    walker
        .hidden(true)
        .git_ignore(true)
        .git_global(false)
        .require_git(false)
        .follow_links(false)
        .add_custom_ignore_filename(IGNORE_FILE);
    if !filter.exclude.is_empty() {
        let mut overrides = ignore::overrides::OverrideBuilder::new(root);
        for pattern in &filter.exclude {
            // Invalid patterns are skipped rather than failing the whole run.
            let _ = overrides.add(&format!("!{pattern}"));
        }
        if let Ok(overrides) = overrides.build() {
            walker.overrides(overrides);
        }
    }
    let mut files: Vec<Candidate> = walker
        .build()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_some_and(|t| t.is_file()))
        .filter_map(|entry| {
            let path = entry.into_path();
            let kind = FileKind::of(&path).filter(|k| filter.accepts(&path, *k))?;
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

/// The folder to index: `requested`, unless it is a bare collection name that is not a
/// folder, in which case the existing collection's folder.
fn resolve_root(
    requested: &Path,
    collection: &str,
    existing: Option<&store::CollectionSource>,
) -> PathBuf {
    match existing {
        Some(source) if !requested.is_dir() && requested == Path::new(collection) => {
            PathBuf::from(&source.root)
        }
        _ => requested.to_path_buf(),
    }
}

/// Compares every collection's folder with its index (blocking: walks the folders).
pub fn check_collections(db_path: &Path, exclude: &[String]) -> Result<Vec<Staleness>, String> {
    let conn = Store::open(db_path)
        .map_err(|e| format!("base de données : {e}"))?
        .into_connection();
    let sources = store::collection_sources(&conn).map_err(|e| format!("base de données : {e}"))?;
    let mut result = Vec::with_capacity(sources.len());
    for source in sources {
        let root = PathBuf::from(&source.root);
        let mut staleness = Staleness {
            collection: source.name.clone(),
            ..Staleness::default()
        };
        if !root.is_dir() {
            staleness.missing_root = true;
            result.push(staleness);
            continue;
        }
        let mut known =
            store::documents(&conn, source.id).map_err(|e| format!("base de données : {e}"))?;
        let filter = ScanFilter {
            types: source.types.clone(),
            exclude: exclude.to_vec(),
        };
        for candidate in scan(&root, &filter) {
            match known.remove(&candidate.relative) {
                None => staleness.added += 1,
                Some(state) if state.size != candidate.size || state.mtime != candidate.mtime => {
                    staleness.modified += 1;
                }
                Some(_) => {}
            }
        }
        staleness.removed = known.len();
        result.push(staleness);
    }
    Ok(result)
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
fn prepare(
    candidate: &Candidate,
    chunk_tokens: usize,
    ocr: Option<&Ocr>,
) -> Result<(u64, Vec<Passage>), String> {
    let limit = match candidate.kind {
        FileKind::Pdf | FileKind::Docx | FileKind::Odt => MAX_DOCUMENT_BYTES,
        FileKind::Markdown | FileKind::Text | FileKind::Code => MAX_TEXT_BYTES,
    };
    if candidate.size > limit {
        return Err(format!("trop gros ({} Mo)", candidate.size / (1024 * 1024)));
    }
    let bytes = std::fs::read(&candidate.path).map_err(|e| format!("illisible ({e})"))?;
    let hash = fingerprint(&bytes);
    let extracted = match extract::extract(candidate.kind, &bytes) {
        Err(error) if error == NO_TEXT_PDF => match ocr {
            Some(ocr) => {
                let extracted = Extracted {
                    kind: FileKind::Pdf,
                    pages: ocr.pdf(&bytes)?,
                };
                if extracted.is_empty() {
                    return Err(format!("{NO_TEXT_PDF} : rien de lisible, même avec l'OCR"));
                }
                extracted
            }
            None => return Err(no_ocr_reason()),
        },
        other => other?,
    };
    Ok((hash, chunk::split(&extracted, chunk_tokens)))
}

/// Why a scan was skipped when no OCR is available.
fn no_ocr_reason() -> String {
    format!("{NO_TEXT_PDF} : installez tesseract et poppler-utils pour l'OCR")
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
    report(IndexEvent::Scanning {
        collection: request.collection.clone(),
    });
    let db = tokio::task::spawn_blocking(move || Store::open(&db_path).map(Store::into_connection))
        .await
        .map_err(|e| format!("tâche interrompue ({e})"))?
        .map_err(|e| format!("base de données : {e}"))?;
    let db = Arc::new(Mutex::new(db));

    let name = request.collection.clone();
    let existing = with_db(&db, move |conn| store::collection_source(conn, &name)).await?;
    let root = resolve_root(&request.root, &request.collection, existing.as_ref());
    if !root.is_dir() {
        return Err(format!("{} n'est pas un dossier", root.display()));
    }
    // Stored absolute, so that updates and checks work from any directory.
    let root = root.canonicalize().unwrap_or(root);
    let types = request
        .types
        .clone()
        .or_else(|| existing.map(|e| e.types))
        .unwrap_or_default();

    let name = request.collection.clone();
    let root_text = root.to_string_lossy().into_owned();
    let model = embedder.model().to_owned();
    let stored_types = types.clone();
    let (collection, reset): (Collection, bool) = with_db(&db, move |conn| {
        let opened = store::open_collection(conn, &name, &root_text, &model, now())?;
        store::set_types(conn, opened.0.id, &stored_types)?;
        Ok(opened)
    })
    .await?;
    let collection_id = collection.id;
    let known = with_db(&db, move |conn| store::documents(conn, collection_id)).await?;

    let scan_root = root.clone();
    let filter = ScanFilter {
        types,
        exclude: request.exclude.clone(),
    };
    let files = tokio::task::spawn_blocking(move || scan(&scan_root, &filter))
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
        let previous = known.get(&candidate.relative).cloned();
        if let Some(p) = &previous
            && p.size == candidate.size
            && p.mtime == candidate.mtime
        {
            // A scan skipped for lack of OCR is read again once OCR is available.
            let retry = request.ocr.is_some() && p.skipped.as_deref() == Some(&no_ocr_reason());
            if !retry {
                match &p.skipped {
                    Some(reason) => summary
                        .skipped
                        .push((candidate.relative.clone(), reason.clone())),
                    None => summary.unchanged += 1,
                }
                continue;
            }
        }

        let chunk_tokens = request.chunk_tokens;
        let prepared = {
            let candidate = candidate.clone();
            let ocr = request.ocr.clone();
            tokio::task::spawn_blocking(move || prepare(&candidate, chunk_tokens, ocr.as_ref()))
                .await
                .map_err(|e| format!("tâche interrompue ({e})"))?
        };
        let (hash, passages) = match prepared {
            Ok(prepared) => prepared,
            Err(reason) => {
                summary
                    .skipped
                    .push((candidate.relative.clone(), reason.clone()));
                let relative = candidate.relative.clone();
                let (size, mtime) = (candidate.size, candidate.mtime);
                with_db(&db, move |conn| {
                    store::write_skipped(
                        conn,
                        collection_id,
                        &relative,
                        size,
                        mtime,
                        &reason,
                        now(),
                    )
                })
                .await?;
                continue;
            }
        };
        if let Some(previous) = previous
            .as_ref()
            .filter(|p| p.hash == hash && p.skipped.is_none())
        {
            let id = previous.id;
            let mtime = candidate.mtime;
            with_db(&db, move |conn| store::touch_document(conn, id, mtime)).await?;
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
        // A file indexed for the first time after being skipped counts as added.
        if previous.is_some_and(|p| p.skipped.is_none()) {
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
    use crate::rag::{embed::HashEmbedder, store::StoredChunk};

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
                ..IndexRequest::default()
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
            vec![("scan.pdf".to_owned(), no_ocr_reason())]
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
                ..IndexRequest::default()
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
            matches!(&events[..], [IndexEvent::Scanning { .. }, IndexEvent::Failed { error, .. }]
                if error.ends_with("n'est pas un dossier"))
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
                ..IndexRequest::default()
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
                ..IndexRequest::default()
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

    async fn index_request(
        setup: &Setup,
        embedder: Arc<HashEmbedder>,
        request: IndexRequest,
    ) -> IndexReport {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        run(
            request,
            setup.db.clone(),
            embedder,
            CancellationToken::new(),
            move |e| sink.lock().unwrap_or_else(PoisonError::into_inner).push(e),
        )
        .await;
        let events = events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        finished(&events)
    }

    fn request(setup: &Setup) -> IndexRequest {
        IndexRequest {
            collection: "docs".into(),
            root: setup.root.clone(),
            chunk_tokens: 800,
            ..IndexRequest::default()
        }
    }

    #[tokio::test]
    async fn types_filter_is_kept_until_changed() {
        let setup = setup();
        let embedder = Arc::new(HashEmbedder::default());
        let only_md = IndexRequest {
            types: Some(vec!["md".into(), "code".into()]),
            ..request(&setup)
        };
        let report = index_request(&setup, embedder.clone(), only_md).await;
        assert_eq!(
            (report.files, report.added),
            (2, 2),
            "notes.md + src/main.rs"
        );

        // Without --types, the collection keeps its filter.
        let report = index_request(&setup, embedder.clone(), request(&setup)).await;
        assert_eq!((report.files, report.unchanged), (2, 2));

        // --types all: every type again.
        let all = IndexRequest {
            types: Some(Vec::new()),
            ..request(&setup)
        };
        let report = index_request(&setup, embedder, all).await;
        assert_eq!(report.files, 6);
    }

    #[tokio::test]
    async fn excluded_patterns_and_chatatuiignore_are_left_out() {
        let setup = setup();
        fs::write(setup.root.join(IGNORE_FILE), "*.odt\n").expect("write");
        let excluded = IndexRequest {
            exclude: vec!["src/".into()],
            ..request(&setup)
        };
        let report = index_request(&setup, Arc::new(HashEmbedder::default()), excluded).await;
        let conn = Store::open(&setup.db).expect("db").into_connection();
        let source = store::collection_source(&conn, "docs")
            .expect("q")
            .expect("exists");
        let docs = store::documents(&conn, source.id).expect("docs");
        let mut paths: Vec<&str> = docs.keys().map(String::as_str).collect();
        paths.sort_unstable();
        assert_eq!(
            paths,
            vec!["cours.pdf", "notes.md", "plan.docx", "scan.pdf"]
        );
        assert_eq!(report.files, 4);
    }

    #[tokio::test]
    async fn skipped_files_are_not_retried_and_a_name_updates_the_collection() {
        let setup = setup();
        let embedder = Arc::new(HashEmbedder::default());
        let first = index_request(&setup, embedder.clone(), request(&setup)).await;
        assert_eq!(first.skipped.len(), 1);

        // `/index docs`: the bare name finds the collection's folder.
        let by_name = IndexRequest {
            root: PathBuf::from("docs"),
            ..request(&setup)
        };
        let calls = embedder.calls.load(Ordering::SeqCst);
        let again = index_request(&setup, embedder.clone(), by_name).await;
        assert_eq!(
            again.skipped, first.skipped,
            "still reported, with its reason"
        );
        assert_eq!(again.unchanged, first.added);
        assert_eq!(
            embedder.calls.load(Ordering::SeqCst),
            calls,
            "nothing embedded"
        );
    }

    #[tokio::test]
    async fn check_reports_what_changed_since_indexing() {
        let setup = setup();
        index_request(&setup, Arc::new(HashEmbedder::default()), request(&setup)).await;
        let fresh = check_collections(&setup.db, &[]).expect("check");
        assert_eq!(fresh.len(), 1);
        assert!(!fresh[0].is_stale(), "{fresh:?}");

        fs::write(
            setup.root.join("notes.md"),
            "# Notes\n\nNouveau contenu, plus long.\n",
        )
        .expect("write");
        fs::write(setup.root.join("ajout.txt"), "nouveau").expect("write");
        fs::remove_file(setup.root.join("plan.odt")).expect("remove");
        let changed = check_collections(&setup.db, &[]).expect("check");
        assert_eq!(
            (changed[0].added, changed[0].modified, changed[0].removed),
            (1, 1, 1)
        );

        fs::remove_dir_all(&setup.root).expect("remove");
        assert!(check_collections(&setup.db, &[]).expect("check")[0].missing_root);
    }

    #[tokio::test]
    async fn scans_are_read_with_ocr_once_it_is_available() {
        let Some(ocr) = Ocr::detect("eng") else {
            eprintln!("tesseract or pdftoppm missing: skipped");
            return;
        };
        let setup = setup();
        fs::copy(
            format!(
                "{}/tests/fixtures/rag/scan-text.pdf",
                env!("CARGO_MANIFEST_DIR")
            ),
            setup.root.join("scan-text.pdf"),
        )
        .expect("copy");
        let embedder = Arc::new(HashEmbedder::default());
        let without = index_request(&setup, embedder.clone(), request(&setup)).await;
        let skipped: Vec<&str> = without.skipped.iter().map(|s| s.0.as_str()).collect();
        assert_eq!(skipped, vec!["scan-text.pdf", "scan.pdf"]);

        // OCR now installed: the skipped scans are read again, unchanged files are not.
        let with = IndexRequest {
            ocr: Some(ocr),
            ..request(&setup)
        };
        let report = index_request(&setup, embedder, with).await;
        assert_eq!(report.added, 1, "{report:?}");
        assert_eq!(
            report.skipped.len(),
            1,
            "the blank page has nothing to read"
        );
        assert!(report.skipped[0].1.contains("même avec l'OCR"));
        let conn = Store::open(&setup.db).expect("db").into_connection();
        let chunks = store::load_chunks(&conn, "docs").expect("chunks");
        let scan: Vec<&StoredChunk> = chunks
            .iter()
            .filter(|c| c.path == "scan-text.pdf")
            .collect();
        assert_eq!(scan.len(), 2, "one passage per page");
        assert_eq!(scan[1].location, "p. 2");
        assert!(scan[1].text.contains("RefCell"));
    }
}
