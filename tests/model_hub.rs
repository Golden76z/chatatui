//! The HuggingFace client and the download job against a minimal local HTTP server.

use std::{path::Path, time::Duration};

use chatatui::models::{
    ModelError,
    download::{self, PullEvent, PullRequest},
    hub::{Hub, RemoteFile},
    store,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tokio_util::sync::CancellationToken;

/// How a test server answers one request.
enum Reply {
    /// Status line plus headers, then this body.
    Full(&'static str, Vec<u8>),
    /// Headers claiming the full length, then the first `cut` bytes in two bursts 200 ms
    /// apart, then silence: the transfer never finishes, and the pause is long enough for
    /// the client's throttled progress callback to fire at least once with bytes written.
    Stall(&'static str, Vec<u8>, usize),
}

/// Serves `replies` in order, one per connection. Returns the base URL.
async fn serve(replies: Vec<Reply>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move {
        for reply in replies {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_head(&mut socket).await;
            // Range requests are honoured only by a reply that claims 206: a 200 must send
            // the whole body from byte zero even when a Range header was sent, exactly like
            // a real server that ignores it.
            let offset = request
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("range: bytes=")
                        .and_then(|v| v.trim().trim_end_matches('-').parse::<usize>().ok())
                })
                .unwrap_or(0);
            match reply {
                Reply::Full(head, full) => {
                    let body = if head.starts_with("HTTP/1.1 206") {
                        full.get(offset.min(full.len())..)
                            .unwrap_or_default()
                            .to_vec()
                    } else {
                        full
                    };
                    let head = head.replace("{len}", &body.len().to_string());
                    socket.write_all(head.as_bytes()).await.expect("head");
                    socket.write_all(&body).await.expect("body");
                    socket.flush().await.expect("flush");
                }
                Reply::Stall(head, full, cut) => {
                    // Declares the full length but only ever writes `cut` bytes, then holds
                    // the connection open instead of closing it: a short, well-formed
                    // response would finish before the test can cancel anything, so the only
                    // way to simulate a stalled transfer is to make the client wait for
                    // bytes that never arrive.
                    let head = head.replace("{len}", &full.len().to_string());
                    socket.write_all(head.as_bytes()).await.expect("head");
                    socket.write_all(&full[..cut / 2]).await.expect("body");
                    socket.flush().await.expect("flush");
                    // Longer than the client's progress throttle, so the next burst is
                    // reported rather than swallowed.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    socket.write_all(&full[cut / 2..cut]).await.expect("body");
                    socket.flush().await.expect("flush");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });
    base
}

async fn read_head(socket: &mut tokio::net::TcpStream) -> String {
    let mut data = Vec::new();
    let mut buffer = [0u8; 2048];
    loop {
        let read = socket.read(&mut buffer).await.expect("read");
        data.extend_from_slice(&buffer[..read]);
        let text = String::from_utf8_lossy(&data).into_owned();
        if read == 0 || text.contains("\r\n\r\n") {
            return text;
        }
    }
}

const OK_JSON: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n";
const OK_BYTES: &str = "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n";
const PARTIAL: &str =
    "HTTP/1.1 206 Partial Content\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n";
const NOT_FOUND: &str = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const RANGE_REFUSED: &str =
    "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

fn hub(base: &str) -> Hub {
    Hub::new(Some(base.to_owned()), None, Duration::from_secs(2)).expect("hub")
}

fn sha256_of(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn payload() -> Vec<u8> {
    (0..64_000u32).flat_map(|n| n.to_le_bytes()).collect()
}

/// The smallest valid GGUF file: the magic, version 3, no tensor, one string value. The
/// download job reads the header before recording the model, so a job that must reach the
/// inventory needs a file that parses.
fn gguf_payload() -> Vec<u8> {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    string(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    string(&mut out, "llama");
    out
}

/// Runs the whole download job against `base`, storing models under `<dir>/models` and
/// recording them in `database`. Returns everything it reported.
async fn run_pull(base: &str, dir: &Path, database: &Path, file: RemoteFile) -> Vec<PullEvent> {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let collect = std::sync::Arc::clone(&events);
    download::run(
        PullRequest {
            repo: "owner/name".to_owned(),
            revision: "main".to_owned(),
            file,
            dir: dir.join("models"),
        },
        database.to_path_buf(),
        std::sync::Arc::new(hub(base)),
        CancellationToken::new(),
        move |event| {
            if let Ok(mut log) = collect.lock() {
                log.push(event);
            }
        },
    )
    .await;
    events.lock().expect("lock").clone()
}

/// Where the job puts the finished file and the part file it resumes from.
fn store_paths(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    download::paths(&dir.join("models"), "owner/name", "model-Q4_K_M.gguf").expect("paths")
}

/// Every model the inventory in `database` holds.
fn inventory(database: &Path) -> Vec<store::LocalModel> {
    let connection = chatatui::storage::Store::open(database)
        .expect("store")
        .into_connection();
    store::list(&connection).expect("listed")
}

fn remote(bytes: &[u8], sha256: Option<String>) -> RemoteFile {
    RemoteFile {
        path: "model-Q4_K_M.gguf".to_owned(),
        bytes: bytes.len() as u64,
        sha256,
    }
}

#[tokio::test]
async fn lists_the_gguf_files_of_a_repository() {
    let body = br#"[{"type":"file","path":"a-Q4_K_M.gguf","size":10,
                     "lfs":{"oid":"sha256:dead","size":4000}},
                    {"type":"file","path":"README.md","size":12}]"#
        .to_vec();
    let base = serve(vec![Reply::Full(OK_JSON, body)]).await;

    let files = hub(&base)
        .list_gguf("owner/name", "main")
        .await
        .expect("listed");

    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "a-Q4_K_M.gguf");
    assert_eq!(files[0].bytes, 4000);
    assert_eq!(files[0].sha256.as_deref(), Some("dead"));
}

/// Review Focus 2: a repository with no GGUF must be distinguishable from a failure.
#[tokio::test]
async fn a_repository_without_gguf_lists_nothing_without_failing() {
    let body = br#"[{"type":"file","path":"model.safetensors","size":99}]"#.to_vec();
    let base = serve(vec![Reply::Full(OK_JSON, body)]).await;

    let files = hub(&base)
        .list_gguf("owner/name", "main")
        .await
        .expect("listed");

    assert!(files.is_empty());
}

/// The spec promises the error names the real reason, and a repository whose GGUF are all
/// shards is not an empty one.
#[tokio::test]
async fn a_repository_whose_gguf_are_only_shards_says_so() {
    for body in [
        // Shards in a subdirectory, as HuggingFace lays them out.
        br#"[{"type":"file","path":"Q4/m-00001-of-00002.gguf","size":10},
             {"type":"file","path":"Q4/m-00002-of-00002.gguf","size":10}]"#
            .to_vec(),
        // And the symmetrical case: shards at the root, which a bare name check lets through.
        br#"[{"type":"file","path":"m-00001-of-00002.gguf","size":10},
             {"type":"file","path":"m-00002-of-00002.gguf","size":10}]"#
            .to_vec(),
    ] {
        let base = serve(vec![Reply::Full(OK_JSON, body)]).await;

        let error = hub(&base)
            .list_gguf("owner/name", "main")
            .await
            .expect_err("refused");

        assert!(
            error.to_string().contains("plusieurs fichiers"),
            "{error} should name the real reason"
        );
    }
}

#[tokio::test]
async fn an_unknown_repository_is_not_found() {
    let base = serve(vec![Reply::Full(NOT_FOUND, Vec::new())]).await;

    let error = hub(&base)
        .list_gguf("owner/name", "main")
        .await
        .expect_err("refused");

    assert_eq!(error, ModelError::NotFound);
}

#[tokio::test]
async fn downloads_a_file_and_returns_its_digest() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("downloaded");

    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(
        std::fs::metadata(&part).expect("part").len(),
        bytes.len() as u64
    );
}

#[tokio::test]
async fn resumes_from_a_part_file_and_still_hashes_the_whole_file() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(PARTIAL, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    // Pretend a previous run stopped a third of the way in.
    let stopped = bytes.len() / 3;
    std::fs::write(&part, &bytes[..stopped]).expect("part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("resumed");

    // The digest covers the bytes already on disk as well as the new ones.
    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(
        std::fs::metadata(&part).expect("part").len(),
        bytes.len() as u64
    );
}

#[tokio::test]
async fn starts_over_when_the_server_ignores_the_range() {
    let bytes = payload();
    // 200 rather than 206: the body starts at zero even though a range was asked for.
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    std::fs::write(&part, &bytes[..1000]).expect("part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("restarted");

    // Appending would have produced 1000 bytes too many and a wrong digest.
    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(
        std::fs::metadata(&part).expect("part").len(),
        bytes.len() as u64
    );
}

/// A part file bigger than the remote file makes a real server answer 416; without this the
/// download could never start again, and nothing in the TUI deletes a part file.
#[tokio::test]
async fn a_part_file_larger_than_the_remote_file_starts_over() {
    let bytes = payload();
    let base = serve(vec![
        Reply::Full(RANGE_REFUSED, Vec::new()),
        Reply::Full(OK_BYTES, bytes.clone()),
    ])
    .await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    std::fs::write(&part, vec![7u8; bytes.len() + 1000]).expect("part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("restarted");

    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(
        std::fs::metadata(&part).expect("part").len(),
        bytes.len() as u64
    );
}

#[tokio::test]
async fn reports_progress_without_flooding() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let collect = std::sync::Arc::clone(&seen);

    hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            move |done, total| {
                if let Ok(mut log) = collect.lock() {
                    log.push((done, total));
                }
            },
        )
        .await
        .expect("downloaded");

    let log = seen.lock().expect("lock").clone();
    // At least the first and last reports, and nowhere near one per chunk.
    assert!(log.len() >= 2, "{log:?}");
    assert!(
        log.len() < 50,
        "throttling should keep this small: {}",
        log.len()
    );
    assert_eq!(log.last().map(|(done, _)| *done), Some(bytes.len() as u64));
    assert_eq!(
        log.last().and_then(|(_, total)| *total),
        Some(bytes.len() as u64)
    );
}

#[tokio::test]
async fn a_cancelled_download_keeps_what_it_wrote() {
    let bytes = payload();
    let base = serve(vec![Reply::Stall(OK_BYTES, bytes.clone(), bytes.len() / 2)]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    let cancel = CancellationToken::new();
    let token = cancel.clone();

    // Cancelling from the callback makes this deterministic: the download is stopped at the
    // first report that bytes are on disk, never before and never after it finished.
    let result = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &cancel,
            move |done, _| {
                if done > 0 {
                    token.cancel();
                }
            },
        )
        .await;

    assert!(result.is_err(), "a cancelled download does not succeed");
    // The part file is the resume state: keeping an empty one would throw away every byte
    // paid for, which is the whole reason the file exists.
    let written = std::fs::metadata(&part)
        .expect("the partial file must be kept")
        .len();
    assert!(
        written > 0,
        "the bytes received must be kept, got {written}"
    );
    assert!(written < bytes.len() as u64, "and the file is not complete");
}

#[tokio::test]
async fn a_wrong_digest_fails_the_job_and_leaves_no_model() {
    let bytes = gguf_payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let database = dir.path().join("test.db");

    let log = run_pull(
        &base,
        dir.path(),
        &database,
        remote(&bytes, Some("0".repeat(64))),
    )
    .await;

    assert!(
        log.iter().any(|e| matches!(e, PullEvent::Failed { .. })),
        "{log:?}"
    );
    let (destination, part) = store_paths(dir.path());
    assert!(!part.exists(), "a corrupt partial file must be removed");
    // "leaves no model" is about both halves: the file and the row that claims it is one.
    assert!(!destination.exists(), "no file at the final place");
    assert!(inventory(&database).is_empty(), "nothing in the inventory");
}

/// Review Focus 4: a write that cannot happen must say so in French, not panic — and must
/// never leave an inventory row claiming the model is there.
#[tokio::test]
async fn a_write_failure_is_reported_and_records_nothing() {
    let bytes = gguf_payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let database = dir.path().join("test.db");
    // A directory where the part file belongs: opening it for writing must fail.
    let (destination, part) = store_paths(dir.path());
    std::fs::create_dir_all(&part).expect("part directory");

    let log = run_pull(&base, dir.path(), &database, remote(&bytes, None)).await;

    let Some(PullEvent::Failed { error, .. }) = log
        .iter()
        .find(|e| matches!(e, PullEvent::Failed { .. }))
        .cloned()
    else {
        panic!("expected a failure: {log:?}")
    };
    assert!(error.contains("écriture de"), "{error}");
    assert!(!destination.exists(), "no file at the final place");
    assert!(inventory(&database).is_empty(), "nothing in the inventory");
}

/// The write that fails in the middle of a transfer, which the part-file path cannot reach:
/// `/dev/full` accepts the open and refuses every byte.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_write_failing_mid_transfer_is_reported_in_french() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;

    let error = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            Path::new("/dev/full"),
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect_err("refused");

    assert!(matches!(error, ModelError::Io(_)), "{error:?}");
    assert!(
        error.to_string().contains("écriture de /dev/full"),
        "{error}"
    );
}

/// A file at its final place with no row is invisible to `/models`, undeletable by `/rm`,
/// and makes the next `/pull` download the whole thing again.
#[tokio::test]
async fn a_failed_inventory_write_leaves_no_file_behind() {
    let bytes = gguf_payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    // A directory where the database belongs: opening it fails, so saving the row fails
    // after the file has already been moved to its final name.
    let unusable = dir.path().join("unusable.db");
    std::fs::create_dir(&unusable).expect("database directory");

    let log = run_pull(&base, dir.path(), &unusable, remote(&bytes, None)).await;

    assert!(
        log.iter().any(|e| matches!(e, PullEvent::Failed { .. })),
        "{log:?}"
    );
    let (destination, part) = store_paths(dir.path());
    assert!(
        !destination.exists(),
        "an unrecorded file must not be left behind"
    );
    assert!(!part.exists(), "and neither must the part file");
}

/// Review Focus 3: a remote listing must not be able to name a path outside the store.
#[test]
fn the_store_path_stays_inside_the_store() {
    let root = Path::new("/tmp/store");

    assert!(chatatui::models::download::paths(root, "owner/name", "../../x.gguf").is_err());
    assert!(chatatui::models::download::paths(root, "../owner/name", "x.gguf").is_err());
    assert!(chatatui::models::download::paths(root, "noslash", "x.gguf").is_err());

    let (file, part) =
        chatatui::models::download::paths(root, "owner/name", "x.gguf").expect("paths");
    assert_eq!(file, root.join("owner").join("name").join("x.gguf"));
    assert_eq!(part, root.join("owner").join("name").join("x.gguf.part"));
}
