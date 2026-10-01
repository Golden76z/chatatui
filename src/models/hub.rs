//! The HuggingFace client: list a repository's GGUF files, and download one with resume.
//!
//! Only the public HTTP API is used — `GET /api/models/{repo}/tree/{revision}` to list and
//! `GET /{repo}/resolve/{revision}/{file}` to fetch — so nothing here depends on Ollama or
//! on a Hub SDK.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

use super::ModelError;

/// Public Hub, overridden in tests.
const DEFAULT_BASE: &str = "https://huggingface.co";
/// Bytes read at a time when re-hashing a part file on resume.
const REHASH_CHUNK: usize = 1 << 20;
/// Shortest gap between two progress reports.
const PROGRESS_EVERY: Duration = Duration::from_millis(100);

/// A GGUF file a repository offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    /// Name inside the repository; always a bare file name (see [`safe_file_name`]).
    pub path: String,
    pub bytes: u64,
    /// sha256 of the LFS object, when the API exposes one.
    pub sha256: Option<String>,
}

/// What a `/pull` argument names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Target {
    /// `owner/name`.
    pub repo: String,
    /// Branch or commit the URL names; `None` when it names none, and the caller falls back
    /// to the Hub's default branch.
    pub revision: Option<String>,
    /// File named by a `/blob/` or `/resolve/` URL.
    pub file: Option<String>,
}

/// Splits what the user typed into a repository and, when the URL names them, a revision
/// and a file.
///
/// Accepts `owner/name`, `hf.co/owner/name`, and a `huggingface.co` URL with or without a
/// `/tree/<rev>` or `/blob/<rev>/<file>` tail, with or without a query string.
pub fn parse_target(arg: &str) -> Result<Target, ModelError> {
    let trimmed = arg.trim();
    let without_scheme = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);
    let path = ["huggingface.co/", "hf.co/"]
        .iter()
        .find_map(|host| without_scheme.strip_prefix(host))
        .unwrap_or(without_scheme);
    // The Download button hands out `…/resolve/main/m.gguf?download=true`, and a copied
    // link can carry an anchor: neither belongs to the file name.
    let path = path.split_once(['?', '#']).map_or(path, |(head, _)| head);
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let [owner, name, tail @ ..] = parts.as_slice() else {
        return Err(ModelError::Http(
            "donnez un dépôt : /pull <propriétaire>/<nom> [fichier]".into(),
        ));
    };
    let (revision, file) = match tail {
        ["blob" | "resolve", revision, file] => {
            (Some((*revision).to_owned()), Some((*file).to_owned()))
        }
        ["tree", revision] => (Some((*revision).to_owned()), None),
        [] | ["tree"] => (None, None),
        _ => {
            return Err(ModelError::Http(
                "lien HuggingFace non reconnu : /pull <propriétaire>/<nom> [fichier]".into(),
            ));
        }
    };
    Ok(Target {
        repo: format!("{owner}/{name}"),
        revision,
        file,
    })
}

/// Said when a repository only offers a model split across several files.
const MULTI_PART: &str =
    "les modèles en plusieurs fichiers ne sont pas gérés (un seul .gguf à la fois)";

/// Accepts a bare file name only. The name comes from a remote API, so anything that could
/// escape the store directory is refused — and so are the multi-part GGUF files that live in
/// subdirectories, which this version cannot assemble.
pub fn safe_file_name(path: &str) -> Result<&str, ModelError> {
    if path.is_empty() || path == "." || path == ".." {
        return Err(ModelError::Http("nom de fichier invalide".into()));
    }
    if path.contains('/') || path.contains('\\') {
        return Err(ModelError::Http(MULTI_PART.into()));
    }
    Ok(path)
}

/// Recognises the `-00001-of-00002` tail of a shard.
///
/// A shard at the root of a repository passes [`safe_file_name`], so without this it would
/// download and land in the inventory as a whole model with half its weights missing.
fn is_multi_part(path: &str) -> bool {
    let stem = path.rsplit_once('.').map_or(path, |(stem, _)| stem);
    let Some((head, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, index)) = head.rsplit_once('-') else {
        return false;
    };
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    digits(index) && digits(total)
}

/// Reads the `.gguf` entries out of a `tree` answer.
fn parse_tree(body: &str) -> Result<Vec<RemoteFile>, ModelError> {
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default, rename = "type")]
        kind: String,
        path: String,
        #[serde(default)]
        size: u64,
        #[serde(default)]
        lfs: Option<Lfs>,
    }
    #[derive(Deserialize)]
    struct Lfs {
        #[serde(default)]
        oid: String,
        #[serde(default)]
        size: u64,
    }

    let entries: Vec<Entry> =
        serde_json::from_str(body).map_err(|e| ModelError::Http(e.to_string()))?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind != "directory" && e.path.to_lowercase().ends_with(".gguf"))
        .map(|e| {
            let lfs = e.lfs.filter(|l| l.size > 0);
            RemoteFile {
                bytes: lfs.as_ref().map_or(e.size, |l| l.size),
                sha256: lfs
                    .as_ref()
                    .map(|l| l.oid.trim_start_matches("sha256:").to_owned())
                    .filter(|oid| !oid.is_empty()),
                path: e.path,
            }
        })
        .collect())
}

/// HTTP client for one Hub.
#[derive(Clone)]
pub struct Hub {
    http: Client,
    base: String,
    token: Option<String>,
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the token.
        f.debug_struct("Hub")
            .field("base", &self.base)
            .field("token", &self.token.is_some())
            .finish()
    }
}

impl Hub {
    /// `base` overrides the public Hub (tests point it at a local server).
    pub fn new(
        base: Option<String>,
        token: Option<String>,
        connect_timeout: Duration,
    ) -> Result<Self, ModelError> {
        let http = Client::builder()
            .connect_timeout(connect_timeout)
            .user_agent(concat!("chatatui/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ModelError::Http(e.to_string()))?;
        Ok(Self {
            http,
            base: base
                .map(|b| b.trim_end_matches('/').to_owned())
                .unwrap_or_else(|| DEFAULT_BASE.to_owned()),
            token: token.filter(|t| !t.trim().is_empty()),
        })
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// The repository's GGUF files, largest name first as the API returns them.
    pub async fn list_gguf(
        &self,
        repo: &str,
        revision: &str,
    ) -> Result<Vec<RemoteFile>, ModelError> {
        let url = format!("{}/api/models/{repo}/tree/{revision}", self.base);
        let response = self
            .authorize(self.http.get(url))
            .send()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        match response.status() {
            StatusCode::OK => {}
            StatusCode::NOT_FOUND => return Err(ModelError::NotFound),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Err(ModelError::Http(
                    "dépôt restreint : renseignez un jeton dans [models] token_env".into(),
                ));
            }
            status => return Err(ModelError::Http(format!("HTTP {status}"))),
        }
        let body = response
            .text()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        let all = parse_tree(&body)?;
        let offered = all.len();
        let files: Vec<RemoteFile> = all
            .into_iter()
            .filter(|f| safe_file_name(&f.path).is_ok() && !is_multi_part(&f.path))
            .collect();
        // Dropping every entry silently would make the repository look empty, when the real
        // reason is that its model is split in several files.
        if files.is_empty() && offered > 0 {
            return Err(ModelError::Http(MULTI_PART.to_owned()));
        }
        Ok(files)
    }

    /// Streams `file` into `part`, resuming when `part` already holds bytes, and returns the
    /// hex sha256 of the whole file.
    ///
    /// `progress` receives `(bytes written, total when known)` at most every 100 ms.
    pub async fn fetch(
        &self,
        repo: &str,
        revision: &str,
        file: &RemoteFile,
        part: &Path,
        cancel: &CancellationToken,
        progress: impl Fn(u64, Option<u64>) + Send,
    ) -> Result<String, ModelError> {
        let name = safe_file_name(&file.path)?;
        if let Some(parent) = part.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ModelError::Io(format!("création de {}: {e}", parent.display())))?;
        }
        let already = tokio::fs::metadata(part)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        let url = format!("{}/{repo}/resolve/{revision}/{name}", self.base);
        let get = |from: u64| {
            let request = self.authorize(self.http.get(url.as_str()));
            match from {
                0 => request,
                from => request.header("Range", format!("bytes={from}-")),
            }
            .send()
        };
        let mut already = already;
        let mut response = get(already)
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        // The part file is longer than the file on the Hub (a changed revision, a truncated
        // upload): it cannot be resumed, and nothing in the TUI deletes it, so ask again
        // from zero instead of failing every single time.
        if already > 0 && response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
            already = 0;
            response = get(0).await.map_err(|e| ModelError::Http(e.to_string()))?;
        }
        let resumed = match response.status() {
            StatusCode::PARTIAL_CONTENT => true,
            // The server ignored the range: the body starts at zero, so start over rather
            // than append and produce a corrupt file.
            StatusCode::OK => false,
            StatusCode::NOT_FOUND => return Err(ModelError::NotFound),
            status => return Err(ModelError::Http(format!("HTTP {status}"))),
        };

        let mut hasher = Sha256::new();
        let mut written = 0;
        let mut handle = if resumed {
            // The digest covers the whole file, so what is already on disk has to go
            // through the hasher before the new bytes.
            written = already;
            rehash(part, &mut hasher).await?;
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(part)
                .await
                .map_err(|e| ModelError::Io(format!("ouverture de {}: {e}", part.display())))?
        } else {
            tokio::fs::File::create(part)
                .await
                .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?
        };

        let total = response
            .content_length()
            .map(|length| written + length)
            .or((file.bytes > 0).then_some(file.bytes));
        let mut stream = response.bytes_stream();
        let mut last = Instant::now();
        progress(written, total);
        loop {
            let chunk = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    // Flush what we have so the next /pull resumes from here.
                    let _ = handle.flush().await;
                    return Err(ModelError::Io("annulé".into()));
                }
                chunk = stream.next() => chunk,
            };
            let Some(chunk) = chunk else { break };
            let chunk = chunk.map_err(|e| ModelError::Http(e.to_string()))?;
            hasher.update(&chunk);
            handle
                .write_all(&chunk)
                .await
                .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?;
            written += chunk.len() as u64;
            if last.elapsed() >= PROGRESS_EVERY {
                progress(written, total);
                last = Instant::now();
            }
        }
        handle
            .flush()
            .await
            .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?;
        handle
            .sync_all()
            .await
            .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?;
        progress(written, total);
        Ok(hex(&hasher.finalize()))
    }
}

/// Feeds an existing part file through `hasher`.
async fn rehash(part: &Path, hasher: &mut Sha256) -> Result<(), ModelError> {
    let mut file = tokio::fs::File::open(part)
        .await
        .map_err(|e| ModelError::Io(format!("lecture de {}: {e}", part.display())))?;
    let mut buffer = vec![0u8; REHASH_CHUNK];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(|e| ModelError::Io(format!("lecture de {}: {e}", part.display())))?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_plain_repository() {
        let target = parse_target("bartowski/Qwen2.5-7B-Instruct-GGUF").expect("parsed");

        assert_eq!(target.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(target.revision, None);
        assert_eq!(target.file, None);
    }

    /// The spec supports pulling another revision, so a URL that names one must not be
    /// quietly downloaded from `main` and recorded as `main`.
    #[test]
    fn a_url_keeps_the_revision_it_names() {
        let tree = parse_target("https://huggingface.co/owner/name/tree/v2.0").expect("parsed");
        assert_eq!(tree.repo, "owner/name");
        assert_eq!(tree.revision.as_deref(), Some("v2.0"));
        assert_eq!(tree.file, None);

        let blob =
            parse_target("https://huggingface.co/owner/name/blob/v2.0/m.gguf").expect("parsed");
        assert_eq!(blob.repo, "owner/name");
        assert_eq!(blob.revision.as_deref(), Some("v2.0"));
        assert_eq!(blob.file.as_deref(), Some("m.gguf"));

        // Nothing named: the caller falls back to the Hub's default.
        assert_eq!(parse_target("owner/name").expect("parsed").revision, None);
    }

    /// Review Focus 1: a pasted URL is what people actually type.
    #[test]
    fn accepts_a_pasted_url() {
        for input in [
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF",
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/tree/main",
            "huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/",
            "hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF",
        ] {
            let target = parse_target(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!(target.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF", "{input}");
            assert_eq!(target.file, None, "{input}");
        }
    }

    /// Review Focus 1: a blob URL names the file, so use it.
    #[test]
    fn a_blob_url_also_names_the_file() {
        let target = parse_target(
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        )
        .expect("parsed");

        assert_eq!(target.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(
            target.file.as_deref(),
            Some("Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        );
    }

    /// Review Focus 1: the URL HuggingFace's own Download button hands out carries a query.
    #[test]
    fn a_download_url_loses_its_query_string() {
        for input in [
            "https://huggingface.co/owner/name/resolve/main/m.gguf?download=true",
            "https://huggingface.co/owner/name/blob/main/m.gguf#usage",
        ] {
            let target = parse_target(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!(target.repo, "owner/name", "{input}");
            assert_eq!(target.file.as_deref(), Some("m.gguf"), "{input}");
        }
    }

    #[test]
    fn rejects_something_that_is_not_a_repository() {
        for input in ["", "llama3.2", "a/b/c/d/e", "/", "owner/"] {
            assert!(parse_target(input).is_err(), "{input} should be refused");
        }
    }

    /// Review Focus 3: the file name comes from a remote API.
    #[test]
    fn refuses_a_file_name_that_escapes_the_store() {
        for path in [
            "../../.bashrc",
            "/etc/passwd",
            "sub/dir/model.gguf",
            "..",
            ".",
            "",
            "C:\\windows\\system32\\x.gguf",
            "dir\\model.gguf",
        ] {
            assert!(safe_file_name(path).is_err(), "{path} should be refused");
        }
    }

    /// A shard is half a model: taking it for a whole one is worse than refusing it.
    #[test]
    fn spots_the_shards_of_a_multi_part_model() {
        for path in [
            "m-00001-of-00002.gguf",
            "Qwen2.5-72B-Instruct-Q4_K_M-00003-of-00009.gguf",
        ] {
            assert!(is_multi_part(path), "{path} is a shard");
        }
        for path in ["m-Q4_K_M.gguf", "model.gguf", "m-of-mine.gguf"] {
            assert!(!is_multi_part(path), "{path} is a whole file");
        }
    }

    #[test]
    fn accepts_a_bare_file_name() {
        assert_eq!(
            safe_file_name("Qwen2.5-7B-Instruct-Q4_K_M.gguf").expect("accepted"),
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        );
    }

    #[test]
    fn keeps_only_gguf_entries_and_prefers_the_lfs_size() {
        let body = r#"[
            {"type":"directory","path":"sub"},
            {"type":"file","path":"README.md","size":1200},
            {"type":"file","path":"model.safetensors","size":99},
            {"type":"file","path":"m-Q4_K_M.gguf","size":135,
             "lfs":{"oid":"sha256:abc123","size":4431401088}},
            {"type":"file","path":"m-Q8_0.gguf","size":7200000000}
        ]"#;

        let files = parse_tree(body).expect("parsed");

        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "m-Q4_K_M.gguf");
        assert_eq!(files[0].bytes, 4_431_401_088);
        assert_eq!(files[0].sha256.as_deref(), Some("abc123"));
        assert_eq!(files[1].path, "m-Q8_0.gguf");
        assert_eq!(files[1].bytes, 7_200_000_000);
        assert_eq!(files[1].sha256, None);
    }
}
