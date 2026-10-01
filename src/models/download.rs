//! The background job behind `/pull`: fetches one GGUF file, verifies it, reads its
//! header, and records it in the inventory.
//!
//! Shaped like [`crate::rag::indexer::run`]: a `CancellationToken` stops it, progress is
//! reported through a callback the runtime turns into `AppEvent`s, and the final event says
//! whether it finished, failed or was cancelled. The partial file survives a cancellation,
//! so the next run resumes from it.

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use tokio_util::sync::CancellationToken;

use super::{
    ModelError, gguf,
    hub::{Hub, RemoteFile},
    store::{self, LocalModel},
};

/// What to download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PullRequest {
    /// `owner/name`.
    pub repo: String,
    pub revision: String,
    pub file: RemoteFile,
    /// Root of the model store.
    pub dir: PathBuf,
}

/// Progress of a download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PullEvent {
    /// `done` bytes written of `total` (unknown until the server says), at `rate` bytes/s.
    Progress {
        repo: String,
        file: String,
        done: u64,
        total: Option<u64>,
        rate: u64,
    },
    /// Downloaded, verified and inspected.
    Finished(Box<LocalModel>),
    Failed {
        repo: String,
        file: String,
        error: String,
    },
    Cancelled {
        repo: String,
        file: String,
    },
}

/// Where a model's file and its in-progress part file live.
///
/// `<dir>/<owner>/<repo>/<file>` keeps the repository's real nesting, so two repositories
/// whose names differ only in where the slash falls cannot collide.
pub fn paths(dir: &Path, repo: &str, file: &str) -> Result<(PathBuf, PathBuf), ModelError> {
    let name = super::hub::safe_file_name(file)?;
    let (owner, repository) = repo
        .split_once('/')
        .ok_or_else(|| ModelError::Http("dépôt invalide".into()))?;
    for part in [owner, repository] {
        if part.is_empty() || part.contains('/') || part.contains('\\') || part.contains("..") {
            return Err(ModelError::Http("dépôt invalide".into()));
        }
    }
    let folder = dir.join(owner).join(repository);
    Ok((folder.join(name), folder.join(format!("{name}.part"))))
}

/// Runs the download, reporting progress and the outcome through `report`.
pub async fn run(
    request: PullRequest,
    db_path: PathBuf,
    hub: std::sync::Arc<Hub>,
    cancel: CancellationToken,
    report: impl Fn(PullEvent) + Send + Sync,
) {
    let repo = request.repo.clone();
    let file = request.file.path.clone();
    let outcome = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(None),
        result = pull(&request, &db_path, hub.as_ref(), &cancel, &report) => result.map_err(Some),
    };
    report(match outcome {
        Ok(model) => PullEvent::Finished(Box::new(model)),
        Err(None) => PullEvent::Cancelled { repo, file },
        Err(Some(error)) => PullEvent::Failed {
            repo,
            file,
            error: error.to_string(),
        },
    });
}

async fn pull(
    request: &PullRequest,
    db_path: &Path,
    hub: &Hub,
    cancel: &CancellationToken,
    report: &(impl Fn(PullEvent) + Send + Sync),
) -> Result<LocalModel, ModelError> {
    let (destination, part) = paths(&request.dir, &request.repo, &request.file.path)?;
    let started = std::time::Instant::now();
    let digest = hub
        .fetch(
            &request.repo,
            &request.revision,
            &request.file,
            &part,
            cancel,
            |done, total| {
                let seconds = started.elapsed().as_secs_f64();
                let rate = if seconds > 0.0 {
                    (done as f64 / seconds) as u64
                } else {
                    0
                };
                report(PullEvent::Progress {
                    repo: request.repo.clone(),
                    file: request.file.path.clone(),
                    done,
                    total,
                    rate,
                });
            },
        )
        .await?;

    // A wrong file is worse than no file.
    if let Some(expected) = &request.file.sha256
        && !expected.eq_ignore_ascii_case(&digest)
    {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(ModelError::Checksum);
    }
    tokio::fs::rename(&part, &destination)
        .await
        .map_err(|e| ModelError::Io(format!("déplacement vers {}: {e}", destination.display())))?;

    let bytes = tokio::fs::metadata(&destination)
        .await
        .map(|m| m.len())
        .unwrap_or(request.file.bytes);
    let inspected = destination.clone();
    let meta = tokio::task::spawn_blocking(move || gguf::read(&inspected))
        .await
        .map_err(|e| ModelError::Io(e.to_string()))??;

    let model = LocalModel {
        repo: request.repo.clone(),
        revision: request.revision.clone(),
        file: request.file.path.clone(),
        path: destination.to_string_lossy().into_owned(),
        bytes,
        // `None` means "not verified", never "verified and fine".
        sha256: request.file.sha256.as_ref().map(|_| digest),
        architecture: meta.architecture,
        quantization: meta.quantization,
        context_length: meta.context_length,
        parameters: meta.parameters,
        downloaded_at: now(),
    };
    let row = model.clone();
    let database = db_path.to_path_buf();
    let saved = tokio::task::spawn_blocking(move || -> Result<(), ModelError> {
        let store =
            crate::storage::Store::open(&database).map_err(|e| ModelError::Io(e.to_string()))?;
        store::save(&store.into_connection(), &row).map_err(|e| ModelError::Io(e.to_string()))
    })
    .await
    .map_err(|e| ModelError::Io(e.to_string()));
    if let Err(error) = saved.and_then(|inner| inner) {
        // The file is already at its final name but nothing records it: `/models` would not
        // show it, `/rm` deletes through the row so it could not remove it, and the next
        // `/pull` would find no part file to resume from and download it all again.
        let _ = tokio::fs::remove_file(&destination).await;
        return Err(error);
    }
    Ok(model)
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}
