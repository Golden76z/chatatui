//! Watching the collections' folders while chatatui runs (`[rag] auto_index`).
//!
//! File system events are debounced (a burst of saves gives one notification) and
//! reduced to the names of the collections whose indexable files changed; the app then
//! updates them in the background. Hidden folders and build output are ignored.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};

use super::extract::FileKind;

/// Quiet time after the last change before a collection is reported.
const DEBOUNCE: Duration = Duration::from_secs(3);
/// Folders never worth re-indexing for.
const IGNORED_DIRS: &[&str] = &["target", "node_modules", "__pycache__"];

/// Keeps the watch alive; dropping it stops watching.
pub struct Watch {
    _debouncer: Debouncer<RecommendedWatcher>,
}

impl std::fmt::Debug for Watch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Watch").finish_non_exhaustive()
    }
}

/// Watches `roots` (collection name, folder); `changed` receives the names of the
/// collections with changed files. Folders that do not exist are skipped.
pub fn watch(
    roots: Vec<(String, PathBuf)>,
    changed: impl Fn(Vec<String>) + Send + 'static,
) -> Result<Watch, String> {
    let watched = roots.clone();
    let mut debouncer = new_debouncer(DEBOUNCE, move |result: DebounceEventResult| {
        let Ok(events) = result else {
            return;
        };
        let names = collections_touched(&watched, events.iter().map(|e| e.path.as_path()));
        if !names.is_empty() {
            changed(names);
        }
    })
    .map_err(|e| format!("surveillance des dossiers : {e}"))?;
    for (_, root) in roots.iter().filter(|(_, root)| root.is_dir()) {
        debouncer
            .watcher()
            .watch(root, RecursiveMode::Recursive)
            .map_err(|e| format!("surveillance de {} : {e}", root.display()))?;
    }
    Ok(Watch {
        _debouncer: debouncer,
    })
}

/// Names of the collections of `roots` containing an indexable file among `paths`.
fn collections_touched<'a>(
    roots: &[(String, PathBuf)],
    paths: impl Iterator<Item = &'a Path>,
) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for path in paths {
        for (name, root) in roots {
            let Ok(relative) = path.strip_prefix(root) else {
                continue;
            };
            if relevant(relative) && !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    names
}

/// `true` for a supported file outside hidden folders and build output.
fn relevant(relative: &Path) -> bool {
    let hidden_or_ignored = relative.components().any(|c| {
        let part = c.as_os_str().to_string_lossy();
        part.starts_with('.') || IGNORED_DIRS.contains(&part.as_ref())
    });
    !hidden_or_ignored && FileKind::of(relative).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_indexable_files_count() {
        let roots = vec![
            ("cours".to_owned(), PathBuf::from("/d/cours")),
            ("code".to_owned(), PathBuf::from("/d/code")),
        ];
        let paths = [
            "/d/cours/ch1.md",
            "/d/cours/.git/index",
            "/d/code/target/debug/build.rs",
            "/d/code/photo.jpg",
            "/elsewhere/a.md",
        ];
        let touched = collections_touched(&roots, paths.iter().map(Path::new));
        assert_eq!(touched, vec!["cours"]);
    }

    #[test]
    fn changes_are_reported_after_a_quiet_time() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().canonicalize().expect("path");
        let (tx, rx) = std::sync::mpsc::channel();
        let _watch = watch(vec![("docs".into(), root.clone())], move |names| {
            let _ = tx.send(names);
        })
        .expect("watch");
        std::thread::sleep(Duration::from_millis(200));
        std::fs::write(root.join("note.md"), "# Note").expect("write");
        std::fs::write(root.join("ignored.bin"), [0u8]).expect("write");
        let names = rx
            .recv_timeout(Duration::from_secs(15))
            .expect("change reported");
        assert_eq!(names, vec!["docs"]);
    }
}
