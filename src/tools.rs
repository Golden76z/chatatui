//! Tools the model may call (`/tools on`): read a file, list a folder, search the
//! document collections. All are read-only, and each call is confirmed by the user
//! (see `App`); files that usually hold secrets are refused whatever the answer.

use std::path::Path;

use serde::Deserialize;

use crate::{
    context::{ContextProvider, ContextQuery},
    files,
    llm::{ToolCall, ToolSpec},
    state::{Conversation, MessageStatus, Role},
};

pub const READ_FILE: &str = "read_file";
pub const LIST_DIR: &str = "list_dir";
pub const SEARCH_DOCUMENTS: &str = "search_documents";
pub const FETCH_URL: &str = "fetch_url";

/// Characters of a file given to the model at most.
const MAX_FILE_CHARS: usize = 60_000;
/// Entries of a folder listed at most.
const MAX_ENTRIES: usize = 300;

/// The tools offered to the model.
pub fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: READ_FILE.into(),
            description: "Read a UTF-8 text file on the user's computer (code, notes, \
                          Markdown, config…). Paths may start with ~.".into(),
            parameters: r#"{"type":"object","properties":{"path":{"type":"string","description":"File path"}},"required":["path"]}"#.into(),
        },
        ToolSpec {
            name: LIST_DIR.into(),
            description: "List the files and folders of a directory on the user's computer \
                          (folders end with /). Paths may start with ~.".into(),
            parameters: r#"{"type":"object","properties":{"path":{"type":"string","description":"Directory path"}},"required":["path"]}"#.into(),
        },
        ToolSpec {
            name: FETCH_URL.into(),
            description: "Read a web page (http or https) and return its text.".into(),
            parameters: r#"{"type":"object","properties":{"url":{"type":"string","description":"Page address"}},"required":["url"]}"#.into(),
        },
        ToolSpec {
            name: SEARCH_DOCUMENTS.into(),
            description: "Search the user's indexed documents (courses, notes, PDFs) and \
                          return the most relevant passages with their source.".into(),
            parameters: r#"{"type":"object","properties":{"query":{"type":"string","description":"What to look for"},"collection":{"type":"string","description":"Collection name (default: the conversation's)"}},"required":["query"]}"#.into(),
        },
    ]
}

/// What a tool produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutput {
    /// `false` for errors and refusals.
    pub ok: bool,
    /// Sent to the model.
    pub text: String,
}

impl ToolOutput {
    fn error(text: impl Into<String>) -> Self {
        Self {
            ok: false,
            text: text.into(),
        }
    }
}

#[derive(Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
struct UrlArgs {
    url: String,
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    collection: Option<String>,
}

/// What a call does, for the user (`lire ~/notes.md`).
pub fn describe(call: &ToolCall) -> String {
    match call.name.as_str() {
        READ_FILE => serde_json::from_str::<PathArgs>(&call.arguments)
            .map_or_else(|_| "lire un fichier".into(), |a| format!("lire {}", a.path)),
        LIST_DIR => serde_json::from_str::<PathArgs>(&call.arguments).map_or_else(
            |_| "lister un dossier".into(),
            |a| format!("lister {}", a.path),
        ),
        FETCH_URL => serde_json::from_str::<UrlArgs>(&call.arguments).map_or_else(
            |_| "ouvrir une page web".into(),
            |a| format!("ouvrir {}", a.url),
        ),
        SEARCH_DOCUMENTS => serde_json::from_str::<SearchArgs>(&call.arguments).map_or_else(
            |_| "chercher dans les documents".into(),
            |a| match a.collection {
                Some(c) => format!("chercher « {} » dans « {c} »", a.query),
                None => format!("chercher « {} » dans les documents", a.query),
            },
        ),
        other => match other.split_once(crate::mcp::SEPARATOR) {
            Some((server, tool)) => describe_mcp(server, tool, &call.arguments),
            None => format!("outil inconnu : {other}"),
        },
    }
}

/// `git › git_log (repo_path: ~/projet, max_count: 5)`: an MCP call, its arguments shortened.
fn describe_mcp(server: &str, tool: &str, arguments: &str) -> String {
    const MAX_ARGS: usize = 120;
    let args = match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(serde_json::Value::Object(map)) => map
            .iter()
            .map(|(key, value)| match value {
                serde_json::Value::String(text) => format!("{key}: {text}"),
                other => format!("{key}: {other}"),
            })
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    };
    let mut args: String = args.chars().take(MAX_ARGS).collect();
    if args.chars().count() == MAX_ARGS {
        args.push('…');
    }
    if args.is_empty() {
        format!("utiliser {server} › {tool}")
    } else {
        format!("utiliser {server} › {tool} ({args})")
    }
}

/// `true` for files and folders that usually hold secrets (keys, tokens, passwords).
pub fn is_sensitive(path: &Path) -> bool {
    const DIRS: &[&str] = &[
        ".ssh",
        ".gnupg",
        ".aws",
        ".kube",
        ".password-store",
        ".docker",
    ];
    path.components().any(|c| {
        let part = c.as_os_str().to_string_lossy().to_lowercase();
        DIRS.contains(&part.as_str())
            || part.starts_with(".env")
            || part.starts_with("id_rsa")
            || part.starts_with("id_ed25519")
            || part.starts_with("id_ecdsa")
            || part.contains("credentials")
            || part.contains("secret")
            || [".pem", ".key", ".p12", ".pfx", ".kdbx"]
                .iter()
                .any(|ext| part.ends_with(ext))
    })
}

/// Runs `call`. `collections` is the conversation's `/rag` value, the default of
/// `search_documents`.
pub async fn run(
    call: &ToolCall,
    context: &dyn ContextProvider,
    collections: Option<&str>,
) -> ToolOutput {
    match call.name.as_str() {
        READ_FILE => match serde_json::from_str::<PathArgs>(&call.arguments) {
            Ok(args) => read_file(args.path).await,
            Err(e) => ToolOutput::error(format!("arguments invalides : {e}")),
        },
        LIST_DIR => match serde_json::from_str::<PathArgs>(&call.arguments) {
            Ok(args) => list_dir(args.path).await,
            Err(e) => ToolOutput::error(format!("arguments invalides : {e}")),
        },
        FETCH_URL => match serde_json::from_str::<UrlArgs>(&call.arguments) {
            Ok(args) => match crate::web::fetch(&args.url).await {
                Ok(page) => ToolOutput {
                    ok: true,
                    text: match page.title {
                        Some(title) => format!("# {title}\n\n{}", page.text),
                        None => page.text,
                    },
                },
                Err(error) => ToolOutput::error(error),
            },
            Err(e) => ToolOutput::error(format!("arguments invalides : {e}")),
        },
        SEARCH_DOCUMENTS => match serde_json::from_str::<SearchArgs>(&call.arguments) {
            Ok(args) => {
                let collection = args.collection.as_deref().or(collections);
                search(context, &args.query, collection).await
            }
            Err(e) => ToolOutput::error(format!("arguments invalides : {e}")),
        },
        other => ToolOutput::error(format!("outil inconnu : {other}")),
    }
}

async fn read_file(path: String) -> ToolOutput {
    if is_sensitive(&files::expand_home(&path)) {
        return ToolOutput::error(format!("{path} : fichier sensible, lecture refusée"));
    }
    let read = tokio::task::spawn_blocking(move || files::read_attachment(&path)).await;
    match read {
        Ok(Ok(attachment)) => {
            let mut text: String = attachment.content.chars().take(MAX_FILE_CHARS).collect();
            if text.len() < attachment.content.len() {
                text.push_str("\n[… fichier tronqué]");
            }
            ToolOutput { ok: true, text }
        }
        Ok(Err(error)) => ToolOutput::error(error),
        Err(e) => ToolOutput::error(format!("lecture interrompue ({e})")),
    }
}

async fn list_dir(path: String) -> ToolOutput {
    let dir = files::expand_home(&path);
    if is_sensitive(&dir) {
        return ToolOutput::error(format!("{path} : dossier sensible, refusé"));
    }
    let listed = tokio::task::spawn_blocking(move || -> Result<Vec<String>, String> {
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{} : {e}", dir.display()))?
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    return None;
                }
                let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
                Some(if is_dir { format!("{name}/") } else { name })
            })
            .collect();
        names.sort();
        Ok(names)
    })
    .await;
    match listed {
        Ok(Ok(names)) if names.is_empty() => ToolOutput {
            ok: true,
            text: "(dossier vide)".into(),
        },
        Ok(Ok(names)) => {
            let total = names.len();
            let mut text = names
                .into_iter()
                .take(MAX_ENTRIES)
                .collect::<Vec<_>>()
                .join("\n");
            if total > MAX_ENTRIES {
                text.push_str(&format!("\n[… {} autres]", total - MAX_ENTRIES));
            }
            ToolOutput { ok: true, text }
        }
        Ok(Err(error)) => ToolOutput::error(error),
        Err(e) => ToolOutput::error(format!("lecture interrompue ({e})")),
    }
}

async fn search(
    context: &dyn ContextProvider,
    query: &str,
    collection: Option<&str>,
) -> ToolOutput {
    let Some(collection) = collection else {
        return ToolOutput::error(
            "aucune collection choisie : l'utilisateur peut en choisir une avec /rag",
        );
    };
    let mut history = Conversation::new();
    history.push(Role::User, query, MessageStatus::Complete);
    let found = context
        .provide(ContextQuery {
            collection: Some(collection),
            history: history.messages(),
        })
        .await;
    match found {
        Ok(context) if context.is_empty() => ToolOutput {
            ok: true,
            text: "aucun passage pertinent".into(),
        },
        Ok(context) => ToolOutput {
            ok: true,
            text: context
                .chunks
                .iter()
                .enumerate()
                .map(|(i, c)| format!("[{}] {}\n{}", i + 1, c.label(), c.text.trim()))
                .collect::<Vec<_>>()
                .join("\n\n"),
        },
        Err(error) => ToolOutput::error(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::NoContext;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    #[test]
    fn specs_are_valid_json_schemas() {
        for spec in specs() {
            let schema: serde_json::Value =
                serde_json::from_str(&spec.parameters).expect("valid JSON");
            assert_eq!(schema["type"], "object", "{}", spec.name);
        }
    }

    #[test]
    fn calls_are_described_for_the_user() {
        assert_eq!(
            describe(&call(READ_FILE, r#"{"path":"~/a.md"}"#)),
            "lire ~/a.md"
        );
        assert_eq!(
            describe(&call(
                SEARCH_DOCUMENTS,
                r#"{"query":"traits","collection":"cours"}"#
            )),
            "chercher « traits » dans « cours »"
        );
        assert_eq!(describe(&call("rm", "{}")), "outil inconnu : rm");
        assert_eq!(
            describe(&call(FETCH_URL, r#"{"url":"https://doc.rust-lang.org"}"#)),
            "ouvrir https://doc.rust-lang.org"
        );
    }

    #[test]
    fn secrets_are_recognized() {
        for path in [
            "~/.ssh/id_ed25519",
            "/p/.env.local",
            "/p/server.key",
            "/x/.aws/config",
            "/x/my-credentials.json",
        ] {
            assert!(is_sensitive(Path::new(path)), "{path}");
        }
        for path in ["/home/d/cours/ch1.md", "/p/src/keymap.rs"] {
            assert!(!is_sensitive(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn mcp_calls_are_described_with_their_server() {
        let call = |name: &str, arguments: &str| ToolCall {
            id: "c".into(),
            name: name.into(),
            arguments: arguments.into(),
        };
        assert_eq!(
            describe(&call(
                "git__git_log",
                r#"{"repo_path":"~/projet","max_count":5}"#
            )),
            "utiliser git › git_log (max_count: 5, repo_path: ~/projet)"
        );
        assert_eq!(describe(&call("x__ping", "{}")), "utiliser x › ping");
        assert!(
            describe(&call("x__y", &format!(r#"{{"t":"{}"}}"#, "a".repeat(300)))).ends_with("…)")
        );
        assert_eq!(describe(&call("inconnu", "{}")), "outil inconnu : inconnu");
    }

    #[tokio::test]
    async fn read_and_list_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("notes.md"), "# Notes").expect("write");
        std::fs::create_dir(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join(".env"), "TOKEN=x").expect("write");
        let root = dir.path().display().to_string();

        let listed = run(
            &call(LIST_DIR, &serde_json::json!({ "path": root }).to_string()),
            &NoContext,
            None,
        )
        .await;
        assert_eq!(
            listed,
            ToolOutput {
                ok: true,
                text: "notes.md\nsrc/".into()
            }
        );
        let read = run(
            &call(
                READ_FILE,
                &serde_json::json!({ "path": dir.path().join("notes.md") }).to_string(),
            ),
            &NoContext,
            None,
        )
        .await;
        assert_eq!(read.text, "# Notes");
        let secret = run(
            &call(
                READ_FILE,
                &serde_json::json!({ "path": dir.path().join(".env") }).to_string(),
            ),
            &NoContext,
            None,
        )
        .await;
        assert!(!secret.ok && secret.text.contains("sensible"));
        let bad = run(&call(READ_FILE, "{"), &NoContext, None).await;
        assert!(!bad.ok);
        let search = run(
            &call(SEARCH_DOCUMENTS, r#"{"query":"x"}"#),
            &NoContext,
            None,
        )
        .await;
        assert!(!search.ok && search.text.contains("/rag"));
    }
}
