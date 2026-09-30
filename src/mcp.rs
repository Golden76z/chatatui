//! MCP (Model Context Protocol) client: servers configured in `[mcp.<name>]` are started
//! at launch (stdio transport: one JSON-RPC message per line), and their tools are offered
//! to the model next to the built-in ones. Each call is confirmed by the user like the
//! others (see `App`).
//!
//! Only what a chat client needs is implemented: `initialize`, `tools/list`,
//! `tools/call`; the server's `ping` and `roots/list` requests are answered, its
//! notifications ignored.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    process::Stdio,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    sync::{mpsc, oneshot},
};

use crate::{
    config::McpServerConfig,
    llm::{ToolCall, ToolSpec},
    tools::ToolOutput,
};

/// Protocol revision asked for (servers answer with the one they speak).
const PROTOCOL_VERSION: &str = "2025-06-18";
/// Time allowed to start a server (`npx -y …` may download it first).
const START_TIMEOUT: Duration = Duration::from_secs(90);
/// Time allowed for one tool call.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// Characters of a tool result given to the model at most.
const MAX_OUTPUT_CHARS: usize = 60_000;
/// Lines of the server's error output kept, to explain a failure.
const STDERR_LINES: usize = 20;
/// Separator between the server and tool names in the name shown to the model.
pub const SEPARATOR: &str = "__";

/// A tool of an MCP server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpTool {
    /// Name given by the server.
    pub name: String,
    /// Name shown to the model: `<server>__<tool>`, limited to `[A-Za-z0-9_-]`.
    pub exposed: String,
    pub description: String,
    /// JSON Schema of the arguments.
    pub schema: String,
}

/// State of a server, for `/mcp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerState {
    Starting,
    Ready {
        /// Name and version reported by the server.
        info: String,
        tools: Vec<McpTool>,
    },
    Failed(String),
}

/// A state change, reported to the app.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpStatus {
    pub server: String,
    pub state: ServerState,
}

struct Entry {
    state: ServerState,
    connection: Option<Arc<Connection>>,
}

/// The MCP servers and their tools, shared by the runtime and the reply tasks.
#[derive(Default)]
pub struct Registry {
    servers: RwLock<BTreeMap<String, Entry>>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry").finish_non_exhaustive()
    }
}

impl Registry {
    /// Tools of the servers that are ready.
    pub fn specs(&self) -> Vec<ToolSpec> {
        let Ok(servers) = self.servers.read() else {
            return Vec::new();
        };
        servers
            .values()
            .filter_map(|entry| match &entry.state {
                ServerState::Ready { tools, .. } => Some(tools),
                _ => None,
            })
            .flatten()
            .map(|tool| ToolSpec {
                name: tool.exposed.clone(),
                description: tool.description.clone(),
                parameters: tool.schema.clone(),
            })
            .collect()
    }

    /// Runs `call` if it names a tool of a ready server; `None` otherwise.
    pub async fn call(&self, call: &ToolCall) -> Option<ToolOutput> {
        let (connection, tool) = {
            let servers = self.servers.read().ok()?;
            servers.values().find_map(|entry| match &entry.state {
                ServerState::Ready { tools, .. } => {
                    let tool = tools.iter().find(|t| t.exposed == call.name)?;
                    Some((entry.connection.clone()?, tool.name.clone()))
                }
                _ => None,
            })?
        };
        let arguments: Value = match serde_json::from_str(&call.arguments) {
            Ok(Value::Null) => json!({}),
            Ok(arguments) => arguments,
            Err(e) => {
                return Some(ToolOutput {
                    ok: false,
                    text: format!("arguments invalides : {e}"),
                });
            }
        };
        let result = connection
            .request(
                "tools/call",
                json!({ "name": tool, "arguments": arguments }),
                CALL_TIMEOUT,
            )
            .await;
        Some(match result {
            Ok(result) => tool_output(&result),
            Err(error) => ToolOutput {
                ok: false,
                text: error,
            },
        })
    }

    fn set(&self, server: &str, state: ServerState, connection: Option<Arc<Connection>>) {
        if let Ok(mut servers) = self.servers.write() {
            servers.insert(server.to_owned(), Entry { state, connection });
        }
    }
}

/// Starts server `name` and records its tools in `registry`, reporting each state change.
pub async fn start(
    registry: Arc<Registry>,
    name: String,
    config: McpServerConfig,
    report: impl Fn(McpStatus),
) {
    registry.set(&name, ServerState::Starting, None);
    report(McpStatus {
        server: name.clone(),
        state: ServerState::Starting,
    });
    let state = match spawn(&config) {
        Ok(connection) => match handshake(&name, &connection).await {
            Ok((info, tools)) => {
                let state = ServerState::Ready { info, tools };
                registry.set(&name, state.clone(), Some(connection));
                state
            }
            Err(error) => {
                let state = ServerState::Failed(connection.explain(error));
                registry.set(&name, state.clone(), None);
                state
            }
        },
        Err(error) => {
            let state = ServerState::Failed(error);
            registry.set(&name, state.clone(), None);
            state
        }
    };
    report(McpStatus {
        server: name,
        state,
    });
}

/// `initialize`, then every page of `tools/list`.
async fn handshake(
    server: &str,
    connection: &Connection,
) -> Result<(String, Vec<McpTool>), String> {
    let init = connection
        .request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "chatatui", "version": env!("CARGO_PKG_VERSION") },
            }),
            START_TIMEOUT,
        )
        .await?;
    connection.notify("notifications/initialized", json!({}));
    let info = match (
        init["serverInfo"]["name"].as_str(),
        init["serverInfo"]["version"].as_str(),
    ) {
        (Some(name), Some(version)) => format!("{name} {version}"),
        (Some(name), None) => name.to_owned(),
        _ => String::new(),
    };
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let params = match &cursor {
            Some(cursor) => json!({ "cursor": cursor }),
            None => json!({}),
        };
        let page = connection
            .request("tools/list", params, START_TIMEOUT)
            .await?;
        for tool in page["tools"].as_array().into_iter().flatten() {
            let Some(name) = tool["name"].as_str() else {
                continue;
            };
            let schema = match &tool["inputSchema"] {
                Value::Object(_) => tool["inputSchema"].to_string(),
                _ => r#"{"type":"object"}"#.to_owned(),
            };
            tools.push(McpTool {
                name: name.to_owned(),
                exposed: exposed_name(server, name),
                description: tool["description"].as_str().unwrap_or_default().to_owned(),
                schema,
            });
        }
        cursor = page["nextCursor"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    Ok((info, tools))
}

/// `<server>__<tool>` with characters the model APIs accept, at most 64.
pub fn exposed_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    format!("{}{SEPARATOR}{}", clean(server), clean(tool))
        .chars()
        .take(64)
        .collect()
}

/// Text given to the model for a `tools/call` result.
fn tool_output(result: &Value) -> ToolOutput {
    let mut parts: Vec<String> = Vec::new();
    for item in result["content"].as_array().into_iter().flatten() {
        let part = match item["type"].as_str() {
            Some("text") => item["text"].as_str().unwrap_or_default().to_owned(),
            Some("image") => format!("[image {}]", item["mimeType"].as_str().unwrap_or_default()),
            Some("audio") => "[audio]".to_owned(),
            Some("resource") => match item["resource"]["text"].as_str() {
                Some(text) => text.to_owned(),
                None => format!(
                    "[ressource {}]",
                    item["resource"]["uri"].as_str().unwrap_or_default()
                ),
            },
            Some("resource_link") => {
                format!("[lien {}]", item["uri"].as_str().unwrap_or_default())
            }
            _ => continue,
        };
        parts.push(part);
    }
    if parts.is_empty() && !result["structuredContent"].is_null() {
        parts.push(result["structuredContent"].to_string());
    }
    let mut text = parts.join("\n");
    if text.chars().count() > MAX_OUTPUT_CHARS {
        text = text.chars().take(MAX_OUTPUT_CHARS).collect();
        text.push_str("\n[… résultat tronqué]");
    }
    ToolOutput {
        ok: !result["isError"].as_bool().unwrap_or(false),
        text,
    }
}

/// Starts the server process and connects to its standard input and output.
fn spawn(config: &McpServerConfig) -> Result<Arc<Connection>, String> {
    let command = config.command.trim();
    if command.is_empty() {
        return Err("command manquant dans la configuration".into());
    }
    let args: Vec<String> = config.args.iter().map(|a| expand_home(a)).collect();
    // `npx`, `uvx`… are scripts on Windows, which only `cmd` runs.
    let mut process = if cfg!(windows) {
        let mut process = tokio::process::Command::new("cmd");
        process.arg("/C").arg(command);
        process
    } else {
        tokio::process::Command::new(expand_home(command))
    };
    process
        .args(&args)
        .envs(&config.env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(cwd) = &config.cwd {
        process.current_dir(expand_home(cwd));
    }
    let mut child = process.spawn().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("« {command} » introuvable (installé ?)"),
        _ => format!("« {command} » : {e}"),
    })?;
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        return Err("entrées-sorties du serveur indisponibles".into());
    };
    let connection = Connection::over(stdout, stdin);
    if let Some(stderr) = child.stderr.take() {
        let tail = connection.stderr.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(mut tail) = tail.lock() {
                    if tail.len() == STDERR_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
            }
        });
    }
    if let Ok(mut slot) = connection.child.lock() {
        *slot = Some(child);
    }
    Ok(connection)
}

fn expand_home(text: &str) -> String {
    if text == "~" || text.starts_with("~/") {
        crate::files::expand_home(text).display().to_string()
    } else {
        text.to_owned()
    }
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// A JSON-RPC connection to a server.
pub struct Connection {
    outgoing: mpsc::UnboundedSender<String>,
    pending: Pending,
    next_id: AtomicU64,
    /// The server process, killed when the connection is dropped.
    child: Mutex<Option<tokio::process::Child>>,
    /// Last lines of the server's error output.
    stderr: Arc<Mutex<VecDeque<String>>>,
}

impl Connection {
    /// A connection reading messages from `reader` and writing them to `writer`.
    pub fn over(
        reader: impl AsyncRead + Unpin + Send + 'static,
        writer: impl AsyncWrite + Unpin + Send + 'static,
    ) -> Arc<Self> {
        let (outgoing, mut queue) = mpsc::unbounded_channel::<String>();
        let pending: Pending = Arc::default();
        // Writer: one message per line.
        tokio::spawn(async move {
            let mut writer = writer;
            while let Some(message) = queue.recv().await {
                let line = format!("{message}\n");
                if writer.write_all(line.as_bytes()).await.is_err() || writer.flush().await.is_err()
                {
                    break;
                }
            }
        });
        // Reader: responses complete their request; server requests are answered.
        let replies = outgoing.clone();
        let waiting = pending.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(message) = serde_json::from_str::<Value>(&line) else {
                    continue; // not JSON-RPC (a stray log line)
                };
                handle_message(&message, &waiting, &replies);
            }
            // The server stopped: nothing will answer the requests left.
            if let Ok(mut waiting) = waiting.lock() {
                for (_, sender) in waiting.drain() {
                    let _ = sender.send(Err("le serveur s'est arrêté".into()));
                }
            }
        });
        Arc::new(Self {
            outgoing,
            pending,
            next_id: AtomicU64::new(1),
            child: Mutex::new(None),
            stderr: Arc::default(),
        })
    }

    /// Sends a request and waits for its result (errors are user-facing).
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        if let Ok(mut pending) = self.pending.lock() {
            pending.insert(id, sender);
        }
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if self.outgoing.send(message.to_string()).is_err() {
            return Err("le serveur s'est arrêté".into());
        }
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("le serveur s'est arrêté".into()),
            Err(_) => {
                if let Ok(mut pending) = self.pending.lock() {
                    pending.remove(&id);
                }
                Err(format!(
                    "pas de réponse à {method} en {} s",
                    timeout.as_secs()
                ))
            }
        }
    }

    /// Sends a notification (no answer expected).
    pub fn notify(&self, method: &str, params: Value) {
        let message = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let _ = self.outgoing.send(message.to_string());
    }

    /// `error`, followed by the last line the server wrote on its error output.
    fn explain(&self, error: String) -> String {
        let last = self
            .stderr
            .lock()
            .ok()
            .and_then(|tail| tail.iter().rev().find(|l| !l.trim().is_empty()).cloned());
        match last {
            Some(line) => format!("{error} ({})", line.trim()),
            None => error,
        }
    }
}

fn handle_message(message: &Value, pending: &Pending, replies: &mpsc::UnboundedSender<String>) {
    let id = &message["id"];
    match message["method"].as_str() {
        // A response to one of our requests.
        None => {
            let Some(id) = id.as_u64() else {
                return;
            };
            let Some(sender) = pending.lock().ok().and_then(|mut p| p.remove(&id)) else {
                return;
            };
            let result = match message.get("error") {
                Some(error) => Err(error["message"]
                    .as_str()
                    .unwrap_or("erreur du serveur")
                    .to_owned()),
                None => Ok(message["result"].clone()),
            };
            let _ = sender.send(result);
        }
        // A notification: nothing to do.
        Some(_) if id.is_null() => {}
        // A request from the server.
        Some(method) => {
            let reply = match method {
                "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
                "roots/list" => json!({ "jsonrpc": "2.0", "id": id, "result": { "roots": [] } }),
                _ => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("{method} non pris en charge") },
                }),
            };
            let _ = replies.send(reply.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake server on an in-memory pipe: answers `initialize`, lists two tools on two
    /// pages, and runs `echo` / `fail`.
    fn fake_server() -> Arc<Connection> {
        let (client, server) = tokio::io::duplex(64 * 1024);
        let (server_read, mut server_write) = tokio::io::split(server);
        tokio::spawn(async move {
            let mut lines = BufReader::new(server_read).lines();
            // The server pings the client first; the answer must come back.
            server_write
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":\"p1\",\"method\":\"ping\"}\nnot json\n")
                .await
                .expect("write");
            while let Ok(Some(line)) = lines.next_line().await {
                let message: Value = serde_json::from_str(&line).expect("json");
                if message["id"] == "p1" {
                    assert_eq!(message["result"], json!({}));
                    continue;
                }
                let id = message["id"].clone();
                let result = match message["method"].as_str().unwrap_or_default() {
                    "initialize" => {
                        assert_eq!(message["params"]["clientInfo"]["name"], "chatatui");
                        json!({ "protocolVersion": PROTOCOL_VERSION, "capabilities": { "tools": {} },
                                "serverInfo": { "name": "fake", "version": "1.0" } })
                    }
                    "notifications/initialized" => continue,
                    "tools/list" if message["params"]["cursor"].is_null() => json!({
                        "tools": [{ "name": "echo", "description": "Repeats",
                                    "inputSchema": { "type": "object", "properties": { "text": { "type": "string" } } } }],
                        "nextCursor": "2",
                    }),
                    "tools/list" => json!({ "tools": [{ "name": "fail.now" }] }),
                    "tools/call" => match message["params"]["name"].as_str() {
                        Some("echo") => json!({ "content": [
                            { "type": "text", "text": message["params"]["arguments"]["text"] },
                            { "type": "image", "mimeType": "image/png", "data": "…" },
                        ] }),
                        _ => {
                            json!({ "content": [{ "type": "text", "text": "boom" }], "isError": true })
                        }
                    },
                    _ => {
                        let reply = json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "inconnu" } });
                        server_write
                            .write_all(format!("{reply}\n").as_bytes())
                            .await
                            .expect("write");
                        continue;
                    }
                };
                let reply = json!({ "jsonrpc": "2.0", "id": id, "result": result });
                server_write
                    .write_all(format!("{reply}\n").as_bytes())
                    .await
                    .expect("write");
            }
        });
        let (client_read, client_write) = tokio::io::split(client);
        Connection::over(client_read, client_write)
    }

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    #[tokio::test]
    async fn tools_are_listed_and_called() {
        let connection = fake_server();
        let (info, tools) = handshake("mes fichiers", &connection)
            .await
            .expect("handshake");
        assert_eq!(info, "fake 1.0");
        let names: Vec<&str> = tools.iter().map(|t| t.exposed.as_str()).collect();
        assert_eq!(names, vec!["mes_fichiers__echo", "mes_fichiers__fail_now"]);
        assert_eq!(tools[1].schema, r#"{"type":"object"}"#, "default schema");

        let registry = Registry::default();
        registry.set(
            "mes fichiers",
            ServerState::Ready {
                info,
                tools: tools.clone(),
            },
            Some(connection.clone()),
        );
        let specs = registry.specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].description, "Repeats");

        let output = registry
            .call(&call("mes_fichiers__echo", r#"{"text":"salut"}"#))
            .await
            .expect("an MCP tool");
        assert_eq!(
            output,
            ToolOutput {
                ok: true,
                text: "salut\n[image image/png]".into()
            }
        );
        let failed = registry
            .call(&call("mes_fichiers__fail_now", ""))
            .await
            .expect("an MCP tool");
        assert!(!failed.ok);
        assert!(registry.call(&call("read_file", "{}")).await.is_none());
        let bad = registry
            .call(&call("mes_fichiers__echo", "{"))
            .await
            .expect("an MCP tool");
        assert!(!bad.ok && bad.text.contains("arguments invalides"));

        let error = connection
            .request("resources/list", json!({}), Duration::from_secs(5))
            .await;
        assert_eq!(error, Err("inconnu".into()));
    }

    #[tokio::test]
    async fn a_stopped_server_fails_its_requests() {
        let (client, server) = tokio::io::duplex(1024);
        drop(server);
        let (read, write) = tokio::io::split(client);
        let connection = Connection::over(read, write);
        let result = connection
            .request("initialize", json!({}), Duration::from_secs(5))
            .await;
        assert_eq!(result, Err("le serveur s'est arrêté".into()));
    }

    #[tokio::test]
    async fn a_missing_program_is_explained() {
        let registry = Arc::new(Registry::default());
        let reports = Arc::new(Mutex::new(Vec::new()));
        let seen = reports.clone();
        start(
            registry.clone(),
            "x".into(),
            McpServerConfig {
                command: "definitely-not-an-mcp-server".into(),
                enabled: true,
                ..McpServerConfig::default()
            },
            move |status| seen.lock().expect("lock").push(status.state),
        )
        .await;
        let reports = reports.lock().expect("lock").clone();
        assert_eq!(reports[0], ServerState::Starting);
        // Windows runs it through `cmd`, which starts but then fails.
        assert!(
            matches!(&reports[1], ServerState::Failed(m) if !m.is_empty()),
            "{reports:?}"
        );
        assert!(registry.specs().is_empty());
    }

    #[test]
    fn exposed_names_fit_the_apis() {
        assert_eq!(exposed_name("git", "git_status"), "git__git_status");
        assert_eq!(
            exposed_name("mes docs", "lire.fichier"),
            "mes_docs__lire_fichier"
        );
        assert_eq!(exposed_name(&"s".repeat(40), &"t".repeat(40)).len(), 64);
    }

    #[test]
    fn results_become_text() {
        let structured = tool_output(&json!({ "structuredContent": { "n": 1 } }));
        assert_eq!(structured.text, r#"{"n":1}"#);
        let resource = tool_output(&json!({ "content": [
            { "type": "resource", "resource": { "uri": "file:///a", "text": "contenu" } },
            { "type": "resource_link", "uri": "file:///b" },
        ] }));
        assert_eq!(resource.text, "contenu\n[lien file:///b]");
        let long = "x".repeat(MAX_OUTPUT_CHARS + 10);
        let cut = tool_output(&json!({ "content": [{ "type": "text", "text": long }] }));
        assert!(cut.text.ends_with("[… résultat tronqué]"));
    }
}
