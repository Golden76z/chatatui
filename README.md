# chatatui

A minimal ChatGPT-like chat client for the terminal, built with
[ratatui](https://ratatui.rs). **Local first** (Ollama by default), with optional cloud
providers: OpenAI (ChatGPT models) and Anthropic (Claude).

Any server exposing the OpenAI-compatible API (`POST /v1/chat/completions` with SSE
streaming, `GET /v1/models`) works: Ollama, llama.cpp server, LM Studio, vLLM, OpenAI.
Claude is reached through the native Anthropic Messages API.

- Streaming replies, cancellable with `Esc`
- Markdown rendering: headings, bold/italic, lists, quotes, tables, syntax-highlighted
  code blocks
- Scrollable conversation with auto-scroll that pauses when you scroll up
- History in SQLite: new conversation, list of past conversations, reopen, rename or
  delete any of them, and full-text search across all messages
- Several providers at once: the model picker lists the models of all of them, and each
  conversation remembers its provider and model
- Context tools: attach files (`/add`), empty (`/clear`) or summarize (`/compact`) the
  context, with a gauge of how full it is
- Answers from your documents (RAG): index folders of Markdown, text, source code, PDF,
  Word and LibreOffice files (`/index`), pick a collection per conversation (`/rag`), and
  replies list the passages they cite
- Configurable system prompt; network errors shown in the UI, never a crash

## Requirements

- Rust 1.88 or newer (`rustup update`)
- A C compiler: SQLite is bundled and compiled from source, as is the TLS backend
- A running server, e.g. `ollama serve` and `ollama pull llama3.2`

## Usage

```sh
cargo run --release
```

On first launch a commented configuration file is created in the platform config
directory (`~/.config/chatatui/config.toml` on Linux):

| Key | Default | Meaning |
|---|---|---|
| `default_provider` | `ollama` | Provider for new conversations |
| `system_prompt` | short assistant prompt | Sent first in every request; `'''…'''` for several lines, `""` for none |
| `connect_timeout_secs` | `5` | Connection timeout |
| `mouse_capture` | `true` | Wheel scrolling; hold Shift to select text. `false` keeps native selection |
| `[providers.<name>]` | see below | One section per provider |

Provider sections accept `kind` (`openai` or `anthropic`), `label`, `base_url`, `model`,
`api_key_env` (environment variable holding the key), `api_key` (key in the file; prefer
the variable), `max_output_tokens` (Anthropic only, default 8192) and `context_window`
(context size in tokens, for servers that do not report it).

### Context gauge

The status bar shows how full the context is, e.g. `ctx 1,6k/4,1k 37 %` (green, yellow
from 80 %, red from 95 %). Token counts come from the server after each reply
(`stream_options.include_usage` for OpenAI-compatible servers, usage events for Claude);
until then, and for messages written since, they are estimated (`≈`, about four characters
per token). The window size comes from `context_window` in the config, otherwise from the
server: Ollama `GET /api/ps` (effective size, once the model is loaded), LM Studio
`GET /api/v1/models`, llama.cpp `GET /props`, Anthropic `GET /v1/models/{id}`. OpenAI does
not report it: set `context_window` if you want the percentage.

Past `compact_threshold` (90 % by default, `0` turns it off) the status bar suggests
`/compact` after each reply. With `auto_compact = true`, the history is summarized
automatically before your next message is sent; if the summary fails or you press `Esc`,
the message stays in the input, unsent.

### Providers

Three providers are predefined; a section only overrides what it sets:

| Name | Protocol | Default URL | Key |
|---|---|---|---|
| `ollama` | OpenAI-compatible | `http://localhost:11434/v1` | none |
| `openai` | OpenAI | `https://api.openai.com/v1` | `$OPENAI_API_KEY` |
| `claude` | Anthropic Messages | `https://api.anthropic.com/v1` | `$ANTHROPIC_API_KEY` |

To use ChatGPT or Claude models, create an API key (platform.openai.com /
console.anthropic.com — API usage is billed separately from ChatGPT Plus or Claude Pro
subscriptions) and export it before starting chatatui:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Then pick a model with `F2` (or `/model claude <model>`). A provider without its key
stays listed with an explanation. Cloud models are marked `☁`: with them, the
conversation (and, later, attached files) is sent to the provider.

Other OpenAI-compatible servers are added the same way:

```toml
[providers.lmstudio]
label = "LM Studio"
base_url = "http://localhost:1234/v1"
```

Older configuration files (with `base_url` / `model` / `api_key` at the top level) still
work: those keys configure the `ollama` provider.

Conversations are stored in `~/.local/share/chatatui/chatatui.db`.

### Document indexing (RAG)

`/index <folder> [name]` walks a folder (respecting `.gitignore` and `.chatatuiignore`
files, and skipping hidden files), extracts the text of Markdown, text, source code, PDF,
`.docx` and `.odt` files, splits it into passages that remember where they come from
(`§ heading`, `p. 3`, `L12-40`), embeds them and stores them in the same database.

- `/index <name>` updates an existing collection: only files that changed are processed,
  and the ones that disappeared are dropped. `Esc` stops a run; what was done is kept.
- `--types pdf,md,docx` limits a collection to some file types (`code` means every
  source file); the choice is remembered, `--types all` lifts it.
- Files that cannot be indexed (scanned PDFs, binary files) are reported once with the
  reason and not retried until they change.
- At startup, and when `/collections` opens, chatatui checks whether the indexed folders
  changed and says which collections need an `/index`.
- `/forget <name>` deletes a collection's index (asks to confirm; your files are not
  touched).

Embeddings come from Ollama by default, so documents never leave the machine:

```sh
ollama pull bge-m3
```

```toml
[rag]
embedding_provider = "ollama"   # any configured OpenAI-compatible provider
embedding_model = "bge-m3"      # changing it re-indexes a collection on its next /index
chunk_tokens = 800              # passage size
top_k = 5                       # passages given to the model per reply
context_tokens = 3000           # their token budget
min_score = 0.3                 # similarity (0–1) below which a passage is left out
keyword_search = true           # also match the question's words (hybrid search)
exclude = ["*.min.js", "node_modules/"]   # never indexed
```

`/rag <collection>` makes the conversation search that collection before each reply. The
question (with the previous one when it is a short follow-up) is searched by meaning
(embedding similarity) and by keywords (SQLite FTS5, accents ignored, which catches names,
codes and rare terms), the two rankings are merged (reciprocal rank fusion), and the best
passages are added to the prompt as numbered sources. The reply
lists them underneath (`Sources : [1] plan.docx § Séance 2`), keeping only those it cites
when it cites any; they are saved with the conversation. `/prompt` shows the passages
sent, `/context` what they cost, and the status bar shows the collection (`⌕ cours`),
with ☁ when the provider is in the cloud, since the passages then leave the machine.
`/rag off` stops; a new conversation keeps the current collection.

## Commands

Type `/` in the input to see the commands (↑↓ to choose, `Tab` to complete, `Enter` to
run), or press `Ctrl+P` for the palette:

| Command | Action |
|---|---|
| `/new` | New conversation (`Ctrl+N`) |
| `/history` | Conversation list (`Ctrl+L`) |
| `/rename <title>` | Rename the conversation |
| `/copy [code [n]]` | Copy the last reply (`Ctrl+Y`), or its code block `n` (default: the last; blocks are numbered when there are several) |
| `/delete` | Delete the conversation (run twice to confirm) |
| `/model [provider] [model]` | Choose the model (`Ctrl+M` / `F2`), or switch directly: `/model qwen2.5:7b`, `/model claude`, `/model claude <model>` |
| `/context` | Context window, tokens used (measured or estimated) and where they go |
| `/prompt` | The exact messages the next request will send |
| `/add <file>` | Attach a text file (≤ 256 KB) to the context; `Tab` completes the path |
| `/clear` | Empty the context: messages stay on screen but are no longer sent |
| `/compact` | Ask the model to summarize the history; the summary replaces it in the context |
| `/index <folder> [name] [--types …]` | Index a folder into a document collection (named after the folder by default), or `/index <name>` to update one; `Tab` completes the path, `Esc` stops |
| `/collections` | Indexed collections, and the result of the last `/index` |
| `/rag [collection\|off]` | Answer from a collection of documents (per conversation), or stop |
| `/forget <collection>` | Delete a collection's index (run twice to confirm); files are not touched |
| `/help` | Commands and key bindings (`F1`) |
| `/quit` | Quit (`Ctrl+C`) |

Start a message with `//` to send text that begins with a slash.

## Keys

| Key | Action |
|---|---|
| `Enter` | Send |
| `Shift+Enter` / `Alt+Enter` / `Ctrl+J` | New line (`Shift+Enter` needs the kitty keyboard protocol) |
| `Esc` | Close the popup or panel, otherwise cancel the running generation, otherwise stop indexing |
| `Ctrl+N` | New conversation |
| `Ctrl+L` | Conversation list: `↑`/`↓` to choose, `Enter` to open, type to search every message (accents and case ignored), `Ctrl+R` to rename, `Suppr` twice to delete, `Esc` clears the search then closes |
| `Ctrl+M` / `F2` | Choose the model (type to filter). `Ctrl+M` needs the kitty keyboard protocol |
| `Ctrl+P` | Command palette |
| `Ctrl+Y` | Copy the last reply |
| `F1` | Help |
| `PgUp` / `PgDn`, mouse wheel | Scroll the conversation |
| `Up` / `Down` (empty input) | Scroll by one line |
| `Ctrl+Home` / `Ctrl+End` | Jump to the top / back to the bottom (resumes auto-scroll) |
| `Ctrl+C` / `Ctrl+Q` | Quit |

Terminals with the kitty keyboard protocol (kitty, WezTerm, Ghostty, foot, recent
Alacritty) get `Shift+Enter` and `Ctrl+M`; elsewhere use `Alt+Enter` and `F2`. The status
bar shows the keys that work in your terminal.

## Architecture

```
Event (key, tick, LLM, storage)
  → keymap → Action
  → App::update(Action) -> Vec<Command>   pure: no I/O
  → Runtime executes the commands         streaming task, storage thread, model list
  → results come back as events on one channel
ui::render(&App, frame)                   read-only
```

- `llm/`: `LlmClient` trait with `OpenAiCompatibleClient` and `AnthropicClient` (one client
  per configured provider), incremental SSE parser, the streaming task (cancelled through
  a `CancellationToken`), and a scripted mock.
- `context/`: `ContextProvider` trait and `NoContext`. **This is where RAG plugs in**: a
  provider receives the conversation and returns `ContextChunk { source, text }`s, which
  `prompt.rs` merges into the system message — the same path as files attached with
  `/add`. It runs inside the streaming task, so a slow retrieval never blocks the UI.
- `rag/`: text extraction (`extract.rs`), passage splitting (`chunk.rs`), the `Embedder`
  trait (`embed.rs`), the collection tables (`store.rs`), the background indexing job
  (`indexer.rs`, cancellable, reports progress as events) and `RagContext`
  (`retrieve.rs`), the `ContextProvider` that searches the conversation's collection. The
  streaming task reports the passages it used, which become the reply's citations.
- `markdown/` + `transcript.rs`: markdown → wrapped lines, cached per message and width;
  while streaming only the last message is re-rendered, at most once per tick.
- `storage/`: SQLite schema with migrations, and a worker thread that runs requests in
  order.

See [PLAN.md](PLAN.md) for the design decisions.

## Development

```sh
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
cargo insta review   # after an intentional UI change
```

## License

Copyright (c) 2026 Damien <dap@csmrouen.com>

This project is licensed under the MIT license ([LICENSE](./LICENSE) or <http://opensource.org/licenses/MIT>)
