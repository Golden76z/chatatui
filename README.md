# chatatui

[![CI](https://github.com/Golden76z/chatatui/actions/workflows/ci.yml/badge.svg)](https://github.com/Golden76z/chatatui/actions/workflows/ci.yml)

A minimal ChatGPT-like chat client for the terminal, built with
[ratatui](https://ratatui.rs). **Local first** (Ollama by default), with optional cloud
providers: OpenAI (ChatGPT models) and Anthropic (Claude).

Any server exposing the OpenAI-compatible API (`POST /v1/chat/completions` with SSE
streaming, `GET /v1/models`) works: Ollama, llama.cpp server, LM Studio, vLLM, OpenAI.
Claude is reached through the native Anthropic Messages API.

- Streaming replies, cancellable with `Esc`; before the first word the application names
  the wait it is in — connecting, searching the documents, the model thinking, a tool
  running — with the seconds it has taken
- Markdown rendering: headings with a rule, bold/italic, lists, quotes, tables, and
  syntax-highlighted code blocks set in by an indent
- Role read from position: the question indented and dim, the reply at the left margin
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
- Local model store: browse well-known GGUF models and the ones already on disk in one
  list (`/models`), download from HuggingFace (`/pull`), read their metadata — downloading
  and inspecting only, running them comes later
- Configurable system prompt; network errors shown in the UI, never a crash

## Install

- **Binaries**: Linux (x86_64, arm64), macOS (Apple silicon) and Windows (x86_64)
  archives are attached to each
  [release](https://github.com/Golden76z/chatatui/releases); unpack and run `chatatui`
  (`chatatui.exe` on Windows, in Windows Terminal).
- **From source**: `cargo install --git https://github.com/Golden76z/chatatui --locked`
  (Rust 1.88 or newer, and a C compiler: SQLite and the TLS backend are compiled from
  source).

You also need a model server, e.g. `ollama serve` and `ollama pull llama3.2`. Optional:
`tesseract-ocr` and `poppler-utils` to index scanned PDFs, `wl-clipboard` or `xclip` for
`/copy` (Windows and WSL use `clip.exe`, macOS `pbcopy`).

## Usage

```sh
chatatui            # or, from a clone: cargo run --release
```

On first launch a commented configuration file is created in the platform config
directory: `~/.config/chatatui/config.toml` on Linux,
`~/Library/Application Support/chatatui/config.toml` on macOS,
`%APPDATA%\chatatui\config\config.toml` on Windows:

| Key | Default | Meaning |
|---|---|---|
| `default_provider` | `ollama` | Provider for new conversations |
| `system_prompt` | short assistant prompt | Sent first in every request; `'''…'''` for several lines, `""` for none |
| `connect_timeout_secs` | `5` | Connection timeout |
| `mouse_capture` | `true` | Wheel scrolling; hold Shift to select text. `false` keeps native selection |
| `theme` | `auto` | `dark`, `light`, or `auto` (from the terminal's `COLORFGBG`, dark when unknown); also picks the code-highlighting theme |
| `compact_threshold`, `auto_compact` | `90`, `false` | See [Context gauge](#context-gauge) |
| `[prompts]` | none | Named system prompts, e.g. `prof = "Explique pas à pas."`; `/persona prof` uses one in the current conversation (saved with it, shown as `✦ prof`) |
| `[providers.<name>]` | see below | One section per provider |

Provider sections accept `kind` (`openai` or `anthropic`), `label`, `base_url`, `model`,
`api_key_env` (environment variable holding the key), `api_key` (key in the file; prefer
the variable), `max_output_tokens` (Anthropic only, default 8192), `context_window`
(context size in tokens, for servers that do not report it), and `price_input` /
`price_output` / `currency` (price per million tokens, e.g. `3.0` and `15.0`): the status
bar then shows what the conversation has cost since it was opened (`· 0,0220 €`), and
`/context` the details. While a reply streams, the status bar shows its speed
(`◐ Génération… 42 t/s`); `/context` keeps the last one (tokens, seconds, time to first
token).

### Interface

Who speaks is read from position, not from a label: the question is indented and dimmed,
the reply sits at the left margin. The roles that position cannot tell apart — `système`,
`résumé de la conversation`, `fichier joint`, `outil` — keep a small dim label above their
body. Rhythm replaces frames: one blank line after every message, one more before a
question, so a turn is separated from the next by two and its own halves by one.

Inside a reply, headings carry a heavier weight and a rule the width of their own text,
lists a middle dot `·`, quotes a bar `│`. A code block is simply indented by four columns,
with its language on the line above and, when the reply holds several blocks, the number
`/copy code N` asks for, right-aligned on that same line.

The input carries one horizontal rule above it rather than a box. While no word has
arrived yet, the application names the wait where the text itself will appear, with a
turning glyph and the seconds past the first, in the order the waits occur: `recherche
dans 2 collections…`, `connexion…`, `llama3.2 réfléchit…`, `exécution de read_file…`. A
tool-using answer names them again for every round, the second one carrying the tool
output and so waiting longest. The version marker
`‹ 2/3 ›` goes under the body of the reply, and only when there is another version to
compare it against.

Both palettes pin indexed tones rather than the eight basic ANSI colours, which every
terminal theme redefines; `theme = "light"` switches to the light one.

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

Conversations are stored in `~/.local/share/chatatui/chatatui.db` (macOS: next to the
configuration; Windows: `%APPDATA%\chatatui\data\chatatui.db`).

### Document indexing (RAG)

`/index <folder> [name]` walks a folder (respecting `.gitignore` and `.chatatuiignore`
files, and skipping hidden files), extracts the text of Markdown, text, source code, PDF,
`.docx` and `.odt` files, splits it into passages that remember where they come from
(`§ heading`, `p. 3`, `L12-40`), embeds them and stores them in the same database.

- `/index <name>` updates an existing collection: only files that changed are processed,
  and the ones that disappeared are dropped. `Esc` stops a run; what was done is kept.
- `--types pdf,md,docx` limits a collection to some file types (`code` means every
  source file); the choice is remembered, `--types all` lifts it.
- Scanned PDFs are read with OCR when `tesseract` and `pdftoppm` are installed
  (`sudo apt install tesseract-ocr tesseract-ocr-fra poppler-utils`); scans skipped
  before OCR was available are read on the next `/index`.
- Files that cannot be indexed (binary files, blank scans) are reported once with the
  reason and not retried until they change.
- At startup, and when `/collections` opens, chatatui checks whether the indexed folders
  changed and says which collections need an `/index`; with `auto_index = true` it
  updates them itself, in the background, and keeps watching their folders while it runs
  (a few seconds after a file changes).
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
ocr = true                      # read scanned PDFs (if tesseract and pdftoppm are installed)
ocr_languages = "fra+eng"       # Tesseract languages (missing ones are skipped)
auto_index = false              # update changed collections (at startup and while running)
rerank_model = ""               # e.g. "bge-reranker-v2-m3": re-score passages (see below)
rerank_provider = ""            # provider serving it (default: embedding_provider)
rerank_url = ""                 # or a dedicated rerank server, e.g. "http://localhost:8081"
rerank_candidates = 20          # passages re-scored before keeping top_k
```

With `rerank_model` or `rerank_url` set, the best `rerank_candidates` passages of the
hybrid search are re-scored by a cross-encoder, and the `top_k` best are kept. More
precise on pointed questions, at the cost of one extra request per reply; if the reranker
is unreachable, the hybrid order is used. `/context` shows the reranker in use.

The request goes to `…/rerank` in the Cohere / Jina format (llama.cpp server, vLLM,
Infinity), or in the Text Embeddings Inference format, detected on the first reply. Ollama
has no rerank endpoint, but a cross-encoder runs locally with llama.cpp, next to Ollama:

```sh
# a GGUF of bge-reranker-v2-m3 (multilingual), e.g. from Hugging Face
llama-server -m bge-reranker-v2-m3-Q8_0.gguf --reranking --port 8081
```

```toml
[rag]
rerank_url = "http://localhost:8081"   # rerank_model can stay empty: the server's model
```

Text Embeddings Inference (`--model-id BAAI/bge-reranker-base`) and Infinity work the same
way with their own port.

`/rag <collection>` makes the conversation search that collection before each reply. The
question (with the previous one when it is a short follow-up) is searched by meaning
(embedding similarity) and by keywords (SQLite FTS5, accents ignored, which catches names,
codes and rare terms), the two rankings are merged (reciprocal rank fusion), and the best
passages are added to the prompt as numbered sources. The reply
lists them underneath (`Sources : [1] plan.docx § Séance 2`), keeping only those it cites
when it cites any; they are saved with the conversation. `/prompt` shows the passages
sent, `/context` what they cost, and the status bar shows the collection (`⌕ cours`),
with ☁ when the provider is in the cloud, since the passages then leave the machine.
`/rag cours,tp` searches several collections at once (sources then name their
collection: `cours › plan.docx`). `/rag off` stops; a new conversation keeps the current
collections.

### Tools

With `/tools on` (or `[tools] enabled = true`), the model may call tools: `read_file`
(a text file), `list_dir` (a folder), `search_documents` (your indexed collections) and
`fetch_url` (a web page, turned into text; PDFs are read too).
Each call opens a question — `Enter` allows it, `t` allows every call of this
conversation, `Esc` refuses (the model is told and answers without it) — and appears as a
card in the conversation (`🔧 lire ~/notes.md · ≈ 350 tokens transmis`). Files that
usually hold secrets (`~/.ssh`, `.env`, keys, credentials…) are refused whatever the
answer. The popup warns when the result will go to a cloud provider. Tools use the
OpenAI `tools` / `tool_calls` format (OpenAI, Ollama, llama.cpp, LM Studio, vLLM) or
Claude's `tool_use` blocks; models without tool support reject the request, hence off by
default.

### MCP servers

Servers speaking the [Model Context Protocol](https://modelcontextprotocol.io) add their
tools to the built-in ones. Each `[mcp.<name>]` section starts one at launch (stdio
transport):

```toml
[mcp.fichiers]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "~/Documents"]

[mcp.git]
command = "uvx"
args = ["mcp-server-git", "--repository", "~/projet"]

[mcp.autre]
command = "mon-serveur"
env = { API_TOKEN = "…" }   # extra environment variables
cwd = "~/outils"            # working directory
enabled = false             # keep the section without starting it
```

`/mcp` shows each server's state (starting, ready with its tools, or why it failed). Their
tools reach the model as `<server>__<tool>` once `/tools on`; every call is confirmed like
the others, and the popup warns that a server's tool may also change or send data (unlike
the built-in ones, which only read). `~` is expanded in `command`, `args` and `cwd`. The
servers stop with chatatui.

### Local models

`/models` is the way in: one list holding what is already on disk (`●`) and a short
hand-picked set of well-known GGUF repositories that are not (`○`), smallest first. Type
to filter, `Entrée` asks HuggingFace for that repository's files and opens the quantization
picker, `Suppr` deletes a downloaded one. `Entrée` works on a row already on disk too —
that is how you fetch a second quantization of a model you already have, since a
repository drops out of the offered list once anything from it has landed. The line under
the list details the highlighted model; the parameter count on the right is the signal for
whether it will fit.

The offered list is chosen, not measured: sorting HuggingFace by download count surfaces
embedding, speech and image models, mirrors and "uncensored" forks rather than models worth
offering. It is deliberately short and will age — `/pull` takes any repository.

`/pull <dépôt> [fichier]` downloads a GGUF file from HuggingFace. The repository is
written `owner/name`, but a pasted page URL works too:

```
/pull unsloth/Qwen3-0.6B-GGUF
/pull https://huggingface.co/unsloth/Qwen3-0.6B-GGUF
/pull unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf
```

A URL that names a branch or a commit (`.../tree/v2.0`, `.../blob/v2.0/m.gguf`) downloads
that revision rather than the default branch. Models split across several files
(`-00001-of-00002.gguf`) are not supported, and `/pull` says so instead of reporting an
empty repository.

Without a file name, a popup lists the repository's `.gguf` files, smallest first (type to
filter, `Enter` downloads). The status bar shows the progress and the rate; `Esc` stops the
download, and running the same `/pull` again resumes it from where it stopped instead of
starting over. When HuggingFace publishes a checksum, the finished file is verified against
it; when it does not, the model is listed as `⚠ non vérifié`.

What a downloaded model shows comes from the file's own header: quantization, size,
architecture, context window and parameter count. `/rm <dépôt> <fichier>` deletes one file
and its inventory row from the input line, the same as `Suppr` in the list.

```toml
[models]
dir = "~/modeles"      # default: <data>/models, next to the database
token_env = "HF_TOKEN" # environment variable holding a token, for gated repositories
```

Files are kept in `~/.local/share/chatatui/models/<owner>/<nom>/<fichier>.gguf` (macOS:
next to the configuration; Windows: `%APPDATA%\chatatui\data\models`), beside a `.part`
file while the download is in progress. The token, like the provider API keys, is read by
the runtime and never reaches the UI.

**A downloaded model cannot answer yet.** This milestone downloads and inspects GGUF
files; it does not run them. To talk to a local model today, serve it with Ollama or
llama.cpp and point a provider at it.

## Commands

Type `/` in the input to see the commands (↑↓ to choose, `Tab` to complete, `Enter` to
run), or press `Ctrl+P` for the palette:

| Command | Action |
|---|---|
| `/new` | New conversation (`Ctrl+N`) |
| `/history` | Conversation list (`Ctrl+L`) |
| `/rename <title>` | Rename the conversation |
| `/persona [name\|off]` | Use a named system prompt from `[prompts]` in this conversation, or go back to `system_prompt` |
| `/tools [on\|off]` | Let the model read files, list folders and search your documents (each call confirmed) |
| `/mcp` | MCP servers: state and tools |
| `/edit` | Put your last message back in the input: sending it replaces it and what followed (`Esc` cancels); the old version is kept (`Alt+←`) |
| `/retry [model]` | Replace the last reply with a new one, from another model if given (`/retry claude`); the old reply is kept (`Alt+←`) |
| `/compare <model>` | The last question answered again by another model, both replies side by side (`←`/`1` or `2`/`→` keeps one, `Esc` keeps the new one); the other stays as a version, and each version shows its model |
| `/export [file.md]` | Save the conversation as Markdown (named after its title by default; never overwrites) |
| `/find [text]` | Search the open conversation (`Ctrl+F`): matches highlighted, accents and case ignored |
| `/copy [code [n]]` | Copy the last reply (`Ctrl+Y`), or its code block `n` (default: the last; blocks are numbered when there are several) |
| `/delete` | Delete the conversation (run twice to confirm) |
| `/model [provider] [model]` | Choose the model (`Ctrl+M` / `F2`), or switch directly: `/model qwen2.5:7b`, `/model claude`, `/model claude <model>` |
| `/context` | Context window, tokens used (measured or estimated) and where they go |
| `/prompt` | The exact messages the next request will send |
| `/add <file\|url>` | Attach a text file (≤ 256 KB), an image (PNG, JPEG, GIF, WebP, ≤ 5 MB) for vision models (llava, qwen2.5-vl, Claude, GPT-4o…), or a web page (`/add https://…`: its text, PDFs included); `Tab` completes the path |
| `/clear` | Empty the context: messages stay on screen but are no longer sent |
| `/compact` | Ask the model to summarize the history; the summary replaces it in the context |
| `/index <folder> [name] [--types …]` | Index a folder into a document collection (named after the folder by default), or `/index <name>` to update one; `Tab` completes the path, `Esc` stops |
| `/collections` | Indexed collections, and the result of the last `/index` |
| `/rag [collection,…\|off]` | Answer from one or more collections of documents (per conversation), or stop |
| `/forget <collection>` | Delete a collection's index (run twice to confirm); files are not touched |
| `/pull <dépôt> [fichier]` | Download a GGUF model from HuggingFace (a repository, or a pasted page URL); without a file name, pick one from the repository's list. `Esc` stops it, the same `/pull` resumes it |
| `/models` | Models on this machine and a short list of well-known ones that are not; `Entrée` downloads, `Suppr` deletes |
| `/rm <dépôt> <fichier>` | Delete a downloaded model (the file and its inventory row) |
| `/help` | Commands and key bindings (`F1`) |
| `/quit` | Quit (`Ctrl+C`) |

Start a message with `//` to send text that begins with a slash.

## Keys

| Key | Action |
|---|---|
| `Enter` | Send |
| `Shift+Enter` / `Alt+Enter` / `Ctrl+J` | New line (`Shift+Enter` needs the kitty keyboard protocol) |
| `Esc` | Close the popup or panel, otherwise cancel the running generation, otherwise stop indexing, otherwise stop a download |
| `Ctrl+N` | New conversation |
| `Ctrl+L` | Conversation list: `↑`/`↓` to choose (the highlighted conversation is previewed; while searching, from its first match; `PgUp`/`PgDn` scroll it), `Enter` to open, type to search every message (accents and case ignored), `Ctrl+R` to rename, `Suppr` twice to delete, `Esc` clears the search then closes |
| `Ctrl+M` / `F2` | Choose the model (type to filter). `Ctrl+M` needs the kitty keyboard protocol |
| `Ctrl+P` | Command palette |
| `Ctrl+Y` | Copy the last reply |
| `Ctrl+F` | Find in the conversation: type to search, `Enter` / `↓` next match, `Shift+Enter` / `↑` previous, `Esc` closes |
| `Ctrl+↑` / `Ctrl+↓` | Previous / next message sent (all conversations); back to what you were typing after the last |
| `Alt+←` / `Alt+→` (empty input) | Previous / next version of the last replaced exchange (`‹ 2/3 ›` under the reply, after `/edit` or `/retry`) |
| `F1` | Help |
| `PgUp` / `PgDn`, mouse wheel | Scroll the conversation |
| `Up` / `Down` (empty input) | Scroll by one line |
| `Ctrl+Home` / `Ctrl+End` | Jump to the top / back to the bottom (resumes auto-scroll) |
| `Ctrl+C` / `Ctrl+Q` | Quit |

Terminals with the kitty keyboard protocol (kitty, WezTerm, Ghostty, foot, recent
Alacritty) and the Windows console get `Shift+Enter` and `Ctrl+M`; elsewhere use
`Alt+Enter` and `F2`. The status
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
- `models/`: the HuggingFace client (`hub.rs`), the GGUF header parser (`gguf.rs`), the
  inventory of downloaded models (`store.rs`) and the cancellable, resumable download job
  (`download.rs`, reports progress as events). Nothing here runs a model.
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
