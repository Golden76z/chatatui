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
- History in SQLite: new conversation, list of past conversations, reopen any of them
- Several providers at once: the model picker lists the models of all of them, and each
  conversation remembers its provider and model
- Context tools: attach files (`/add`), empty (`/clear`) or summarize (`/compact`) the
  context, with a gauge of how full it is
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

## Commands

Type `/` in the input to see the commands (↑↓ to choose, `Tab` to complete, `Enter` to
run), or press `Ctrl+P` for the palette:

| Command | Action |
|---|---|
| `/new` | New conversation (`Ctrl+N`) |
| `/history` | Conversation list (`Ctrl+L`) |
| `/model [provider] [model]` | Choose the model (`Ctrl+M` / `F2`), or switch directly: `/model qwen2.5:7b`, `/model claude`, `/model claude <model>` |
| `/context` | Context window, tokens used (measured or estimated) and where they go |
| `/prompt` | The exact messages the next request will send |
| `/add <file>` | Attach a text file (≤ 256 KB) to the context; `Tab` completes the path |
| `/clear` | Empty the context: messages stay on screen but are no longer sent |
| `/compact` | Ask the model to summarize the history; the summary replaces it in the context |
| `/help` | Commands and key bindings (`F1`) |
| `/quit` | Quit (`Ctrl+C`) |

Start a message with `//` to send text that begins with a slash.

## Keys

| Key | Action |
|---|---|
| `Enter` | Send |
| `Shift+Enter` / `Alt+Enter` / `Ctrl+J` | New line (`Shift+Enter` needs the kitty keyboard protocol) |
| `Esc` | Close the popup or panel, otherwise cancel the running generation |
| `Ctrl+N` | New conversation |
| `Ctrl+L` | Conversation list (`↑`/`↓` to choose, `Enter` to open) |
| `Ctrl+M` / `F2` | Choose the model (type to filter). `Ctrl+M` needs the kitty keyboard protocol |
| `Ctrl+P` | Command palette |
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
