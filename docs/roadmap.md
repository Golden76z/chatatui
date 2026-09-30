# Roadmap

Ideas for after J16, from most useful day to day to most ambitious. Suggested order:
start with **J17** (edit / retry / export), then either re-ranking (heavy RAG use for
courses) or tool calling (towards a real assistant).

## J17 — Everyday conversation tools

- **Edit and resend a previous message**: `↑` on an empty input (or `/edit`) loads the
  last user message; sending it replaces that message and drops what followed, so the
  conversation restarts from there with the new version.
- **Regenerate the last reply** (`/retry`), optionally with another model
  (`/retry claude`) to compare answers.
- **Export a conversation** to Markdown (`/export [file]`): messages, attachments as
  links, and the cited sources under each reply.

## Comfort

- **Light / dark theme** and configurable colours. Code highlighting already uses
  syntect: expose its theme (`theme = "…"`) and derive the UI colours from a `[theme]`
  section.
- **Named system prompts**: `[prompts.prof]`, `[prompts.relecteur]` in the config,
  `/persona <name>` per conversation (stored with it), shown in the status bar.

## RAG, further

- **Re-ranking**: after hybrid search, a cross-encoder (e.g. `bge-reranker-v2-m3` via
  an OpenAI-compatible `/rerank` endpoint, or Ollama) re-scores the ~20 best passages
  and keeps the top-k. Much more precise on pointed questions; costs one extra request
  per reply. Off by default (`[rag] rerank_model`).
- **Watch indexed folders while chatatui runs** (`notify` crate): re-index a collection
  shortly after one of its files changes, instead of waiting for the next start.

## Bigger projects

- **Tool calling**: the model may call tools (read a file, search a collection, list a
  folder, fetch a URL…), each call confirmed by the user in a popup. OpenAI `tools` and
  Anthropic `tool_use` formats behind one `Tool` trait; results shown as cards in the
  transcript. The largest change: the streaming task becomes a loop (reply → tool call →
  result → reply).
- **Images as input** (`/add photo.png`) for models that accept them (llava, qwen2.5-vl,
  Claude, GPT-4o): base64 content parts, a card in the transcript, refused with a clear
  message for text-only models.
