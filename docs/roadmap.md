# Roadmap

## Done

- **J17**: `/edit` the last message and resend, `/retry [model]`, `/export [file.md]`.
- **J18**: light / dark theme (`theme = "auto"`), named system prompts (`[prompts]`,
  `/persona`).
- **J19**: re-ranking through `/v1/rerank`, folders watched while running
  (`auto_index`).
- **J20**: tool calling — `read_file`, `list_dir`, `search_documents`, each call
  confirmed.
- **J21**: images as input (`/add photo.png`).
- **J22**: GitHub CI and release binaries.
- **J23**: input history (`Ctrl+↑` / `Ctrl+↓`), generation speed, cost of cloud
  conversations.
- **J24**: web pages — `fetch_url` tool and `/add https://…`.
- **J25**: versions — `/edit` and `/retry` keep what they replace, `Alt+←` / `Alt+→`
  switch between versions.
- **J26**: Windows — `clip.exe`, Windows paths, CI job and release binary.
- **J27**: local re-ranking — `rerank_url` (llama.cpp `--reranking`, Text Embeddings
  Inference, Infinity).
- **J28**: preview of the highlighted conversation in the list (from the first search
  match), wider list and command suggestions.
- **J29**: find in the conversation (`Ctrl+F`).
- **J30**: MCP servers (`[mcp.<name>]`, `/mcp`).
- **J31**: `/compare <model>`, two replies side by side.
- **J32**: local model store — `/models` browses what is on disk alongside a short
  hand-picked set of well-known GGUF repositories (`Entrée` downloads, `Suppr` deletes),
  `/pull` downloads from HuggingFace (resumable, sha256-checked) and the file's own header
  supplies the metadata. Downloading and inspecting only in this milestone: J34 makes
  Qwen3 GGUFs runnable.
- **J33**: editorial visual direction — both palettes pin indexed tones, role is read from
  position instead of a coloured header, the markdown vocabulary is reduced to weight,
  rules and indentation, and the application names the wait it is in before the first
  token (phase, turning glyph, seconds) instead of freezing.
- **J34**: running local models — a `local` provider decodes a Qwen3 GGUF from the J32
  store in-process, on CPU, on its own thread, tokenizer and chat template read from the
  GGUF's own metadata. In: one architecture (Qwen3), K-quants. Out: GPU, tools, i-quants
  (unsupported by the underlying engine).

## Next

- MCP over HTTP (remote servers), and `tools/list_changed` notifications.
