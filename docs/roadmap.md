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

## Next

- MCP servers: external tools next to `read_file` and `fetch_url`, each call confirmed.
- `/compare <model>`: the same question to a second model, both replies side by side.
