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

## Next

### J24 — Web pages

- **`fetch_url` tool**: the model may read a web page (HTML turned into text, size
  limited), confirmed like the other tools; `/add https://…` attaches a page directly.

### Later

- Windows support (clipboard through `clip.exe`, paths, CI job).
- Conversation branches: keep the replaced messages of `/edit` and `/retry` and switch
  between versions.
- Re-ranking with a local cross-encoder when the server has no `/v1/rerank`.
