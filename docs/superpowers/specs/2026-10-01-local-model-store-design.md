# J32 — Local model store (design)

Download GGUF models from HuggingFace, store them, inspect them. No inference: running
them is J33.

## Why

chatatui is a pure client today: `LlmClient` talks HTTP/SSE to a server somebody else
started, and getting a model means leaving the TUI to run `ollama pull`. The goal is a
model runtime that is ours — we own the format, the store and the lifecycle — without
depending on Ollama and without vendoring an inference engine.

"From scratch" is staged in layers, because only some of them are realistically ours:

| Layer | In this milestone |
|---|---|
| 1. Download + content-addressed store | yes |
| 2. GGUF parsing | yes |
| 3. BPE tokenizer | vocabulary located, encoder deferred to J33 |
| 4. Inference engine (kernels, matmul, GPU) | J33, on a tensor backend |

Layers 1–3 are needed whichever engine J33 picks, so they are worth building first and
on their own.

### Accepted limitation

J32 alone downloads and inspects models it cannot yet run. That gap is deliberate: the
alternative was one oversized milestone from download to generation. Local models stay
out of the `F2` picker until J33 can answer with them.

## Scope

In: listing a repository's GGUF files, resumable download with progress and
cancellation, sha256 verification, GGUF metadata parsing, a SQLite inventory, `/pull`,
`/models`, deletion.

Out: inference, the BPE encoder, GPU support, the Ollama registry, automatic model
updates, concurrent downloads.

## Architecture

A new `src/models/` module, shaped like `src/rag/` so it introduces no new pattern:

```
src/models/
  mod.rs       ModelsConfig, shared types, ModelError
  hub.rs       HuggingFace client: list a repository's .gguf files, download with resume
  gguf.rs      GGUF parsing: header, KV metadata, tensor descriptors
  store.rs     SQLite inventory plus the files on disk
  download.rs  the background job
```

`download.rs` plays the part `rag/indexer.rs` plays for indexing (one cancellable job
with progress events); `models/store.rs` follows `rag/store.rs` (free functions taking a
`&Connection`, so both the storage worker and the job can call them on their own
connection).

Control flow is the existing one, unchanged:

```
/pull … → Action → App::update → Effect::StartPull
        → runtime spawns a tokio task
        → AppEvent::Pull(PullEvent) → App::update → status bar
```

`App::update` stays pure; no I/O moves into it.

## Layer 1 — download (`hub.rs`, `download.rs`)

### Listing

`GET https://huggingface.co/api/models/{repo}/tree/{rev}` returns one entry per file
with `path`, `size` and, for LFS files, `lfs.oid`. Keep the `.gguf` entries. One JSON
request; no HTML scraping.

### Transfer

`GET https://huggingface.co/{repo}/resolve/{rev}/{path}` (redirects to the CDN, which
`reqwest` follows). The body streams into `<store>/tmp/<hash>.part`, where `<hash>` is
derived from `repo`, `revision` and `path` so a given file always resumes into the same
part file.

Resume: when a part file exists, send `Range: bytes=<len>-` and append on `206 Partial
Content`. A `200 OK` means the server ignored the range, so the part file is truncated
and the transfer restarts from zero rather than producing a corrupt file.

The file name comes from a remote API, so only a bare name is accepted: anything holding
a path separator, `..`, or an absolute prefix is refused before it can be joined onto the
store directory. That one rule also rejects the multi-part GGUF files (`-00001-of-00002`)
that live in subdirectories, which J32 does not support; the error says so.

Completion: `sync_all`, then an atomic rename into
`<store>/models/{owner}/{repo}/{file}`. The real directory nesting avoids the name
collisions a flattened `owner__repo` scheme would allow.

### Verification

sha256 is computed while writing. A resumed transfer first reads the existing part file
through the hasher before appending, since the digest covers the whole file and not just
the new bytes — a couple of seconds of I/O on a large part file, and the only way the
check means anything after a resume. At rename time the digest is compared with
`lfs.oid`. When the
API exposes no oid (a GGUF small enough not to be an LFS file), the model is recorded
with `sha256 = NULL` and the UI says the file could not be verified — never that it was.

A mismatch deletes the part file and reports a failure: a wrong file is worse than no
file.

### Progress and cancellation

`PullEvent::Progress { repo, file, done, total, rate }` is emitted at most once every
100 ms, not once per chunk, so a fast link cannot flood the event channel — the same
reasoning that caps markdown work at the tick rate.

`CancellationToken`, as for indexing. Cancelling keeps the part file, so the next
`/pull` of the same file resumes, including after chatatui is restarted: the part file
on disk is the whole resume state, nothing is recorded in the database.

```rust
pub enum PullEvent {
    /// The repository's files are being listed.
    Listing { repo: String },
    Progress { repo: String, file: String, done: u64, total: Option<u64>, rate: u64 },
    /// Downloaded, verified and inspected.
    Finished(LocalModel),
    Failed { repo: String, file: String, error: String },
    Cancelled { repo: String, file: String },
}
```

`total` is optional because a server may answer without `Content-Length`.
`LocalModel` is the inventory row described under Store below: what was downloaded,
plus what `gguf.rs` read out of it.

## Layer 2 — GGUF parsing (`gguf.rs`)

Read-only, header and metadata only. The tensor data is never loaded: the parser reads
the first pages of a file that may be several gigabytes.

Format: the `GGUF` magic, a `u32` version (3, with 2 accepted), `tensor_count` and
`metadata_kv_count` as `u64`, then the typed key/value pairs (string keys as `u64`
length plus bytes; value types 0–12, including arrays), then one descriptor per tensor
(name, `n_dims`, dimensions, `ggml_type`, offset).

Extracted:

- `general.architecture`, `general.name`
- the dominant quantization, derived from the tensors' `ggml_type`
- `{arch}.context_length`, `{arch}.block_count`, `{arch}.attention.head_count`,
  `{arch}.attention.head_count_kv`, `{arch}.embedding_length`
- the parameter count, summed from the tensor dimensions
- `tokenizer.ggml.model` (`gpt2`, `llama`, `spm`)

Every field is optional: a missing key yields `None`, never a panic. A truncated file, an
unknown version or an unknown value type yields a `GgufError` carrying a readable
message, and the downloaded file is left alone.

### The vocabulary is located, not stored

`tokenizer.ggml.tokens` and `.merges` are megabytes of data whose only consumer is the
encoder that J33 will write. J32 records the tokenizer kind and nothing else; J33 reads
the arrays back out of the GGUF, which costs milliseconds. Copying them into SQLite now
would duplicate data with no reader.

## Layer 3 — tokenizer

Deferred to J33, with the engine that needs it. `gguf.rs` exposes the tokenizer kind and
can locate the vocabulary arrays, so the encoder is additive work.

## Store (`store.rs`)

Files live at `<data>/models/{owner}/{repo}/{file}.gguf`, under the directory from
`[models] dir` when set, otherwise `<data>/models` next to `chatatui.db`.

Schema **v11** (the database is at v10; migrations are append-only):

```sql
CREATE TABLE local_models (
    id             INTEGER PRIMARY KEY,
    repo           TEXT NOT NULL,
    revision       TEXT NOT NULL,
    file           TEXT NOT NULL,
    path           TEXT NOT NULL,
    bytes          INTEGER NOT NULL,
    sha256         TEXT,
    architecture   TEXT,
    quantization   TEXT,
    context_length INTEGER,
    parameters     INTEGER,
    downloaded_at  INTEGER NOT NULL
);
CREATE UNIQUE INDEX local_models_file ON local_models (repo, file);
```

`sha256` is `NULL` when the API exposed no oid. The unique index makes re-downloading an
idempotent upsert, matching how messages are written.

The index is on `(repo, file)` and **not** on `(repo, revision, file)`: the path on disk
carries no revision, so two revisions of one filename would otherwise own two rows and
fight over one file. `revision` is recorded as metadata, and pulling another revision of
a file replaces both the file and the row.

Deleting a model removes the file and the row; a file already gone is not an error, the
row goes anyway.

## UI

- `/pull <repo> [file]` — without a file, a picker lists the repository's `.gguf` files
  (name, quantization, size) and the chosen one is downloaded; with a file, it is
  downloaded directly.
- `/models` — a read-only inventory overlay: repository, quantization, size,
  architecture, context window, and whether the checksum was verified.
- `/rm <repo> <file>` — deletes a downloaded model, file and row. Deletion is a command
  rather than `Suppr` inside the overlay because `/forget <collection>` already sets that
  precedent, and it keeps `/models` a plain `OverlayKind::Text` popup like `/collections`
  instead of needing a new overlay kind.
- While a download runs the status bar shows `⬇ Qwen2.5-7B Q4_K_M 2,1/4,4 Go · 18 Mo/s`
  and `Esc` cancels — the slot indexing already uses, so nothing new is invented.
- Two new `Overlay` variants — `Models` (`Text`, scrollable) and `GgufPicker` (`List`,
  filterable, `Enter` starts the download) — one shown at a time like every other popup.
  `Esc` closes the topmost one before cancelling a download.
- Local models are **absent from the `F2` picker** in J32: with no engine they cannot be
  selected, and a greyed-out group would be state with no behaviour. They join it in J33.

UI strings are French, code and comments English, as everywhere else.

## Errors

`ModelError` with `thiserror`, in the shape of `LlmError`: `Http`, `NotFound`, `Gguf`,
`Io`, `Checksum`, `Cancelled`. Every variant renders to a French sentence shown in the
UI. No `unwrap()` outside tests; a failed download never takes the app down.

## Configuration

```toml
[models]
# Where models are stored (default: <data>/models)
dir = ""
# Environment variable holding a HuggingFace token, for gated repositories
token_env = "HF_TOKEN"
```

The token is read like provider keys: from the environment, never reaching `App`, never
printed in `Debug`.

## Dependencies

One addition: `sha2`, pure Rust, for checksum verification. `reqwest` already has
`stream`, and `futures`, `tokio`, `serde_json` and `directories` are in place. The
`cargo install` story and the release binary size are unaffected.

## Testing

- `gguf.rs`: fixtures built in the test itself (header, a few KV pairs, one tensor), so
  no multi-gigabyte binary enters the repository. Cases: v2 and v3, a string array, a
  missing key, a truncated file, an unknown value type, an unknown architecture.
- `hub.rs` and `download.rs`: a local tokio server that honours `Range`, that ignores it,
  and that cuts mid-transfer. Cases: a full download, a resume from a part file, a
  restart after a `200` answer to a range request, cancellation mid-transfer then resume,
  a sha256 mismatch, a missing `Content-Length`.
- `store.rs`: the v10 → v11 migration, insert, list, re-download as an upsert, delete
  with the file already missing.
- `app_flow.rs`: `/pull` and `/models` emit the expected `Effect`s, `PullEvent` updates
  the status bar, `Esc` closes an open overlay first and cancels the download only when
  none is open.

Each task ends with `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and
`cargo test` passing.

## Task breakdown

| | Task | Depends on |
|---|---|---|
| 1 | `models/mod.rs` skeleton (`ModelError`, `ModelsConfig`, module declarations) + `gguf.rs` | — |
| 2 | `hub.rs` + `download.rs`: listing, resumable download, the job | 1 |
| 3 | `store.rs` + the v11 migration | 1 |
| 4 | Configuration, commands, state, runtime wiring | 1, 2, 3 |
| 5 | The two views, the status bar, `README` / `roadmap` / `PLAN` | 4 |

Task 1 owns `src/models/mod.rs`, so it lands first: were tasks 2 and 3 each to add their own
`pub mod` line, the two edits would fall in the same region and conflict. With the module
declared against compiling stubs, tasks 2 and 3 share no file and run at once. One commit
for the milestone.
