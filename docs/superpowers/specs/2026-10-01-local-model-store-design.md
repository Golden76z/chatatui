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

Completion: `sync_all`, then an atomic rename into
`<store>/models/{owner}/{repo}/{file}`. The real directory nesting avoids the name
collisions a flattened `owner__repo` scheme would allow.

### Verification

sha256 is computed while writing. At rename time it is compared with `lfs.oid`. When the
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
CREATE UNIQUE INDEX local_models_file ON local_models (repo, revision, file);
```

`sha256` is `NULL` when the API exposed no oid. The unique index makes re-downloading an
idempotent upsert, matching how messages are written.

Deleting a model removes the file and the row; a file already gone is not an error, the
row goes anyway.

## UI

- `/pull <repo> [file]` — without a file, a picker lists the repository's `.gguf` files
  (name, quantization, size) and the chosen one is downloaded; with a file, it is
  downloaded directly.
- `/models` — an inventory overlay: repository, quantization, size, architecture,
  context window, and whether the checksum was verified. `Suppr` deletes after a
  confirmation.
- While a download runs the status bar shows `⬇ Qwen2.5-7B Q4_K_M 2,1/4,4 Go · 18 Mo/s`
  and `Esc` cancels — the slot indexing already uses, so nothing new is invented.
- Two new `Overlay` variants, one shown at a time like every other popup, and `Esc`
  closes the topmost one before cancelling a download.
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
| 1 | `gguf.rs`: parsing and its tests | — |
| 2 | `hub.rs` + `download.rs`: listing, resumable download, the job | — |
| 3 | `store.rs` + the v11 migration | — |
| 4 | UI, commands, runtime wiring, `README` and `docs/roadmap.md` | 1, 2, 3 |

Tasks 1, 2 and 3 touch disjoint files and run in parallel; task 4 follows. One commit
for the milestone.
