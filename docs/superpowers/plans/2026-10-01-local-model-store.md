# J32 — Local model store: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download GGUF models from HuggingFace into a local store, inspect their metadata, and list and delete them from the TUI — without any inference.

**Architecture:** A new `src/models/` module shaped exactly like `src/rag/`: `hub.rs` is the HuggingFace HTTP client, `gguf.rs` parses the file header, `store.rs` keeps the SQLite inventory, and `download.rs` is a cancellable background job that reports `PullEvent`s. Control flow uses the existing `Effect` → tokio task → `AppEvent` loop, so `App::update` stays pure.

**Tech Stack:** Rust 2024 (1.88+), tokio, reqwest (`stream`), rusqlite (bundled), ratatui 0.30, thiserror, and one new dependency: `sha2`.

**Spec:** `docs/superpowers/specs/2026-10-01-local-model-store-design.md`

## Execution Order

```
Task 1 (skeleton + gguf)
   ├── Task 2 (hub + download)  ┐
   └── Task 3 (store + v11)     ┘ in parallel, disjoint files
                                 └── Task 4 (config, commands, state, runtime)
                                        └── Task 5 (views, status bar, docs)
```

Task 1 owns `src/models/mod.rs` and must land before 2 and 3 start. Tasks 2 and 3 share no
file and can run at once. Tasks 4 and 5 are serial.

## Global Constraints

- Rust edition 2024, `rust-version = "1.88"`. No `unwrap()` or `expect()` outside `#[cfg(test)]` code.
- UI strings in French; code, comments and doc comments in English.
- Every task ends with `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test` all passing. Show the output before claiming a task is done.
- **One commit for the whole milestone.** Do not commit per task — the per-task steps below end at the three checks. The orchestrator squashes the branch into a single `feat(modeles): …` commit at the end, per the repo's convention.
- One new dependency only: `sha2`. Do not add `hf-hub`, `tokenizers`, `candle`, or an HTTP server crate.
- `src/storage/schema.rs` migrations are append-only: add v11 at the end of `MIGRATIONS`, never edit a released entry.
- `Config` derives `#[serde(deny_unknown_fields)]`, so a `[models]` section does not parse until `Config` declares the field.
- No GGUF binary fixtures in the repository: tests build their bytes in code.
- Secrets never reach `App` and never appear in a `Debug` output, as with provider API keys.

## Review Focus

These are the input classes the spec implies but does not enumerate. Each line's test is added to the task that owns the code.

1. **A pasted URL instead of `owner/name`** — `/pull https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF` is what a person actually pastes. It must resolve to the repo, not build a malformed API path. (Task 2)
2. **A repository with no `.gguf` file at all** — pointing `/pull` at a safetensors-only repo is a common mistake and must say so, not open an empty picker. (Task 2)
3. **A remote file path that escapes the store** — the file name comes from a remote API, so `../../.bashrc`, `/etc/passwd` and `sub/dir/model.gguf` must be refused before being joined onto the store directory. (Task 2)
4. **A write failure mid-download** (disk full, read-only directory) — must surface a readable French message and leave no row claiming a complete model. (Task 2)
5. **A corrupt or hostile GGUF header** — a declared `metadata_kv_count` of 2^60, a string length past the end of the file, or a truncated file must error out in bounded memory rather than allocate wildly or panic. (Task 1)

---

### Task 1: module skeleton and GGUF metadata parsing

**Files:**
- Create: `src/models/mod.rs` — **this task owns it**; no later task edits it
- Create: `src/models/gguf.rs`
- Create: `src/models/hub.rs`, `src/models/download.rs`, `src/models/store.rs` — doc-comment stubs only, filled in by Tasks 2 and 3
- Modify: `src/lib.rs` (add `pub mod models;`)

**Why the stubs:** Tasks 2 and 3 run in parallel. If each added its own `pub mod` line to
`src/models/mod.rs`, the two edits would land in the same three-line region and conflict on
merge. Declaring all four modules here, against stub files that compile, gives the parallel
pair genuinely disjoint files.

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `models::ModelError` — the module's error enum, used by every other task.
  - `models::gguf::Metadata { architecture: Option<String>, name: Option<String>, quantization: Option<String>, context_length: Option<u64>, block_count: Option<u64>, head_count: Option<u64>, head_count_kv: Option<u64>, embedding_length: Option<u64>, parameters: Option<u64>, tokenizer: Option<String> }`
  - `models::gguf::parse<R: Read + Seek>(reader: &mut R) -> Result<Metadata, ModelError>`
  - `models::gguf::read(path: &Path) -> Result<Metadata, ModelError>`

**Background the implementer needs:**

A GGUF file starts with a little-endian header: the ASCII magic `GGUF`, a `u32` version (3 current, 2 still in the wild), a `u64` tensor count, and a `u64` metadata key/value count. Then come the key/value pairs: each is a string (a `u64` byte length followed by UTF-8 bytes), a `u32` value type, and the value. Then one descriptor per tensor: the name as a string, a `u32` dimension count, that many `u64` dimensions, a `u32` ggml type, and a `u64` offset.

Value types are `0..=12`: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `bool`, `string`, `array`, `u64`, `i64`, `f64`. An array is a `u32` element type, a `u64` count, then the elements.

Only a handful of keys matter here, and the vocabulary arrays (`tokenizer.ggml.tokens`, `.merges`) are **skipped by seeking past them**, never read into memory — that is why the parser takes `Read + Seek` rather than a byte slice. Tests drive it with `std::io::Cursor`, which implements both.

- [ ] **Step 1: Create the module skeleton with the error type**

Create `src/models/mod.rs`, declaring all four submodules and the config section up front:

```rust
//! Local model store: GGUF files downloaded from HuggingFace, kept on disk with an
//! inventory in SQLite.
//!
//! - [`hub`]: the HuggingFace client (list a repository's GGUF files, download one);
//! - [`gguf`]: the metadata in a GGUF file's header;
//! - [`store`]: the inventory of downloaded models;
//! - [`download`]: the background job behind `/pull`.
//!
//! Running a model is not part of this module yet.

pub mod download;
pub mod gguf;
pub mod hub;
pub mod store;

/// `[models]` section of the configuration.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    /// Where models are kept; empty means `<data>/models`.
    pub dir: String,
    /// Environment variable holding a HuggingFace token, for gated repositories.
    pub token_env: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            dir: String::new(),
            token_env: "HF_TOKEN".to_owned(),
        }
    }
}

/// Why a model operation failed. Every message is shown to the user as written.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelError {
    #[error("{0}")]
    Http(String),
    #[error("dépôt ou fichier introuvable")]
    NotFound,
    #[error("fichier GGUF illisible : {0}")]
    Gguf(String),
    #[error("{0}")]
    Io(String),
    #[error("le fichier téléchargé est corrompu (empreinte sha256 incorrecte)")]
    Checksum,
}
```

Then create the three stub files so the crate compiles, each holding only its module doc
comment — Tasks 2 and 3 fill them in:

```rust
// src/models/hub.rs
//! The HuggingFace client: list a repository's GGUF files, and download one with resume.
```

```rust
// src/models/download.rs
//! The background job behind `/pull`.
```

```rust
// src/models/store.rs
//! The inventory of downloaded models (table `local_models`, schema v11).
```

Add `pub mod models;` to `src/lib.rs`, in the existing alphabetical position among the module declarations.

Cancellation is **not** a `ModelError` variant: the job reports it separately, the way `rag::indexer::run` does with `Err(None)`.

Run `cargo build` before moving on: the skeleton must compile with the stubs in place.

- [ ] **Step 2: Write the failing test for a minimal v3 file**

Create `src/models/gguf.rs` with only the test module at first, so the test genuinely fails to compile against a missing function:

```rust
#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// Builds a GGUF file: `kv` are `(key, type, encoded value)`, `tensors` are
    /// `(name, dims, ggml_type)`.
    fn gguf(
        version: u32,
        kv: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, Vec<u64>, u32)],
    ) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(version.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((kv.len() as u64).to_le_bytes());
        for (key, kind, value) in kv {
            out.extend(string(key));
            out.extend(kind.to_le_bytes());
            out.extend(value);
        }
        for (name, dims, kind) in tensors {
            out.extend(string(name));
            out.extend((dims.len() as u32).to_le_bytes());
            for dim in dims {
                out.extend(dim.to_le_bytes());
            }
            out.extend(kind.to_le_bytes());
            out.extend(0u64.to_le_bytes());
        }
        out
    }

    fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend(text.as_bytes());
        out
    }

    fn u32_value(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn string_array(items: &[&str]) -> Vec<u8> {
        let mut out = 8u32.to_le_bytes().to_vec();
        out.extend((items.len() as u64).to_le_bytes());
        for item in items {
            out.extend(string(item));
        }
        out
    }

    #[test]
    fn reads_the_architecture_name_and_dimensions() {
        let bytes = gguf(
            3,
            &[
                ("general.architecture", 8, string("llama")),
                ("general.name", 8, string("Qwen2.5 7B Instruct")),
                ("general.file_type", 4, u32_value(15)),
                ("llama.context_length", 4, u32_value(32768)),
                ("llama.block_count", 4, u32_value(28)),
                ("llama.attention.head_count", 4, u32_value(28)),
                ("llama.attention.head_count_kv", 4, u32_value(4)),
                ("llama.embedding_length", 4, u32_value(3584)),
                ("tokenizer.ggml.model", 8, string("gpt2")),
            ],
            &[("token_embd.weight", vec![3584, 152064], 12)],
        );

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.name.as_deref(), Some("Qwen2.5 7B Instruct"));
        assert_eq!(meta.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(meta.context_length, Some(32768));
        assert_eq!(meta.block_count, Some(28));
        assert_eq!(meta.head_count, Some(28));
        assert_eq!(meta.head_count_kv, Some(4));
        assert_eq!(meta.embedding_length, Some(3584));
        assert_eq!(meta.parameters, Some(3584 * 152064));
        assert_eq!(meta.tokenizer.as_deref(), Some("gpt2"));
    }
}
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cargo test --lib models::gguf`
Expected: a compile error — `cannot find function 'parse' in this scope`.

- [ ] **Step 4: Implement the parser**

Write the rest of `src/models/gguf.rs` above the test module:

```rust
//! Metadata in a GGUF file's header.
//!
//! The header is little-endian: the `GGUF` magic, a `u32` version, a `u64` tensor count,
//! a `u64` metadata count, the typed key/value pairs, then one descriptor per tensor.
//! Tensor *data* is never read — the parser only seeks over it — so inspecting a model of
//! several gigabytes costs a few reads.
//!
//! Every field is optional: a key this build does not know about is skipped, and a key it
//! wants but does not find leaves `None`. Lengths declared by the file are checked against
//! [`MAX_COUNT`], [`MAX_STRING`] and [`MAX_ARRAY`] before anything is allocated, so a
//! corrupt or hostile header fails instead of exhausting memory.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use super::ModelError;

/// Largest metadata or tensor count accepted (real files are in the hundreds).
const MAX_COUNT: u64 = 1 << 20;
/// Longest string accepted, in bytes.
const MAX_STRING: u64 = 1 << 20;
/// Most elements accepted in an array (a 256k vocabulary fits).
const MAX_ARRAY: u64 = 1 << 24;

/// What a GGUF file's header says about the model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metadata {
    /// `general.architecture`: `llama`, `qwen2`, `gemma3`, …
    pub architecture: Option<String>,
    pub name: Option<String>,
    /// Quantization as users name it (`Q4_K_M`), from `general.file_type`, falling back to
    /// the most common tensor type.
    pub quantization: Option<String>,
    /// Training context window, `{arch}.context_length`.
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub embedding_length: Option<u64>,
    /// Parameters, summed from the tensor dimensions.
    pub parameters: Option<u64>,
    /// `tokenizer.ggml.model`: `gpt2`, `llama`, `spm`. The vocabulary itself is skipped.
    pub tokenizer: Option<String>,
}

/// Reads the header of the GGUF file at `path`.
pub fn read(path: &Path) -> Result<Metadata, ModelError> {
    let file = File::open(path).map_err(|e| ModelError::Io(e.to_string()))?;
    parse(&mut BufReader::new(file))
}

/// Reads a GGUF header from `reader`.
pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Metadata, ModelError> {
    let mut magic = [0u8; 4];
    read_exact(reader, &mut magic)?;
    if &magic != b"GGUF" {
        return Err(ModelError::Gguf("ce n'est pas un fichier GGUF".into()));
    }
    let version = read_u32(reader)?;
    if !(2..=3).contains(&version) {
        return Err(ModelError::Gguf(format!("version {version} non gérée")));
    }
    let tensor_count = bounded(read_u64(reader)?, MAX_COUNT, "nombre de tenseurs")?;
    let kv_count = bounded(read_u64(reader)?, MAX_COUNT, "nombre de métadonnées")?;

    let mut meta = Metadata::default();
    let mut file_type = None;
    // Keys are `{arch}.…`, and the architecture may arrive after them, so keep the
    // suffixes and resolve them once everything is read.
    let mut suffixed: Vec<(String, u64)> = Vec::new();
    for _ in 0..kv_count {
        let key = read_string(reader)?;
        let kind = read_u32(reader)?;
        match (key.as_str(), kind) {
            ("general.architecture", 8) => meta.architecture = Some(read_string(reader)?),
            ("general.name", 8) => meta.name = Some(read_string(reader)?),
            ("tokenizer.ggml.model", 8) => meta.tokenizer = Some(read_string(reader)?),
            // Skip the value when it is not an integer, or the stream desyncs and every
            // key after this one is misread.
            ("general.file_type", _) => match read_unsigned(reader, kind)? {
                Some(value) => file_type = Some(value),
                None => skip_value(reader, kind)?,
            },
            _ => match read_unsigned(reader, kind)? {
                Some(value) => suffixed.push((key, value)),
                None => skip_value(reader, kind)?,
            },
        }
    }

    let mut parameters: u64 = 0;
    let mut types: Vec<u32> = Vec::new();
    for _ in 0..tensor_count {
        let _name = read_string(reader)?;
        let dimensions = read_u32(reader)?;
        if u64::from(dimensions) > MAX_COUNT {
            return Err(ModelError::Gguf("descripteur de tenseur invalide".into()));
        }
        let mut elements: u64 = 1;
        for _ in 0..dimensions {
            elements = elements.saturating_mul(read_u64(reader)?);
        }
        types.push(read_u32(reader)?);
        let _offset = read_u64(reader)?;
        parameters = parameters.saturating_add(elements);
    }

    if let Some(arch) = &meta.architecture {
        let get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            suffixed.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
        };
        meta.context_length = get("context_length");
        meta.block_count = get("block_count");
        meta.head_count = get("attention.head_count");
        meta.head_count_kv = get("attention.head_count_kv");
        meta.embedding_length = get("embedding_length");
    }
    meta.parameters = (parameters > 0).then_some(parameters);
    meta.quantization = file_type
        .and_then(file_type_name)
        .or_else(|| dominant_type(&types).and_then(ggml_type_name))
        .map(str::to_owned);
    Ok(meta)
}

/// `general.file_type`, as llama.cpp names the quantizations users ask for.
fn file_type_name(value: u64) -> Option<&'static str> {
    Some(match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        32 => "BF16",
        _ => return None,
    })
}

/// A tensor's ggml type, for files whose `general.file_type` is missing or unknown.
fn ggml_type_name(value: u32) -> Option<&'static str> {
    Some(match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        30 => "BF16",
        _ => return None,
    })
}

/// The type most tensors use, ignoring the F32 norms every quantized file keeps.
fn dominant_type(types: &[u32]) -> Option<u32> {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for kind in types.iter().copied().filter(|k| *k != 0) {
        match counts.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, count)) => *count += 1,
            None => counts.push((kind, 1)),
        }
    }
    counts.into_iter().max_by_key(|(_, count)| *count).map(|(kind, _)| kind)
}

/// Reads an integer or boolean value as a `u64`; `None` for the other types, whose bytes
/// the caller must skip.
fn read_unsigned<R: Read + Seek>(reader: &mut R, kind: u32) -> Result<Option<u64>, ModelError> {
    let value = match kind {
        0 | 7 => u64::from(read_n::<R, 1>(reader)?[0]),
        1 => {
            let byte = read_n::<R, 1>(reader)?[0] as i8;
            u64::try_from(byte).unwrap_or(0)
        }
        2 => u64::from(u16::from_le_bytes(read_n(reader)?)),
        3 => u64::try_from(i16::from_le_bytes(read_n(reader)?)).unwrap_or(0),
        4 => u64::from(read_u32(reader)?),
        5 => u64::try_from(i32::from_le_bytes(read_n(reader)?)).unwrap_or(0),
        10 => read_u64(reader)?,
        11 => u64::try_from(i64::from_le_bytes(read_n(reader)?)).unwrap_or(0),
        _ => return Ok(None),
    };
    Ok(Some(value))
}

/// Moves past a value this parser does not want: a string, a float, or a whole array.
fn skip_value<R: Read + Seek>(reader: &mut R, kind: u32) -> Result<(), ModelError> {
    match kind {
        6 => skip(reader, 4),
        12 => skip(reader, 8),
        8 => {
            let length = bounded(read_u64(reader)?, MAX_STRING, "longueur de chaîne")?;
            skip(reader, length)
        }
        9 => {
            let element = read_u32(reader)?;
            let count = bounded(read_u64(reader)?, MAX_ARRAY, "taille de tableau")?;
            match fixed_width(element) {
                // Fixed-width elements: one seek over the lot.
                Some(width) => skip(reader, count.saturating_mul(width)),
                // Strings are length-prefixed, so each one has to be stepped over.
                None if element == 8 => {
                    for _ in 0..count {
                        let length = bounded(read_u64(reader)?, MAX_STRING, "longueur de chaîne")?;
                        skip(reader, length)?;
                    }
                    Ok(())
                }
                None => Err(ModelError::Gguf(format!(
                    "type de tableau inconnu ({element})"
                ))),
            }
        }
        _ => Err(ModelError::Gguf(format!("type inconnu ({kind})"))),
    }
}

/// Bytes one value of this type takes, when it is fixed.
fn fixed_width(kind: u32) -> Option<u64> {
    Some(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        _ => return None,
    })
}

fn bounded(value: u64, limit: u64, what: &str) -> Result<u64, ModelError> {
    if value > limit {
        return Err(ModelError::Gguf(format!("{what} invalide ({value})")));
    }
    Ok(value)
}

fn skip<R: Read + Seek>(reader: &mut R, bytes: u64) -> Result<(), ModelError> {
    let offset = i64::try_from(bytes)
        .map_err(|_| ModelError::Gguf("décalage invalide".into()))?;
    reader
        .seek(SeekFrom::Current(offset))
        .map(|_| ())
        .map_err(|e| ModelError::Gguf(e.to_string()))
}

fn read_exact<R: Read>(reader: &mut R, buffer: &mut [u8]) -> Result<(), ModelError> {
    reader
        .read_exact(buffer)
        .map_err(|e| ModelError::Gguf(format!("fichier tronqué ({e})")))
}

fn read_n<R: Read, const N: usize>(reader: &mut R) -> Result<[u8; N], ModelError> {
    let mut buffer = [0u8; N];
    read_exact(reader, &mut buffer)?;
    Ok(buffer)
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, ModelError> {
    Ok(u32::from_le_bytes(read_n(reader)?))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, ModelError> {
    Ok(u64::from_le_bytes(read_n(reader)?))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, ModelError> {
    let length = bounded(read_u64(reader)?, MAX_STRING, "longueur de chaîne")?;
    let mut bytes = vec![0u8; usize::try_from(length).unwrap_or(0)];
    read_exact(reader, &mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `cargo test --lib models::gguf`
Expected: PASS.

- [ ] **Step 6: Write the remaining tests, including the Review Focus case**

Append to the `tests` module in `src/models/gguf.rs`:

```rust
    #[test]
    fn accepts_version_2_and_skips_the_vocabulary() {
        let bytes = gguf(
            2,
            &[
                ("general.architecture", 8, string("qwen2")),
                ("tokenizer.ggml.tokens", 9, string_array(&["<s>", "hello", "world"])),
                ("qwen2.context_length", 10, 4096u64.to_le_bytes().to_vec()),
            ],
            &[("blk.0.attn_q.weight", vec![64, 64], 8)],
        );

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        assert_eq!(meta.architecture.as_deref(), Some("qwen2"));
        assert_eq!(meta.context_length, Some(4096));
        // The vocabulary is stepped over, so the tensor after it still parses.
        assert_eq!(meta.parameters, Some(4096));
        assert_eq!(meta.quantization.as_deref(), Some("Q8_0"));
    }

    #[test]
    fn a_missing_key_leaves_none_rather_than_failing() {
        let bytes = gguf(3, &[("general.architecture", 8, string("llama"))], &[]);

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.context_length, None);
        assert_eq!(meta.name, None);
        assert_eq!(meta.parameters, None);
    }

    #[test]
    fn keys_of_another_architecture_are_ignored() {
        let bytes = gguf(
            3,
            &[
                ("general.architecture", 8, string("llama")),
                ("gemma3.context_length", 4, u32_value(8192)),
            ],
            &[],
        );

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        assert_eq!(meta.context_length, None);
    }

    #[test]
    fn rejects_a_file_that_is_not_gguf() {
        let error = parse(&mut Cursor::new(b"ZIP\0rest".to_vec())).expect_err("rejected");

        assert!(matches!(error, ModelError::Gguf(_)), "{error:?}");
    }

    #[test]
    fn rejects_a_truncated_file() {
        let mut bytes = gguf(3, &[("general.architecture", 8, string("llama"))], &[]);
        bytes.truncate(bytes.len() - 3);

        let error = parse(&mut Cursor::new(bytes)).expect_err("rejected");

        assert!(matches!(error, ModelError::Gguf(_)), "{error:?}");
    }

    /// Review Focus 5: a hostile header must fail in bounded memory, not allocate 2^60.
    #[test]
    fn rejects_absurd_declared_counts_without_allocating() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend((1u64 << 60).to_le_bytes());

        let error = parse(&mut Cursor::new(bytes)).expect_err("rejected");

        assert!(matches!(error, ModelError::Gguf(_)), "{error:?}");
    }

    /// Review Focus 5: a string whose declared length is absurd must be refused too.
    #[test]
    fn rejects_an_absurd_string_length() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend((u64::MAX / 2).to_le_bytes());

        let error = parse(&mut Cursor::new(bytes)).expect_err("rejected");

        assert!(matches!(error, ModelError::Gguf(_)), "{error:?}");
    }

    #[test]
    fn rejects_an_unknown_value_type() {
        let bytes = gguf(3, &[("general.quantization_version", 99, vec![0u8; 4])], &[]);

        let error = parse(&mut Cursor::new(bytes)).expect_err("rejected");

        assert!(matches!(error, ModelError::Gguf(_)), "{error:?}");
    }

    #[test]
    fn falls_back_to_the_dominant_tensor_type_without_a_file_type() {
        let bytes = gguf(
            3,
            &[("general.architecture", 8, string("llama"))],
            &[
                ("blk.0.attn_norm.weight", vec![64], 0),
                ("blk.0.attn_q.weight", vec![64, 64], 14),
                ("blk.0.attn_k.weight", vec![64, 64], 14),
            ],
        );

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        // The F32 norm does not outvote the quantized weights.
        assert_eq!(meta.quantization.as_deref(), Some("Q6_K"));
    }
```

- [ ] **Step 7: Run the whole task's tests**

Run: `cargo test --lib models::gguf`
Expected: 10 tests, all PASS.

- [ ] **Step 8: Checks**

Run, and show the output:

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

---

### Task 2: the HuggingFace client and the download job

**Files:**
- Create: `src/models/hub.rs`
- Create: `src/models/download.rs`
- Create: `tests/model_hub.rs`
- Modify: `Cargo.toml` (add `sha2 = "0.10"`)

Task 1 already declared `hub` and `download` in `src/models/mod.rs` and left them as stubs,
so **do not edit `src/models/mod.rs`** — Task 3 is running in parallel against the same file.

**Depends on Task 1** (the module skeleton, `ModelError` and `gguf::read`). Runs in parallel with Task 3.

**Interfaces:**
- Consumes: `models::ModelError` and `models::gguf::read` from Task 1.
- Produces:
  - `models::hub::RemoteFile { path: String, bytes: u64, sha256: Option<String> }`
  - `models::hub::parse_target(arg: &str) -> Result<(String, Option<String>), ModelError>`
  - `models::hub::safe_file_name(path: &str) -> Result<&str, ModelError>`
  - `models::hub::Hub::new(base: Option<String>, token: Option<String>, connect_timeout: Duration) -> Result<Hub, ModelError>`
  - `models::hub::Hub::list_gguf(&self, repo: &str, revision: &str) -> Result<Vec<RemoteFile>, ModelError>`
  - `models::hub::Hub::fetch(&self, repo, revision, file: &RemoteFile, part: &Path, cancel: &CancellationToken, progress: impl Fn(u64, Option<u64>) + Send) -> Result<String, ModelError>`
  - `models::download::{PullRequest, PullEvent}`, `models::download::paths(...)` and `models::download::run(...)`
- Note for the implementer: `download::run` writes the inventory row through `models::store::save` from Task 3. Task 3 defines `store::save(conn, &LocalModel) -> Result<(), StoreError>` and the `LocalModel` struct; depend on those exact names.

**Background the implementer needs:**

`GET https://huggingface.co/api/models/{repo}/tree/{revision}` answers a JSON array. Each entry has `type` (`file` or `directory`), `path`, `size`, and for a large file an `lfs` object carrying `oid` (the sha256, sometimes prefixed `sha256:`) and `size`. The real byte length of an LFS file is `lfs.size`; the top-level `size` is the pointer's size for some entries, so prefer `lfs.size` when present.

`GET https://huggingface.co/{repo}/resolve/{revision}/{path}` redirects to a CDN; `reqwest` follows redirects by default. A `Range: bytes=N-` request answers `206 Partial Content` when honoured and `200 OK` when ignored — a `200` means the body starts from byte zero, so the part file must be truncated rather than appended to.

The digest covers the whole file. After a resume, the bytes already on disk must be fed through the hasher before the new ones, or the comparison is meaningless.

- [ ] **Step 1: Add the dependency**

In `Cargo.toml`, under `[dependencies]`, in alphabetical position:

```toml
sha2 = "0.10"
```

Run: `cargo build` — expected: compiles, `Cargo.lock` updated.

- [ ] **Step 2: Write the failing unit tests for argument parsing and path safety**

Create `src/models/hub.rs` containing only this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_plain_repository() {
        let (repo, file) = parse_target("bartowski/Qwen2.5-7B-Instruct-GGUF").expect("parsed");

        assert_eq!(repo, "bartowski/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(file, None);
    }

    /// Review Focus 1: a pasted URL is what people actually type.
    #[test]
    fn accepts_a_pasted_url() {
        for input in [
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF",
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/tree/main",
            "huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/",
            "hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF",
        ] {
            let (repo, file) = parse_target(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!(repo, "bartowski/Qwen2.5-7B-Instruct-GGUF", "{input}");
            assert_eq!(file, None, "{input}");
        }
    }

    /// Review Focus 1: a blob URL names the file, so use it.
    #[test]
    fn a_blob_url_also_names_the_file() {
        let (repo, file) = parse_target(
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        )
        .expect("parsed");

        assert_eq!(repo, "bartowski/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(file.as_deref(), Some("Qwen2.5-7B-Instruct-Q4_K_M.gguf"));
    }

    #[test]
    fn rejects_something_that_is_not_a_repository() {
        for input in ["", "llama3.2", "a/b/c/d/e", "/", "owner/"] {
            assert!(parse_target(input).is_err(), "{input} should be refused");
        }
    }

    /// Review Focus 3: the file name comes from a remote API.
    #[test]
    fn refuses_a_file_name_that_escapes_the_store() {
        for path in [
            "../../.bashrc",
            "/etc/passwd",
            "sub/dir/model.gguf",
            "..",
            ".",
            "",
            "C:\\windows\\system32\\x.gguf",
            "dir\\model.gguf",
        ] {
            assert!(safe_file_name(path).is_err(), "{path} should be refused");
        }
    }

    #[test]
    fn accepts_a_bare_file_name() {
        assert_eq!(
            safe_file_name("Qwen2.5-7B-Instruct-Q4_K_M.gguf").expect("accepted"),
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        );
    }

    #[test]
    fn keeps_only_gguf_entries_and_prefers_the_lfs_size() {
        let body = r#"[
            {"type":"directory","path":"sub"},
            {"type":"file","path":"README.md","size":1200},
            {"type":"file","path":"model.safetensors","size":99},
            {"type":"file","path":"m-Q4_K_M.gguf","size":135,
             "lfs":{"oid":"sha256:abc123","size":4431401088}},
            {"type":"file","path":"m-Q8_0.gguf","size":7200000000}
        ]"#;

        let files = parse_tree(body).expect("parsed");

        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "m-Q4_K_M.gguf");
        assert_eq!(files[0].bytes, 4_431_401_088);
        assert_eq!(files[0].sha256.as_deref(), Some("abc123"));
        assert_eq!(files[1].path, "m-Q8_0.gguf");
        assert_eq!(files[1].bytes, 7_200_000_000);
        assert_eq!(files[1].sha256, None);
    }
}
```

- [ ] **Step 3: Run them to confirm they fail**

Run: `cargo test --lib models::hub`
Expected: compile errors — `parse_target`, `safe_file_name` and `parse_tree` not found.

- [ ] **Step 4: Implement `hub.rs`**

Replace the stub `src/models/hub.rs` with this, keeping the test module at the bottom:

```rust
//! The HuggingFace client: list a repository's GGUF files, and download one with resume.
//!
//! Only the public HTTP API is used — `GET /api/models/{repo}/tree/{revision}` to list and
//! `GET /{repo}/resolve/{revision}/{file}` to fetch — so nothing here depends on Ollama or
//! on a Hub SDK.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

use super::ModelError;

/// Public Hub, overridden in tests.
const DEFAULT_BASE: &str = "https://huggingface.co";
/// Bytes read at a time when re-hashing a part file on resume.
const REHASH_CHUNK: usize = 1 << 20;
/// Shortest gap between two progress reports.
const PROGRESS_EVERY: Duration = Duration::from_millis(100);

/// A GGUF file a repository offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    /// Name inside the repository; always a bare file name (see [`safe_file_name`]).
    pub path: String,
    pub bytes: u64,
    /// sha256 of the LFS object, when the API exposes one.
    pub sha256: Option<String>,
}

/// Splits what the user typed into a repository and, when the URL names one, a file.
///
/// Accepts `owner/name`, `hf.co/owner/name`, and a `huggingface.co` URL with or without a
/// `/tree/<rev>` or `/blob/<rev>/<file>` tail.
pub fn parse_target(arg: &str) -> Result<(String, Option<String>), ModelError> {
    let trimmed = arg.trim();
    let without_scheme = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);
    let path = ["huggingface.co/", "hf.co/"]
        .iter()
        .find_map(|host| without_scheme.strip_prefix(host))
        .unwrap_or(without_scheme);
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let [owner, name, tail @ ..] = parts.as_slice() else {
        return Err(ModelError::Http(
            "donnez un dépôt : /pull <propriétaire>/<nom> [fichier]".into(),
        ));
    };
    let file = match tail {
        ["blob" | "resolve", _revision, file] => Some((*file).to_owned()),
        [] | ["tree", _revision] | ["tree"] => None,
        _ => {
            return Err(ModelError::Http(
                "lien HuggingFace non reconnu : /pull <propriétaire>/<nom> [fichier]".into(),
            ));
        }
    };
    Ok((format!("{owner}/{name}"), file))
}

/// Accepts a bare file name only. The name comes from a remote API, so anything that could
/// escape the store directory is refused — and so are the multi-part GGUF files that live in
/// subdirectories, which this version cannot assemble.
pub fn safe_file_name(path: &str) -> Result<&str, ModelError> {
    if path.is_empty() || path == "." || path == ".." {
        return Err(ModelError::Http("nom de fichier invalide".into()));
    }
    if path.contains('/') || path.contains('\\') {
        return Err(ModelError::Http(
            "les modèles en plusieurs fichiers ne sont pas gérés (un seul .gguf à la fois)".into(),
        ));
    }
    Ok(path)
}

/// Reads the `.gguf` entries out of a `tree` answer.
fn parse_tree(body: &str) -> Result<Vec<RemoteFile>, ModelError> {
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default, rename = "type")]
        kind: String,
        path: String,
        #[serde(default)]
        size: u64,
        #[serde(default)]
        lfs: Option<Lfs>,
    }
    #[derive(Deserialize)]
    struct Lfs {
        #[serde(default)]
        oid: String,
        #[serde(default)]
        size: u64,
    }

    let entries: Vec<Entry> =
        serde_json::from_str(body).map_err(|e| ModelError::Http(e.to_string()))?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind != "directory" && e.path.to_lowercase().ends_with(".gguf"))
        .map(|e| {
            let lfs = e.lfs.filter(|l| l.size > 0);
            RemoteFile {
                bytes: lfs.as_ref().map_or(e.size, |l| l.size),
                sha256: lfs
                    .as_ref()
                    .map(|l| l.oid.trim_start_matches("sha256:").to_owned())
                    .filter(|oid| !oid.is_empty()),
                path: e.path,
            }
        })
        .collect())
}

/// HTTP client for one Hub.
#[derive(Clone)]
pub struct Hub {
    http: Client,
    base: String,
    token: Option<String>,
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the token.
        f.debug_struct("Hub")
            .field("base", &self.base)
            .field("token", &self.token.is_some())
            .finish()
    }
}

impl Hub {
    /// `base` overrides the public Hub (tests point it at a local server).
    pub fn new(
        base: Option<String>,
        token: Option<String>,
        connect_timeout: Duration,
    ) -> Result<Self, ModelError> {
        let http = Client::builder()
            .connect_timeout(connect_timeout)
            .user_agent(concat!("chatatui/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ModelError::Http(e.to_string()))?;
        Ok(Self {
            http,
            base: base
                .map(|b| b.trim_end_matches('/').to_owned())
                .unwrap_or_else(|| DEFAULT_BASE.to_owned()),
            token: token.filter(|t| !t.trim().is_empty()),
        })
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// The repository's GGUF files, largest name first as the API returns them.
    pub async fn list_gguf(
        &self,
        repo: &str,
        revision: &str,
    ) -> Result<Vec<RemoteFile>, ModelError> {
        let url = format!("{}/api/models/{repo}/tree/{revision}", self.base);
        let response = self
            .authorize(self.http.get(url))
            .send()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        match response.status() {
            StatusCode::OK => {}
            StatusCode::NOT_FOUND => return Err(ModelError::NotFound),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Err(ModelError::Http(
                    "dépôt restreint : renseignez un jeton dans [models] token_env".into(),
                ));
            }
            status => return Err(ModelError::Http(format!("HTTP {status}"))),
        }
        let body = response
            .text()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        let mut files = parse_tree(&body)?;
        files.retain(|f| safe_file_name(&f.path).is_ok());
        Ok(files)
    }

    /// Streams `file` into `part`, resuming when `part` already holds bytes, and returns the
    /// hex sha256 of the whole file.
    ///
    /// `progress` receives `(bytes written, total when known)` at most every 100 ms.
    pub async fn fetch(
        &self,
        repo: &str,
        revision: &str,
        file: &RemoteFile,
        part: &Path,
        cancel: &CancellationToken,
        progress: impl Fn(u64, Option<u64>) + Send,
    ) -> Result<String, ModelError> {
        let name = safe_file_name(&file.path)?;
        if let Some(parent) = part.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ModelError::Io(format!("création de {}: {e}", parent.display())))?;
        }
        let already = tokio::fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);
        let url = format!("{}/{repo}/resolve/{revision}/{name}", self.base);
        let mut request = self.authorize(self.http.get(url));
        if already > 0 {
            request = request.header("Range", format!("bytes={already}-"));
        }
        let response = request
            .send()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        let resumed = match response.status() {
            StatusCode::PARTIAL_CONTENT => true,
            // The server ignored the range: the body starts at zero, so start over rather
            // than append and produce a corrupt file.
            StatusCode::OK => false,
            StatusCode::NOT_FOUND => return Err(ModelError::NotFound),
            status => return Err(ModelError::Http(format!("HTTP {status}"))),
        };

        let mut hasher = Sha256::new();
        let mut written = 0;
        let mut handle = if resumed {
            // The digest covers the whole file, so what is already on disk has to go
            // through the hasher before the new bytes.
            written = already;
            rehash(part, &mut hasher).await?;
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(part)
                .await
                .map_err(|e| ModelError::Io(format!("ouverture de {}: {e}", part.display())))?
        } else {
            tokio::fs::File::create(part)
                .await
                .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?
        };

        let total = response
            .content_length()
            .map(|length| written + length)
            .or((file.bytes > 0).then_some(file.bytes));
        let mut stream = response.bytes_stream();
        let mut last = Instant::now();
        progress(written, total);
        loop {
            let chunk = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    // Flush what we have so the next /pull resumes from here.
                    let _ = handle.flush().await;
                    return Err(ModelError::Io("annulé".into()));
                }
                chunk = stream.next() => chunk,
            };
            let Some(chunk) = chunk else { break };
            let chunk = chunk.map_err(|e| ModelError::Http(e.to_string()))?;
            hasher.update(&chunk);
            handle
                .write_all(&chunk)
                .await
                .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?;
            written += chunk.len() as u64;
            if last.elapsed() >= PROGRESS_EVERY {
                progress(written, total);
                last = Instant::now();
            }
        }
        handle
            .flush()
            .await
            .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?;
        handle
            .sync_all()
            .await
            .map_err(|e| ModelError::Io(format!("écriture de {}: {e}", part.display())))?;
        progress(written, total);
        Ok(hex(&hasher.finalize()))
    }
}

/// Feeds an existing part file through `hasher`.
async fn rehash(part: &Path, hasher: &mut Sha256) -> Result<(), ModelError> {
    let mut file = tokio::fs::File::open(part)
        .await
        .map_err(|e| ModelError::Io(format!("lecture de {}: {e}", part.display())))?;
    let mut buffer = vec![0u8; REHASH_CHUNK];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(|e| ModelError::Io(format!("lecture de {}: {e}", part.display())))?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
```

- [ ] **Step 5: Run the unit tests to confirm they pass**

Run: `cargo test --lib models::hub`
Expected: 7 tests PASS.

- [ ] **Step 6: Write the failing job, starting from its events**

Replace the stub `src/models/download.rs` with:

```rust
//! The background job behind `/pull`: fetches one GGUF file, verifies it, reads its
//! header, and records it in the inventory.
//!
//! Shaped like [`crate::rag::indexer::run`]: a `CancellationToken` stops it, progress is
//! reported through a callback the runtime turns into `AppEvent`s, and the final event says
//! whether it finished, failed or was cancelled. The partial file survives a cancellation,
//! so the next run resumes from it.

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use tokio_util::sync::CancellationToken;

use super::{
    ModelError,
    gguf,
    hub::{Hub, RemoteFile},
    store::{self, LocalModel},
};

/// What to download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PullRequest {
    /// `owner/name`.
    pub repo: String,
    pub revision: String,
    pub file: RemoteFile,
    /// Root of the model store.
    pub dir: PathBuf,
}

/// Progress of a download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PullEvent {
    /// `done` bytes written of `total` (unknown until the server says), at `rate` bytes/s.
    Progress {
        repo: String,
        file: String,
        done: u64,
        total: Option<u64>,
        rate: u64,
    },
    /// Downloaded, verified and inspected.
    Finished(Box<LocalModel>),
    Failed {
        repo: String,
        file: String,
        error: String,
    },
    Cancelled {
        repo: String,
        file: String,
    },
}

/// Where a model's file and its in-progress part file live.
///
/// `<dir>/<owner>/<repo>/<file>` keeps the repository's real nesting, so two repositories
/// whose names differ only in where the slash falls cannot collide.
pub fn paths(dir: &Path, repo: &str, file: &str) -> Result<(PathBuf, PathBuf), ModelError> {
    let name = super::hub::safe_file_name(file)?;
    let (owner, repository) = repo
        .split_once('/')
        .ok_or_else(|| ModelError::Http("dépôt invalide".into()))?;
    for part in [owner, repository] {
        if part.is_empty() || part.contains('/') || part.contains('\\') || part.contains("..") {
            return Err(ModelError::Http("dépôt invalide".into()));
        }
    }
    let folder = dir.join(owner).join(repository);
    Ok((folder.join(name), folder.join(format!("{name}.part"))))
}

/// Runs the download, reporting progress and the outcome through `report`.
pub async fn run(
    request: PullRequest,
    db_path: PathBuf,
    hub: std::sync::Arc<Hub>,
    cancel: CancellationToken,
    report: impl Fn(PullEvent) + Send + Sync,
) {
    let repo = request.repo.clone();
    let file = request.file.path.clone();
    let outcome = tokio::select! {
        biased;
        () = cancel.cancelled() => Err(None),
        result = pull(&request, &db_path, hub.as_ref(), &cancel, &report) => result.map_err(Some),
    };
    report(match outcome {
        Ok(model) => PullEvent::Finished(Box::new(model)),
        Err(None) => PullEvent::Cancelled { repo, file },
        Err(Some(error)) => PullEvent::Failed {
            repo,
            file,
            error: error.to_string(),
        },
    });
}

async fn pull(
    request: &PullRequest,
    db_path: &Path,
    hub: &Hub,
    cancel: &CancellationToken,
    report: &(impl Fn(PullEvent) + Send + Sync),
) -> Result<LocalModel, ModelError> {
    let (destination, part) = paths(&request.dir, &request.repo, &request.file.path)?;
    let started = std::time::Instant::now();
    let digest = hub
        .fetch(
            &request.repo,
            &request.revision,
            &request.file,
            &part,
            cancel,
            |done, total| {
                let seconds = started.elapsed().as_secs_f64();
                let rate = if seconds > 0.0 {
                    (done as f64 / seconds) as u64
                } else {
                    0
                };
                report(PullEvent::Progress {
                    repo: request.repo.clone(),
                    file: request.file.path.clone(),
                    done,
                    total,
                    rate,
                });
            },
        )
        .await?;

    // A wrong file is worse than no file.
    if let Some(expected) = &request.file.sha256
        && !expected.eq_ignore_ascii_case(&digest)
    {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(ModelError::Checksum);
    }
    tokio::fs::rename(&part, &destination)
        .await
        .map_err(|e| ModelError::Io(format!("déplacement vers {}: {e}", destination.display())))?;

    let bytes = tokio::fs::metadata(&destination)
        .await
        .map(|m| m.len())
        .unwrap_or(request.file.bytes);
    let inspected = destination.clone();
    let meta = tokio::task::spawn_blocking(move || gguf::read(&inspected))
        .await
        .map_err(|e| ModelError::Io(e.to_string()))??;

    let model = LocalModel {
        repo: request.repo.clone(),
        revision: request.revision.clone(),
        file: request.file.path.clone(),
        path: destination.to_string_lossy().into_owned(),
        bytes,
        // `None` means "not verified", never "verified and fine".
        sha256: request.file.sha256.as_ref().map(|_| digest),
        architecture: meta.architecture,
        quantization: meta.quantization,
        context_length: meta.context_length,
        parameters: meta.parameters,
        downloaded_at: now(),
    };
    let row = model.clone();
    let database = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<(), ModelError> {
        let store = crate::storage::Store::open(&database)
            .map_err(|e| ModelError::Io(e.to_string()))?;
        store::save(&store.into_connection(), &row).map_err(|e| ModelError::Io(e.to_string()))
    })
    .await
    .map_err(|e| ModelError::Io(e.to_string()))??;
    Ok(model)
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}
```


- [ ] **Step 7: Write the integration tests against a local HTTP server**

Create `tests/model_hub.rs`. The server helper follows `tests/http_clients.rs`, which already serves HTTP by hand over a `TcpListener` — read that file first and keep the same shape.

```rust
//! The HuggingFace client and the download job against a minimal local HTTP server.

use std::{path::Path, time::Duration};

use chatatui::models::{
    ModelError,
    hub::{Hub, RemoteFile},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tokio_util::sync::CancellationToken;

/// How a test server answers one request.
enum Reply {
    /// Status line plus headers, then this body.
    Full(&'static str, Vec<u8>),
    /// Headers, then only the first `cut` bytes of the body, then close.
    Cut(&'static str, Vec<u8>, usize),
}

/// Serves `replies` in order, one per connection. Returns the base URL.
async fn serve(replies: Vec<Reply>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move {
        for reply in replies {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_head(&mut socket).await;
            let (head, body) = match &reply {
                Reply::Full(head, body) => ((*head).to_owned(), body.clone()),
                Reply::Cut(head, body, cut) => ((*head).to_owned(), body[..*cut].to_vec()),
            };
            // Range requests are answered from the offset the client asked for.
            let offset = request
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("range: bytes=")
                        .and_then(|v| v.trim().trim_end_matches('-').parse::<usize>().ok())
                })
                .unwrap_or(0);
            let body = body.get(offset.min(body.len())..).unwrap_or_default().to_vec();
            let head = head.replace("{len}", &body.len().to_string());
            socket.write_all(head.as_bytes()).await.expect("head");
            socket.write_all(&body).await.expect("body");
            socket.flush().await.expect("flush");
        }
    });
    base
}

async fn read_head(socket: &mut tokio::net::TcpStream) -> String {
    let mut data = Vec::new();
    let mut buffer = [0u8; 2048];
    loop {
        let read = socket.read(&mut buffer).await.expect("read");
        data.extend_from_slice(&buffer[..read]);
        let text = String::from_utf8_lossy(&data).into_owned();
        if read == 0 || text.contains("\r\n\r\n") {
            return text;
        }
    }
}

const OK_JSON: &str =
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n";
const OK_BYTES: &str =
    "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n";
const PARTIAL: &str =
    "HTTP/1.1 206 Partial Content\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n";
const NOT_FOUND: &str = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

fn hub(base: &str) -> Hub {
    Hub::new(Some(base.to_owned()), None, Duration::from_secs(2)).expect("hub")
}

fn sha256_of(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn payload() -> Vec<u8> {
    (0..64_000u32).flat_map(|n| n.to_le_bytes()).collect()
}

fn remote(bytes: &[u8], sha256: Option<String>) -> RemoteFile {
    RemoteFile {
        path: "model-Q4_K_M.gguf".to_owned(),
        bytes: bytes.len() as u64,
        sha256,
    }
}

#[tokio::test]
async fn lists_the_gguf_files_of_a_repository() {
    let body = br#"[{"type":"file","path":"a-Q4_K_M.gguf","size":10,
                     "lfs":{"oid":"sha256:dead","size":4000}},
                    {"type":"file","path":"README.md","size":12}]"#
        .to_vec();
    let base = serve(vec![Reply::Full(OK_JSON, body)]).await;

    let files = hub(&base).list_gguf("owner/name", "main").await.expect("listed");

    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "a-Q4_K_M.gguf");
    assert_eq!(files[0].bytes, 4000);
    assert_eq!(files[0].sha256.as_deref(), Some("dead"));
}

/// Review Focus 2: a repository with no GGUF must be distinguishable from a failure.
#[tokio::test]
async fn a_repository_without_gguf_lists_nothing_without_failing() {
    let body = br#"[{"type":"file","path":"model.safetensors","size":99}]"#.to_vec();
    let base = serve(vec![Reply::Full(OK_JSON, body)]).await;

    let files = hub(&base).list_gguf("owner/name", "main").await.expect("listed");

    assert!(files.is_empty());
}

#[tokio::test]
async fn an_unknown_repository_is_not_found() {
    let base = serve(vec![Reply::Full(NOT_FOUND, Vec::new())]).await;

    let error = hub(&base).list_gguf("owner/name", "main").await.expect_err("refused");

    assert_eq!(error, ModelError::NotFound);
}

#[tokio::test]
async fn downloads_a_file_and_returns_its_digest() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("downloaded");

    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(std::fs::metadata(&part).expect("part").len(), bytes.len() as u64);
}

#[tokio::test]
async fn resumes_from_a_part_file_and_still_hashes_the_whole_file() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(PARTIAL, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    // Pretend a previous run stopped a third of the way in.
    let stopped = bytes.len() / 3;
    std::fs::write(&part, &bytes[..stopped]).expect("part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("resumed");

    // The digest covers the bytes already on disk as well as the new ones.
    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(std::fs::metadata(&part).expect("part").len(), bytes.len() as u64);
}

#[tokio::test]
async fn starts_over_when_the_server_ignores_the_range() {
    let bytes = payload();
    // 200 rather than 206: the body starts at zero even though a range was asked for.
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    std::fs::write(&part, &bytes[..1000]).expect("part");

    let digest = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("restarted");

    // Appending would have produced 1000 bytes too many and a wrong digest.
    assert_eq!(digest, sha256_of(&bytes));
    assert_eq!(std::fs::metadata(&part).expect("part").len(), bytes.len() as u64);
}

#[tokio::test]
async fn reports_progress_without_flooding() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let collect = std::sync::Arc::clone(&seen);

    hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            move |done, total| {
                if let Ok(mut log) = collect.lock() {
                    log.push((done, total));
                }
            },
        )
        .await
        .expect("downloaded");

    let log = seen.lock().expect("lock").clone();
    // At least the first and last reports, and nowhere near one per chunk.
    assert!(log.len() >= 2, "{log:?}");
    assert!(log.len() < 50, "throttling should keep this small: {}", log.len());
    assert_eq!(log.last().map(|(done, _)| *done), Some(bytes.len() as u64));
    assert_eq!(log.last().and_then(|(_, total)| *total), Some(bytes.len() as u64));
}

#[tokio::test]
async fn a_cancelled_download_keeps_what_it_wrote() {
    let bytes = payload();
    let base = serve(vec![Reply::Cut(OK_BYTES, bytes.clone(), bytes.len() / 2)]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let part = dir.path().join("model.gguf.part");
    let cancel = CancellationToken::new();
    let token = cancel.clone();
    let hub = hub(&base);
    let file = remote(&bytes, None);

    let cancelling = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        token.cancel();
    });
    let result = hub
        .fetch("owner/name", "main", &file, &part, &cancel, |_, _| {})
        .await;
    cancelling.await.expect("join");

    assert!(result.is_err(), "a cancelled download does not succeed");
    // The part file is the resume state, so it must survive.
    assert!(part.exists(), "the partial file must be kept");
}

#[tokio::test]
async fn a_wrong_digest_fails_the_job_and_leaves_no_model() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let database = dir.path().join("test.db");
    chatatui::storage::Store::open(&database).expect("store");

    let request = chatatui::models::download::PullRequest {
        repo: "owner/name".to_owned(),
        revision: "main".to_owned(),
        file: remote(&bytes, Some("0".repeat(64))),
        dir: dir.path().join("models"),
    };
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let collect = std::sync::Arc::clone(&events);
    chatatui::models::download::run(
        request,
        database,
        std::sync::Arc::new(hub(&base)),
        CancellationToken::new(),
        move |event| {
            if let Ok(mut log) = collect.lock() {
                log.push(event);
            }
        },
    )
    .await;

    let log = events.lock().expect("lock").clone();
    assert!(
        log.iter().any(|e| matches!(
            e,
            chatatui::models::download::PullEvent::Failed { .. }
        )),
        "{log:?}"
    );
    let (_, part) = chatatui::models::download::paths(
        &dir.path().join("models"),
        "owner/name",
        "model-Q4_K_M.gguf",
    )
    .expect("paths");
    assert!(!part.exists(), "a corrupt partial file must be removed");
}

/// Review Focus 4: a write that cannot happen must say so in French, not panic.
#[tokio::test]
async fn a_write_failure_is_reported_not_panicked() {
    let bytes = payload();
    let base = serve(vec![Reply::Full(OK_BYTES, bytes.clone())]).await;
    let dir = tempfile::tempdir().expect("tempdir");
    // A file where the store wants a directory: creating the parent must fail.
    let blocker = dir.path().join("blocked");
    std::fs::write(&blocker, b"not a directory").expect("write");
    let part = blocker.join("sub").join("model.gguf.part");

    let error = hub(&base)
        .fetch(
            "owner/name",
            "main",
            &remote(&bytes, None),
            &part,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect_err("refused");

    assert!(matches!(error, ModelError::Io(_)), "{error:?}");
    assert!(!error.to_string().is_empty());
}

/// Review Focus 3: a remote listing must not be able to name a path outside the store.
#[test]
fn the_store_path_stays_inside_the_store() {
    let root = Path::new("/tmp/store");

    assert!(chatatui::models::download::paths(root, "owner/name", "../../x.gguf").is_err());
    assert!(chatatui::models::download::paths(root, "../owner/name", "x.gguf").is_err());
    assert!(chatatui::models::download::paths(root, "noslash", "x.gguf").is_err());

    let (file, part) =
        chatatui::models::download::paths(root, "owner/name", "x.gguf").expect("paths");
    assert_eq!(file, root.join("owner").join("name").join("x.gguf"));
    assert_eq!(part, root.join("owner").join("name").join("x.gguf.part"));
}
```

`tempfile` and `sha2` must be reachable from an integration test: `tempfile` is already a dev-dependency; `sha2` is a normal dependency, so it is available to `tests/` too.

- [ ] **Step 8: Run the integration tests**

Run: `cargo test --test model_hub`
Expected: the two tests that depend on Task 3's `store::save` (`a_wrong_digest_fails_the_job_and_leaves_no_model`) fail to compile until Task 3 lands. Everything else PASSES. If Task 3 is not merged yet, confirm the other tests pass and re-run this command after merging.

- [ ] **Step 9: Checks**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

---

### Task 3: the inventory and the v11 migration

**Files:**
- Modify: `src/models/store.rs` (replace Task 1's stub)
- Modify: `src/storage/schema.rs` (append the v11 migration)

Task 1 already declared `store` in `src/models/mod.rs`, so **do not edit
`src/models/mod.rs`** — Task 2 is running in parallel against the same file.

**Depends on Task 1** (the module skeleton). Runs in parallel with Task 2.

**Interfaces:**
- Consumes: nothing but Task 1's skeleton (`StoreError` already exists in `crate::storage`).
- Produces:
  - `models::store::LocalModel { repo: String, revision: String, file: String, path: String, bytes: u64, sha256: Option<String>, architecture: Option<String>, quantization: Option<String>, context_length: Option<u64>, parameters: Option<u64>, downloaded_at: i64 }`
  - `models::store::save(conn: &Connection, model: &LocalModel) -> Result<(), StoreError>`
  - `models::store::list(conn: &Connection) -> Result<Vec<LocalModel>, StoreError>`
  - `models::store::delete(conn: &Connection, repo: &str, file: &str) -> Result<Option<LocalModel>, StoreError>`

- [ ] **Step 1: Write the failing tests**

Replace the stub `src/models/store.rs` with only the test module at first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn connection() -> Connection {
        let mut conn = Connection::open_in_memory().expect("memory");
        crate::storage::schema::migrate(&mut conn).expect("migrate");
        conn
    }

    fn model(file: &str) -> LocalModel {
        LocalModel {
            repo: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_owned(),
            revision: "main".to_owned(),
            file: file.to_owned(),
            path: format!("/models/bartowski/Qwen2.5-7B-Instruct-GGUF/{file}"),
            bytes: 4_431_401_088,
            sha256: Some("abc".to_owned()),
            architecture: Some("qwen2".to_owned()),
            quantization: Some("Q4_K_M".to_owned()),
            context_length: Some(32768),
            parameters: Some(7_615_616_512),
            downloaded_at: 1_760_000_000,
        }
    }

    #[test]
    fn saves_and_lists_a_model() {
        let conn = connection();

        save(&conn, &model("a-Q4_K_M.gguf")).expect("saved");
        let models = list(&conn).expect("listed");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0], model("a-Q4_K_M.gguf"));
    }

    #[test]
    fn lists_the_most_recent_first() {
        let conn = connection();
        let mut older = model("old.gguf");
        older.downloaded_at = 1_700_000_000;
        save(&conn, &older).expect("saved");
        save(&conn, &model("new.gguf")).expect("saved");

        let models = list(&conn).expect("listed");

        assert_eq!(models[0].file, "new.gguf");
        assert_eq!(models[1].file, "old.gguf");
    }

    #[test]
    fn downloading_again_replaces_the_row_rather_than_adding_one() {
        let conn = connection();
        save(&conn, &model("a.gguf")).expect("saved");
        let mut again = model("a.gguf");
        again.revision = "d34db33f".to_owned();
        again.sha256 = None;
        again.bytes = 999;

        save(&conn, &again).expect("saved again");
        let models = list(&conn).expect("listed");

        assert_eq!(models.len(), 1, "one file, one row");
        assert_eq!(models[0].revision, "d34db33f");
        assert_eq!(models[0].sha256, None);
        assert_eq!(models[0].bytes, 999);
    }

    #[test]
    fn deletes_a_model_and_returns_what_it_held() {
        let conn = connection();
        save(&conn, &model("a.gguf")).expect("saved");

        let removed = delete(&conn, "bartowski/Qwen2.5-7B-Instruct-GGUF", "a.gguf")
            .expect("deleted");

        assert_eq!(removed.as_ref().map(|m| m.file.as_str()), Some("a.gguf"));
        assert!(list(&conn).expect("listed").is_empty());
    }

    #[test]
    fn deleting_an_unknown_model_is_not_an_error() {
        let conn = connection();

        let removed = delete(&conn, "owner/name", "absent.gguf").expect("no error");

        assert_eq!(removed, None);
    }

    #[test]
    fn a_model_whose_metadata_is_unknown_round_trips_as_none() {
        let conn = connection();
        let bare = LocalModel {
            sha256: None,
            architecture: None,
            quantization: None,
            context_length: None,
            parameters: None,
            ..model("bare.gguf")
        };

        save(&conn, &bare).expect("saved");

        assert_eq!(list(&conn).expect("listed")[0], bare);
    }
}
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `cargo test --lib models::store`
Expected: compile errors — `LocalModel`, `save`, `list`, `delete` not found.

- [ ] **Step 3: Append the v11 migration**

In `src/storage/schema.rs`, add a new entry at the **end** of `MIGRATIONS` (never edit an existing one — `LATEST_VERSION` is `MIGRATIONS.len()`, so appending is the whole change):

```rust
    // v11: models downloaded from HuggingFace and kept on disk.
    r#"
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
    "#,
```

The index is on `(repo, file)` and not `(repo, revision, file)`: the path on disk has no revision component, so two revisions of one file name would own two rows while fighting over one file. `revision` is metadata.

- [ ] **Step 4: Implement the store**

Write this above the test module in `src/models/store.rs`:

```rust
//! The inventory of downloaded models (table `local_models`, schema v11).
//!
//! Like [`crate::rag::store`], these are free functions over a plain `Connection`, so both
//! the storage worker (listing, deleting) and the download job (saving, on its own
//! connection) can call them.

use rusqlite::{Connection, OptionalExtension, params};

use crate::storage::StoreError;

/// A model on this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModel {
    /// `owner/name` on HuggingFace.
    pub repo: String,
    /// Branch or commit the file came from.
    pub revision: String,
    /// File name inside the repository.
    pub file: String,
    /// Absolute path on disk.
    pub path: String,
    pub bytes: u64,
    /// Verified sha256; `None` means the Hub exposed none, so nothing was checked.
    pub sha256: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub context_length: Option<u64>,
    pub parameters: Option<u64>,
    /// Unix seconds.
    pub downloaded_at: i64,
}

impl LocalModel {
    /// How the model is named in the UI: `owner/name · Q4_K_M`.
    pub fn label(&self) -> String {
        match &self.quantization {
            Some(quantization) => format!("{} · {quantization}", self.repo),
            None => self.repo.clone(),
        }
    }
}

/// Inserts the model, replacing any row for the same `(repo, file)`.
pub fn save(conn: &Connection, model: &LocalModel) -> Result<(), StoreError> {
    conn.execute(
        "INSERT INTO local_models
             (repo, revision, file, path, bytes, sha256, architecture, quantization,
              context_length, parameters, downloaded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
         ON CONFLICT (repo, file) DO UPDATE SET
             revision = excluded.revision,
             path = excluded.path,
             bytes = excluded.bytes,
             sha256 = excluded.sha256,
             architecture = excluded.architecture,
             quantization = excluded.quantization,
             context_length = excluded.context_length,
             parameters = excluded.parameters,
             downloaded_at = excluded.downloaded_at",
        params![
            model.repo,
            model.revision,
            model.file,
            model.path,
            model.bytes,
            model.sha256,
            model.architecture,
            model.quantization,
            model.context_length,
            model.parameters,
            model.downloaded_at,
        ],
    )?;
    Ok(())
}

/// Every model, most recently downloaded first.
pub fn list(conn: &Connection) -> Result<Vec<LocalModel>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT repo, revision, file, path, bytes, sha256, architecture, quantization,
                context_length, parameters, downloaded_at
           FROM local_models
          ORDER BY downloaded_at DESC, repo, file",
    )?;
    let rows = statement.query_map([], row_to_model)?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

/// Removes a model's row, returning what it held (`None` when there was none).
pub fn delete(
    conn: &Connection,
    repo: &str,
    file: &str,
) -> Result<Option<LocalModel>, StoreError> {
    let found = conn
        .query_row(
            "SELECT repo, revision, file, path, bytes, sha256, architecture, quantization,
                    context_length, parameters, downloaded_at
               FROM local_models WHERE repo = ?1 AND file = ?2",
            params![repo, file],
            row_to_model,
        )
        .optional()?;
    if found.is_some() {
        conn.execute(
            "DELETE FROM local_models WHERE repo = ?1 AND file = ?2",
            params![repo, file],
        )?;
    }
    Ok(found)
}

fn row_to_model(row: &rusqlite::Row<'_>) -> rusqlite::Result<LocalModel> {
    Ok(LocalModel {
        repo: row.get(0)?,
        revision: row.get(1)?,
        file: row.get(2)?,
        path: row.get(3)?,
        bytes: row.get(4)?,
        sha256: row.get(5)?,
        architecture: row.get(6)?,
        quantization: row.get(7)?,
        context_length: row.get(8)?,
        parameters: row.get(9)?,
        downloaded_at: row.get(10)?,
    })
}
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `cargo test --lib models::store`
Expected: 6 tests PASS.

- [ ] **Step 6: Confirm the migration tests still pass**

Run: `cargo test --lib storage::schema`
Expected: PASS — the existing tests assert that a fresh database reaches `LATEST_VERSION` and that a v1 database migrates all the way up, which now means v11.

- [ ] **Step 7: Checks**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

---

### Task 4: configuration, commands, state and runtime wiring

**Depends on Tasks 1, 2 and 3.**

**Files:**
- Modify: `src/config.rs` (the `models` field, its default, the `[models]` block in `DEFAULT_CONFIG_TOML`)
- Modify: `src/action.rs` (three `Effect` variants, two `Action` variants)
- Modify: `src/event.rs` (two `AppEvent` variants)
- Modify: `src/commands.rs` (three `CommandId`s and their specs)
- Modify: `src/state/overlay.rs` (two `Overlay` variants)
- Create: `src/state/gguf_picker.rs`
- Modify: `src/state/mod.rs` (re-export `GgufPicker`)
- Modify: `src/storage/mod.rs` (two `StoreRequest`s, two `StoreEvent`s)
- Modify: `src/storage/worker.rs` (handle them)
- Modify: `src/app.rs` (state fields, command handlers, `on_pull_event`)
- Modify: `src/runtime.rs` (a `ModelsBackend`, the three effects)
- Modify: `tests/app_flow.rs` (the harness handles the new effects, plus flow tests)

**Interfaces:**
- Consumes: everything Tasks 1–3 produce.
- Produces: `App::models`, `App::pulling`, `App::gguf_files` state read by Task 5's views.

- [ ] **Step 1: Declare the configuration section**

In `src/config.rs`, add to `Config` next to the other section fields:

```rust
    /// `[models]` section: where downloaded models live.
    pub models: crate::models::ModelsConfig,
```

Add `models: crate::models::ModelsConfig::default(),` to `Config::default()`. `Config` has `deny_unknown_fields`, so without this field a `[models]` section is a parse error.

Append to `DEFAULT_CONFIG_TOML`, following the commented style of the existing sections:

```toml
# Modèles téléchargés depuis HuggingFace (/pull, /models).
# [models]
# Dossier des modèles (défaut : à côté de la base de données).
# dir = ""
# Variable d'environnement contenant un jeton HuggingFace, pour les dépôts restreints.
# token_env = "HF_TOKEN"
```

- [ ] **Step 2: Write the failing flow test**

In `tests/app_flow.rs`, add:

```rust
#[tokio::test]
async fn pull_lists_the_files_then_downloads_the_chosen_one() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    let effects = harness.app.update(Action::Submit("/pull owner/name".into()));

    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::ListGguf { repo } if repo == "owner/name")),
        "{effects:?}"
    );

    let files = vec![chatatui::models::hub::RemoteFile {
        path: "m-Q4_K_M.gguf".to_owned(),
        bytes: 4_000_000_000,
        sha256: Some("abc".to_owned()),
    }];
    harness.app.update(Action::GgufFiles {
        repo: "owner/name".to_owned(),
        result: Ok(files),
    });

    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::GgufPicker(_))
    ));

    let effects = harness
        .app
        .update(Action::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)));

    assert!(
        effects.iter().any(|e| matches!(
            e,
            Effect::StartPull { repo, file } if repo == "owner/name" && file.path == "m-Q4_K_M.gguf"
        )),
        "{effects:?}"
    );
    assert!(harness.app.overlay.is_none(), "the picker closes on Enter");
}

#[tokio::test]
async fn a_repository_without_gguf_says_so_instead_of_opening_an_empty_picker() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));

    harness.app.update(Action::Submit("/pull owner/name".into()));
    harness.app.update(Action::GgufFiles {
        repo: "owner/name".to_owned(),
        result: Ok(Vec::new()),
    });

    assert!(harness.app.overlay.is_none());
    assert!(
        matches!(&harness.app.status, Status::Error(message) if message.contains("gguf")
            || message.contains("GGUF")),
        "{:?}",
        harness.app.status
    );
}

#[tokio::test]
async fn a_second_pull_is_refused_while_one_runs() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    harness.app.update(Action::Pull(chatatui::models::download::PullEvent::Progress {
        repo: "owner/name".to_owned(),
        file: "m.gguf".to_owned(),
        done: 10,
        total: Some(100),
        rate: 5,
    }));

    let effects = harness
        .app
        .update(Action::Submit("/pull other/name".into()));

    assert!(
        !effects.iter().any(|e| matches!(e, Effect::ListGguf { .. })),
        "{effects:?}"
    );
    assert!(matches!(harness.app.status, Status::Error(_)));
}

#[tokio::test]
async fn esc_cancels_the_download_when_no_overlay_is_open() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    harness.app.update(Action::Pull(chatatui::models::download::PullEvent::Progress {
        repo: "owner/name".to_owned(),
        file: "m.gguf".to_owned(),
        done: 10,
        total: Some(100),
        rate: 5,
    }));

    let effects = harness
        .app
        .update(Action::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));

    assert!(effects.iter().any(|e| matches!(e, Effect::CancelPull)), "{effects:?}");
}

#[tokio::test]
async fn a_finished_pull_records_the_model_and_refreshes_the_list() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(Vec::new())));
    let model = chatatui::models::store::LocalModel {
        repo: "owner/name".to_owned(),
        revision: "main".to_owned(),
        file: "m-Q4_K_M.gguf".to_owned(),
        path: "/models/owner/name/m-Q4_K_M.gguf".to_owned(),
        bytes: 4_000_000_000,
        sha256: Some("abc".to_owned()),
        architecture: Some("qwen2".to_owned()),
        quantization: Some("Q4_K_M".to_owned()),
        context_length: Some(32768),
        parameters: Some(7_615_616_512),
        downloaded_at: 1_760_000_000,
    };

    let effects = harness.app.update(Action::Pull(
        chatatui::models::download::PullEvent::Finished(Box::new(model)),
    ));

    assert!(harness.app.pulling.is_none());
    assert!(matches!(harness.app.status, Status::Info(_)));
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, Effect::Store(chatatui::storage::StoreRequest::ListModels))),
        "{effects:?}"
    );
}
```

- [ ] **Step 3: Run them to confirm they fail**

Run: `cargo test --test app_flow`
Expected: compile errors — `Effect::ListGguf`, `Action::GgufFiles`, `App::pulling` and friends do not exist.

- [ ] **Step 4: Add the effects, events and actions**

In `src/action.rs`, add to `Effect`:

```rust
    /// List a HuggingFace repository's GGUF files, to choose one.
    ListGguf { repo: String },
    /// Download one of a repository's GGUF files.
    StartPull {
        repo: String,
        file: crate::models::hub::RemoteFile,
    },
    /// Stop the running download (the partial file is kept, so it can resume).
    CancelPull,
```

Deleting a model needs no `Effect` of its own: it is a storage operation, so it goes
through `StoreRequest::DeleteModel` exactly as `/forget` goes through `DeleteCollection`.

and to `Action`:

```rust
    /// A repository's GGUF files, or why they could not be listed.
    GgufFiles {
        repo: String,
        result: Result<Vec<crate::models::hub::RemoteFile>, String>,
    },
    /// Progress of the download job.
    Pull(crate::models::download::PullEvent),
```

In `src/event.rs`, add the matching `AppEvent` variants:

```rust
    /// GGUF files offered by a repository (`/pull <repo>`).
    GgufFiles {
        repo: String,
        result: Result<Vec<crate::models::hub::RemoteFile>, String>,
    },
    /// Progress of the download job.
    Pull(crate::models::download::PullEvent),
```

and map them to their `Action`s wherever the other `AppEvent`s are converted (follow how `AppEvent::Index` becomes `Action::Index`).

- [ ] **Step 5: Add the commands**

In `src/commands.rs`, add to `CommandId`: `Pull`, `Models`, `RmModel`. Add their specs to the registry, next to `Index` and `Collections` so related commands stay together:

```rust
    CommandSpec {
        id: CommandId::Pull,
        name: "pull",
        aliases: &["telecharger"],
        arg: Arg::Required("<dépôt> [fichier]"),
        description: "Télécharger un modèle GGUF depuis HuggingFace",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Models,
        name: "models",
        aliases: &["modeles"],
        arg: Arg::None,
        description: "Modèles téléchargés sur cette machine",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::RmModel,
        name: "rm",
        aliases: &[],
        arg: Arg::Required("<dépôt> <fichier>"),
        description: "Supprimer un modèle téléchargé",
        shortcut: None,
        legacy_shortcut: None,
    },
```

Check whether `commands.rs` has a test asserting the registry's length or that every `CommandId` has a spec; update it if so.

- [ ] **Step 6: Add the picker state**

Create `src/state/gguf_picker.rs`:

```rust
//! The popup that picks which GGUF file of a repository to download.

use crate::models::hub::RemoteFile;

/// State of the open popup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GgufPicker {
    /// Repository the files come from.
    pub repo: String,
    pub files: Vec<RemoteFile>,
    /// Case-insensitive words typed by the user; all must match.
    pub filter: String,
    /// Index into [`GgufPicker::visible`].
    pub selected: usize,
}

impl GgufPicker {
    /// A picker over a repository's files, smallest first so the usual choice is on top.
    pub fn new(repo: String, mut files: Vec<RemoteFile>) -> Self {
        files.sort_by_key(|f| (f.bytes, f.path.clone()));
        Self {
            repo,
            files,
            filter: String::new(),
            selected: 0,
        }
    }

    /// Files matching the filter.
    pub fn visible(&self) -> Vec<&RemoteFile> {
        let words: Vec<String> = self
            .filter
            .split_whitespace()
            .map(str::to_lowercase)
            .collect();
        self.files
            .iter()
            .filter(|file| {
                let haystack = file.path.to_lowercase();
                words.iter().all(|word| haystack.contains(word))
            })
            .collect()
    }

    /// The highlighted file, if the filter matches anything.
    pub fn selected(&self) -> Option<&RemoteFile> {
        self.visible().get(self.selected).copied()
    }

    /// Moves the highlight by `delta`, clamped to the visible files.
    pub fn move_selection(&mut self, delta: isize) {
        let count = self.visible().len();
        if count == 0 {
            self.selected = 0;
            return;
        }
        let last = count - 1;
        self.selected = match delta {
            d if d < 0 => self.selected.saturating_sub(d.unsigned_abs()),
            d => (self.selected + d.unsigned_abs()).min(last),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, bytes: u64) -> RemoteFile {
        RemoteFile {
            path: path.to_owned(),
            bytes,
            sha256: None,
        }
    }

    #[test]
    fn lists_the_smallest_file_first() {
        let picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("b-Q8_0.gguf", 7_000), file("a-Q4_K_M.gguf", 4_000)],
        );

        assert_eq!(picker.selected().map(|f| f.path.as_str()), Some("a-Q4_K_M.gguf"));
    }

    #[test]
    fn the_filter_keeps_only_matching_files() {
        let mut picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("a-Q4_K_M.gguf", 4_000), file("b-Q8_0.gguf", 7_000)],
        );
        picker.filter = "q8".to_owned();

        assert_eq!(picker.visible().len(), 1);
        assert_eq!(picker.selected().map(|f| f.path.as_str()), Some("b-Q8_0.gguf"));
    }

    #[test]
    fn a_filter_matching_nothing_selects_nothing() {
        let mut picker =
            GgufPicker::new("owner/name".to_owned(), vec![file("a.gguf", 4_000)]);
        picker.filter = "zzz".to_owned();

        assert_eq!(picker.selected(), None);
    }

    #[test]
    fn the_highlight_stays_inside_the_list() {
        let mut picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("a.gguf", 1), file("b.gguf", 2)],
        );

        picker.move_selection(10);
        assert_eq!(picker.selected().map(|f| f.path.as_str()), Some("b.gguf"));
        picker.move_selection(-10);
        assert_eq!(picker.selected().map(|f| f.path.as_str()), Some("a.gguf"));
    }
}
```

Declare and re-export it from `src/state/mod.rs` the way `model_picker` is.

- [ ] **Step 7: Add the overlays**

In `src/state/overlay.rs`, add to `Overlay`:

```rust
    /// Downloaded models (/models).
    Models {
        scroll: u16,
    },
    /// Which GGUF file of a repository to download (/pull <repo>).
    GgufPicker(GgufPicker),
```

Extend `kind()` — `Models` is `OverlayKind::Text`, `GgufPicker` is `OverlayKind::List` — and `scroll_mut()`, adding `Models { scroll }` to the `Some(scroll)` arm and `GgufPicker(_)` to the `None` arm. Both matches are exhaustive, so the compiler names every site that needs updating.

- [ ] **Step 8: Add the storage requests**

In `src/storage/mod.rs`, add to `StoreRequest`:

```rust
    /// List downloaded models.
    ListModels,
    /// Delete a downloaded model: its row and its file.
    DeleteModel { repo: String, file: String },
```

and to `StoreEvent`:

```rust
    /// Downloaded models, listed at `now` (Unix seconds).
    Models {
        models: Vec<crate::models::store::LocalModel>,
        now: i64,
    },
    /// Result of [`StoreRequest::DeleteModel`]: `found` is `false` if there was none.
    ModelDeleted { file: String, found: bool },
```

In `src/storage/worker.rs`, handle both, following `ListCollections` and `DeleteCollection`. `DeleteModel` removes the row through `models::store::delete`, then removes the file at the row's `path`; a file that is already gone is not an error, since the point is to leave nothing behind:

```rust
            StoreRequest::ListModels => match crate::models::store::list(&self.conn) {
                Ok(models) => Some(StoreEvent::Models { models, now: now() }),
                Err(error) => Some(StoreEvent::Error(error.to_string())),
            },
            StoreRequest::DeleteModel { repo, file } => {
                match crate::models::store::delete(&self.conn, &repo, &file) {
                    Ok(Some(model)) => {
                        // An already-missing file is not a failure: the goal is that
                        // nothing is left.
                        let _ = std::fs::remove_file(&model.path);
                        Some(StoreEvent::ModelDeleted { file, found: true })
                    }
                    Ok(None) => Some(StoreEvent::ModelDeleted { file, found: false }),
                    Err(error) => Some(StoreEvent::Error(error.to_string())),
                }
            }
```

Match the surrounding code's exact shape — whether handlers return `Option<StoreEvent>` or send directly — rather than this sketch.

- [ ] **Step 9: Add the app state and handlers**

In `src/app.rs`, add to `App`:

```rust
    /// Downloaded models and when they were listed (`None`: not loaded yet).
    pub models: Option<(Vec<crate::models::store::LocalModel>, i64)>,
    /// The running download, if any.
    pub pulling: Option<PullProgress>,
    /// File named on the `/pull` command line, held until the listing comes back and its
    /// size and checksum are known.
    wanted_file: Option<String>,
```

and the progress struct, next to `IndexProgress`:

```rust
/// Progress of the running download.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PullProgress {
    pub repo: String,
    pub file: String,
    pub done: u64,
    /// Total size; `None` until the server says.
    pub total: Option<u64>,
    /// Bytes per second since the download started.
    pub rate: u64,
}
```

Initialise all three to `None` in `App::new`. Then the handlers:

```rust
    /// `/pull <repo> [file]`: lists a repository's GGUF files, or downloads one directly.
    fn start_pull(&mut self, arg: &str) -> Vec<Effect> {
        if let Some(progress) = &self.pulling {
            self.status = Status::Error(format!(
                "téléchargement de « {} » déjà en cours (Échap pour l'arrêter)",
                progress.file
            ));
            return Vec::new();
        }
        let mut words = arg.split_whitespace();
        let Some(target) = words.next() else {
            self.status =
                Status::Error("usage : /pull <propriétaire>/<nom> [fichier.gguf]".into());
            return Vec::new();
        };
        let (repo, from_url) = match crate::models::hub::parse_target(target) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.status = Status::Error(error.to_string());
                return Vec::new();
            }
        };
        match words.next().map(str::to_owned).or(from_url) {
            // A named file still needs its size and checksum, so list first and pick it
            // out of the answer.
            Some(file) => {
                self.wanted_file = Some(file);
                vec![Effect::ListGguf { repo }]
            }
            None => {
                self.wanted_file = None;
                vec![Effect::ListGguf { repo }]
            }
        }
    }

    fn on_gguf_files(
        &mut self,
        repo: String,
        result: Result<Vec<crate::models::hub::RemoteFile>, String>,
    ) -> Vec<Effect> {
        let files = match result {
            Ok(files) => files,
            Err(error) => {
                self.status = Status::Error(format!("{repo} : {error}"));
                return Vec::new();
            }
        };
        if files.is_empty() {
            self.status =
                Status::Error(format!("aucun fichier .gguf dans le dépôt {repo}"));
            return Vec::new();
        }
        if let Some(wanted) = self.wanted_file.take() {
            let Some(file) = files.iter().find(|f| f.path == wanted).cloned() else {
                self.status = Status::Error(format!("{wanted} introuvable dans {repo}"));
                return Vec::new();
            };
            return self.begin_pull(repo, file);
        }
        self.overlay = Some(Overlay::GgufPicker(crate::state::GgufPicker::new(
            repo, files,
        )));
        Vec::new()
    }

    /// Starts the download and shows its progress in the status bar.
    fn begin_pull(
        &mut self,
        repo: String,
        file: crate::models::hub::RemoteFile,
    ) -> Vec<Effect> {
        self.pulling = Some(PullProgress {
            repo: repo.clone(),
            file: file.path.clone(),
            total: (file.bytes > 0).then_some(file.bytes),
            ..PullProgress::default()
        });
        // The status bar shows the progress (and Esc to stop it).
        self.status = Status::Ready;
        vec![Effect::StartPull { repo, file }]
    }

    fn on_pull_event(&mut self, event: crate::models::download::PullEvent) -> Vec<Effect> {
        use crate::models::download::PullEvent;
        match event {
            PullEvent::Progress {
                repo,
                file,
                done,
                total,
                rate,
            } => {
                self.pulling = Some(PullProgress {
                    repo,
                    file,
                    done,
                    total,
                    rate,
                });
                Vec::new()
            }
            PullEvent::Finished(model) => {
                self.pulling = None;
                self.status = Status::Info(format!(
                    "{} téléchargé ({})",
                    model.file,
                    crate::tokens::format_bytes(model.bytes)
                ));
                vec![Effect::Store(StoreRequest::ListModels)]
            }
            PullEvent::Failed { file, error, .. } => {
                self.pulling = None;
                self.status = Status::Error(format!("téléchargement de {file} : {error}"));
                Vec::new()
            }
            PullEvent::Cancelled { file, .. } => {
                self.pulling = None;
                self.status = Status::Info(format!(
                    "téléchargement de {file} arrêté (/pull reprendra où il s'est arrêté)"
                ));
                Vec::new()
            }
        }
    }

    /// `/rm <repo> <file>`: deletes a downloaded model.
    fn remove_model(&mut self, arg: &str) -> Vec<Effect> {
        let mut words = arg.split_whitespace();
        match (words.next(), words.next(), words.next()) {
            (Some(repo), Some(file), None) => vec![Effect::Store(StoreRequest::DeleteModel {
                repo: repo.to_owned(),
                file: file.to_owned(),
            })],
            _ => {
                self.status =
                    Status::Error("usage : /rm <propriétaire>/<nom> <fichier.gguf>".into());
                Vec::new()
            }
        }
    }
```

Wire the rest:
- `CommandId::Pull` → `start_pull`, `CommandId::RmModel` → `remove_model`, and `CommandId::Models` opens `Overlay::Models { scroll: 0 }` and emits `Effect::Store(StoreRequest::ListModels)` — follow exactly what `CommandId::Collections` does.
- `Action::GgufFiles { .. }` → `on_gguf_files`, `Action::Pull(event)` → `on_pull_event`.
- `StoreEvent::Models { models, now }` → `self.models = Some((models, now))`. `StoreEvent::ModelDeleted { file, found }` → a `Status::Info` or `Status::Error`, then `Effect::Store(StoreRequest::ListModels)` to refresh, mirroring `CollectionDeleted`.
- In the `GgufPicker` overlay, `Enter` takes `selected()`, closes the overlay and calls `begin_pull`; arrows call `move_selection`; typed characters extend `filter` and reset `selected` to 0. Copy the `ModelPicker` key handling.
- Esc: extend the existing chain (`app.rs:880` area) so that with no overlay open and a download running, Esc emits `Effect::CancelPull`. Keep the documented order: overlays first, then the generation, then indexing, then the download.

`crate::tokens::format_bytes` may not exist — check `src/tokens.rs` for an existing byte formatter (`format_count` is for tokens). If there is none, add one there beside it:

```rust
/// A byte count for the UI: `4,4 Go`, `812 Mo`, `96 ko`.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 3] = [("Go", 1 << 30), ("Mo", 1 << 20), ("ko", 1 << 10)];
    for (unit, scale) in UNITS {
        if bytes >= scale {
            let value = bytes as f64 / scale as f64;
            return if value < 10.0 {
                format!("{value:.1} {unit}").replace('.', ",")
            } else {
                format!("{} {unit}", value.round() as u64)
            };
        }
    }
    format!("{bytes} o")
}
```

with tests for `0 o`, `1,0 ko`, `512 Mo`, `4,4 Go` and `u64::MAX`.

- [ ] **Step 10: Wire the runtime**

In `src/runtime.rs`, add to `Runtime`:

```rust
    /// What `/pull` needs.
    models: ModelsBackend,
    /// Cancellation handle of the running download.
    running_pull: Option<CancellationToken>,
```

and the backend, next to `RagBackend` (with a hand-written `Debug` that never prints the token, like `RagBackend`'s):

```rust
/// The HuggingFace client and the store directory used by `/pull`, or why it is
/// unavailable.
struct ModelsBackend {
    hub: Result<Arc<crate::models::hub::Hub>, String>,
    /// Root of the model store.
    dir: PathBuf,
    database: Result<PathBuf, String>,
}
```

Build it where `RagBackend` is built: the token comes from `std::env::var(&config.models.token_env)` (an empty or absent variable means no token); `dir` is `config.models.dir` expanded with `files::expand_home` when non-empty, otherwise the database's parent joined with `models`.

Handle the effects:

```rust
            Effect::ListGguf { repo } => {
                let sender = self.events.sender();
                let hub = match &self.models.hub {
                    Ok(hub) => Arc::clone(hub),
                    Err(error) => {
                        // Fails only while shutting down.
                        let _ = sender.send(Event::App(AppEvent::GgufFiles {
                            repo,
                            result: Err(error.clone()),
                        }));
                        return;
                    }
                };
                tokio::spawn(async move {
                    let result = hub
                        .list_gguf(&repo, "main")
                        .await
                        .map_err(|e| e.to_string());
                    // Fails only while shutting down.
                    let _ = sender.send(Event::App(AppEvent::GgufFiles { repo, result }));
                });
            }
            Effect::StartPull { repo, file } => self.start_pull(repo, file),
            Effect::CancelPull => {
                if let Some(token) = &self.running_pull {
                    token.cancel();
                }
            }
```

and the job starter, modelled on `start_index`:

```rust
    /// Spawns the download job, or reports right away why it cannot run.
    fn start_pull(&mut self, repo: String, file: crate::models::hub::RemoteFile) {
        let sender = self.events.sender();
        let fail = |error: String| {
            // Fails only while shutting down.
            let _ = sender.send(Event::App(AppEvent::Pull(PullEvent::Failed {
                repo: repo.clone(),
                file: file.path.clone(),
                error,
            })));
        };
        if self.running_pull.is_some() {
            return fail("un téléchargement est déjà en cours".into());
        }
        let hub = match &self.models.hub {
            Ok(hub) => Arc::clone(hub),
            Err(error) => return fail(error.clone()),
        };
        let database = match &self.models.database {
            Ok(path) => path.clone(),
            Err(error) => return fail(error.clone()),
        };
        let request = PullRequest {
            repo,
            revision: "main".to_owned(),
            file,
            dir: self.models.dir.clone(),
        };
        let token = CancellationToken::new();
        self.running_pull = Some(token.clone());
        let sender = self.events.sender();
        tokio::spawn(download::run(
            request,
            database,
            hub,
            token,
            move |event| {
                // Fails only while shutting down.
                let _ = sender.send(Event::App(AppEvent::Pull(event)));
            },
        ));
    }
```

Clear `running_pull` when a terminal `PullEvent` arrives, exactly where `running_index` is cleared (`runtime.rs:244`).

- [ ] **Step 11: Teach the test harness the new effects**

In `tests/app_flow.rs`, extend the `Harness` effect loop to handle `Effect::ListGguf`, `Effect::StartPull` and `Effect::CancelPull`. The flow tests above drive events in by hand, so the harness only needs to record the effects rather than perform real HTTP — follow how it treats `Effect::StartIndex`.

- [ ] **Step 12: Run the tests**

Run: `cargo test`
Expected: the new flow tests PASS along with everything that already passed.

- [ ] **Step 13: Checks**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

---

### Task 5: the views, the status bar and the docs

**Depends on Task 4.**

**Files:**
- Create: `src/ui/models_view.rs`
- Create: `src/ui/gguf_picker.rs`
- Modify: `src/ui/mod.rs` (declare both, render the overlays)
- Modify: `src/ui/status_bar.rs` (the download line)
- Modify: `src/ui/help.rs` if it lists commands by hand
- Modify: `README.md`, `docs/roadmap.md`, `PLAN.md`

**Interfaces:**
- Consumes: `App::models`, `App::pulling`, `Overlay::Models`, `Overlay::GgufPicker` from Task 4; `LocalModel::label` and `tokens::format_bytes`.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the `/models` view**

Create `src/ui/models_view.rs`, following `src/ui/collections_view.rs` closely — same `lines(app: &App, width: usize) -> Vec<Line<'static>>` signature, same `bold` / `dim` / `title` styles from `crate::theme::palette()`, same `ago` helper for relative dates:

```rust
//! Content of the /models popup: the models downloaded on this machine.
```

Each model gets a bold first line with `model.repo` and `ago(now - model.downloaded_at)`, then a dim detail line: the quantization, `format_bytes(model.bytes)`, the architecture, the context window through `format_count`, and the parameter count. A model whose `sha256` is `None` is marked `⚠ non vérifié` — never silently as if it had been checked.

Mirror `collections_view`'s three states: `None` renders ` chargement…`, an empty list renders ` Aucun modèle téléchargé. /pull <dépôt> pour en ajouter.`, and otherwise one entry per model.

Close with a dim footer naming the next step honestly, because a model that cannot be run is confusing otherwise:

```
 Les modèles locaux ne sont pas encore utilisables pour répondre (J33).
```

- [ ] **Step 2: Write the GGUF picker view**

Create `src/ui/gguf_picker.rs`, following `src/ui/model_picker.rs`: the repository as a title, the filter as typed, and one row per visible file — the file name, then `format_bytes(file.bytes)` dim and right-aligned — with the selected row using the selection style `model_picker.rs` already uses. A filter matching nothing renders ` Aucun fichier ne correspond.`

- [ ] **Step 3: Render both overlays**

In `src/ui/mod.rs`, declare both modules and add the two `Overlay` arms where `Collections` and `ModelPicker` are handled. `Models` is a scrollable text popup, so reuse the `text_popup` path with the title `Modèles` and `models_view::lines`. `GgufPicker` reuses the list-popup path with the title `Télécharger un modèle`.

- [ ] **Step 4: Write the failing status-bar test**

In `src/ui/status_bar.rs`, add to its test module (follow the existing tests' style for building an `App`):

```rust
    #[test]
    fn shows_the_download_progress_and_its_rate() {
        let mut app = test_app();
        app.pulling = Some(crate::app::PullProgress {
            repo: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_owned(),
            file: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_owned(),
            done: 2_100_000_000,
            total: Some(4_400_000_000),
            rate: 18_000_000,
        });

        let text = line_text(&app, 120);

        assert!(text.contains('⬇'), "{text}");
        assert!(text.contains("Q4_K_M"), "{text}");
        assert!(text.contains("2,0/4,1 Go") || text.contains("Go"), "{text}");
        assert!(text.contains("/s"), "{text}");
    }

    #[test]
    fn a_download_of_unknown_size_shows_what_it_has() {
        let mut app = test_app();
        app.pulling = Some(crate::app::PullProgress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 1_048_576,
            total: None,
            rate: 0,
        });

        let text = line_text(&app, 120);

        assert!(text.contains("1,0 Mo"), "{text}");
        // No total, so no percentage and no bogus denominator.
        assert!(!text.contains('%'), "{text}");
    }
```

`test_app` and `line_text` may be named differently — read the existing tests in `status_bar.rs` and reuse whatever they use.

- [ ] **Step 5: Run them to confirm they fail**

Run: `cargo test --lib ui::status_bar`
Expected: FAIL — the download segment is not rendered yet.

- [ ] **Step 6: Render the download in the status bar**

Add `with_pull`, beside `with_indexing` at `src/ui/status_bar.rs:163`, and call it from the same place in the left-hand line:

```rust
/// Appends the download progress (`⬇ Q4_K_M 2,1/4,4 Go · 18 Mo/s`) while `/pull` runs.
fn with_pull(app: &App, mut line: Line<'static>) -> Line<'static> {
    let Some(progress) = &app.pulling else {
        return line;
    };
    let done = crate::tokens::format_bytes(progress.done);
    let size = match progress.total {
        Some(total) => format!("{done}/{}", crate::tokens::format_bytes(total)),
        None => done,
    };
    let rate = if progress.rate > 0 {
        format!(" · {}/s", crate::tokens::format_bytes(progress.rate))
    } else {
        String::new()
    };
    // The file name carries the quantization, which is what the user is waiting on.
    let name = progress.file.trim_end_matches(".gguf");
    line.spans.push(Span::styled(
        format!("  ⬇ {name} {size}{rate}"),
        Style::default().fg(crate::theme::palette().info),
    ));
    line
}
```

Also extend the hint at `src/ui/status_bar.rs:261` so `Échap` is advertised while a download runs, the way it is for indexing.

- [ ] **Step 7: Run them to confirm they pass**

Run: `cargo test --lib ui::status_bar`
Expected: PASS.

- [ ] **Step 8: Update the help screen**

Check whether `src/ui/help.rs` builds its command list from the `commands.rs` registry or by hand. If by hand, add `/pull`, `/models` and `/rm`. If from the registry, confirm they already appear and that any snapshot test (`insta`) is reviewed with `cargo insta review` rather than blindly accepted.

- [ ] **Step 9: Update the documentation**

`README.md`: add a bullet to the feature list and a `### Modèles locaux` subsection after the providers section, documenting `/pull <dépôt> [fichier]`, `/models`, `/rm`, the `[models]` keys (`dir`, `token_env`), where files are stored, that a cancelled download resumes, and — stated plainly — that downloaded models cannot answer yet.

`docs/roadmap.md`: add to **Done** —

```markdown
- **J32**: local model store — `/pull` downloads a GGUF from HuggingFace (resumable,
  sha256-checked), `/models` lists what is on disk with the metadata read from the file,
  `/rm` deletes one.
```

and replace the **Next** bullet's neighbours so the engine is named: `- Running local models: GGUF tokenizer and inference (J33).`

`PLAN.md`: add the J32 row to the milestone table in the established voice, and add the `models/` module to the module tree.

- [ ] **Step 10: Final checks**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

Then confirm the new paths work end to end by hand, since no test exercises the real Hub:

```bash
cargo run --release
```

In the app: `/pull hf.co/unsloth/Qwen3-0.6B-GGUF` (a small repository, so the download finishes quickly), pick a file, watch the status bar, press `Esc` mid-download, re-run the same `/pull` and confirm it resumes rather than restarting, then `/models` and `/rm`.
