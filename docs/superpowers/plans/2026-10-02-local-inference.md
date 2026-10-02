# J34 — Local inference: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a Qwen3 0.6B GGUF already in the local store answer in streaming inside the TUI, on CPU, with no daemon and no companion file.

**Architecture:** A fourth provider. `LocalClient` implements the existing `LlmClient` trait, so nothing in `App`, the runtime or the streaming task changes shape. It loads weights with `candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf`, builds its tokenizer from the same `Content` candle already parsed, renders the prompt itself (no server does it any more), and decodes on a dedicated OS thread whose tokens cross back through an mpsc channel presented as a `TokenStream`.

**Tech Stack:** Rust 2024 (`rust-version = "1.88"`), candle 0.11 (`candle-core`, `candle-nn`, `candle-transformers`), `tokenizers`, tokio, futures.

**Spec:** `docs/superpowers/specs/2026-10-02-local-inference-design.md`

## Execution Order

```
Task 1 (tokenizer)  ┐
Task 2 (template)   ┘ in parallel, isolated worktrees
   └── Task 3 (engine, thread, stream)
          └── Task 4 (wiring, docs, the cargo tree gate)
```

Tasks 1 and 2 share no file and consume nothing from each other. **They must run in
*isolated worktrees*, not the shared one.** In Rust the crate is a single compilation unit:
each task's TDD red phase leaves the lib test binary uncompilable, so one agent's
`cargo test` would fail on the other's half-written file. This was hit for real in J32 and
confirmed in J33 — "disjoint files" is true for editing and false for compiling.

Task 3 needs both. Task 4 comes last because it is the only task that can run the whole
suite and the dependency gate.

## Global Constraints

- Rust edition 2024, `rust-version = "1.88"`. No `unwrap()` or `expect()` outside `#[cfg(test)]` code.
- UI strings in French; code, comments, doc comments and the documentation files in English.
- **`cargo tree` must contain no `*-sys` crate.** This is a gate, not a nicety: it is the one check that proves the milestone's central trade-off — pure Rust over `llama-cpp-2` — was actually honoured. The release matrix cross-compiles `aarch64-unknown-linux-gnu`; a C dependency would need a cross toolchain there.
- **`App::update` stays pure: no I/O, and no clock.** Nothing in this milestone touches it.
- `ui::render(&App, frame)` stays read-only.
- **The 30 fps markdown cap must survive.** `update` skips `refresh_view()` for a token and the runtime redraws a tick only when `transcript.revision()` or the spinner frame changed. This milestone adds no tick-driven state, so the cap is preserved by not touching it — but a task that makes the transcript dirty per token is a regression even if it looks right.
- **One commit for the whole milestone.** Do not commit per task; each task ends at its checks. The orchestrator squashes the branch into one commit, per the repo's `J<N>: …` convention, in English like `J31`/`J32`/`J33`.
- Inference runs on a dedicated OS thread, never on the tokio runtime.
- The local provider offers no tools and emits no `StreamItem::ToolCallDelta`.

## Review Focus

Input classes the spec implies but no task's own tests would otherwise exercise. Each line's test is added to the task that owns the code.

1. **Qwen3 emits `<think>` blocks.** Qwen3 is a hybrid reasoning model and its official chat template carries `enable_thinking` logic. Render plain ChatML without it and the model writes `<think>…</think>` into the reply — and J33 put reasoning tokens explicitly out of scope, so they would land in the conversation as literal text. (Task 2)
2. **A vocabulary entry that is not a string.** `Value::Array` is `Vec<Value>`, so a malformed or unusual GGUF can hold mixed element types. `Value::to_string()` returns a `Result`; unwrapping it would panic the application on a bad file. (Task 1)
3. **A GGUF with no tokenizer at all.** Sharded parts and embedding-only models carry no `tokenizer.ggml.tokens`. Loading one must fail with a message naming what is missing. (Task 1)
4. **The reader dropped mid-generation.** Cancellation works only because the engine thread's `send` fails once the stream is dropped. An engine that ignores the send error keeps decoding to completion, burning a core with nobody listening. (Task 3)
5. **A model file deleted between listing and loading.** `/model` lists from a directory scan; the file can be gone by the time `chat_stream` opens it. (Task 3)

---

### Task 1: The tokenizer, from candle's metadata

**Files:**
- Create: `src/models/tokenizer.rs`
- Modify: `src/models/mod.rs` (add `pub mod tokenizer;`)
- Modify: `Cargo.toml` (add `candle-core` and `tokenizers`)
- Test: appended `#[cfg(test)] mod tests` inside `src/models/tokenizer.rs`

**Interfaces:**
- Consumes: nothing from other tasks. `ModelError` from `crate::models` — its variants are `Http(String)`, `NotFound`, `Gguf(String)`, `Io(String)`, `Checksum`, and **`Gguf`'s `Display` already reads `"fichier GGUF illisible : {0}"`**, so do not repeat that prefix in the message you pass it.
- Produces:
  ```rust
  // src/models/tokenizer.rs
  pub struct TokenizerData {
      pub model: String,
      pub tokens: Vec<String>,
      pub merges: Vec<String>,
      pub token_type: Vec<i32>,
      pub bos: Option<u32>,
      pub eos: Option<u32>,
      pub unknown: Option<u32>,
      pub add_bos: Option<bool>,
      pub chat_template: Option<String>,
  }
  pub fn from_metadata(
      metadata: &std::collections::HashMap<String, candle_core::quantized::gguf_file::Value>,
  ) -> Result<TokenizerData, ModelError>;
  pub fn build(data: &TokenizerData) -> Result<tokenizers::Tokenizer, ModelError>;
  ```
  Task 3 calls `from_metadata(&content.metadata)` then `build(&data)`, and reads `data.eos`
  to know when to stop decoding and `data.chat_template` to decide whether it recognises the
  template.

- [ ] **Step 1: The dependency audit — this is a gate, and it decides how much code this task contains**

Before writing anything, settle the `tokenizers` dependency. The spec's resolution order:

1. `tokenizers = { version = "1.0.0-rc.2", default-features = false }` — preferred.
2. `tokenizers = { version = "0.23.2", default-features = false }` — only if the byte-level
   BPE path compiles and its tests pass with no regex engine at all.
3. Neither: **report `BLOCKED`** to the orchestrator with the `cargo tree` output. Do not
   improvise a third dependency and do not enable `unstable_wasm` to dodge `onig` — that
   feature is wasm-oriented and explicitly unstable, and using it in a native build to avoid
   a C library is a bodge. The fallback is a hand-written byte-level BPE, which the
   orchestrator will scope as its own task.

Add, in `Cargo.toml` under `[dependencies]`, keeping the file's existing alphabetical-ish grouping:

```toml
candle-core = "0.11"
tokenizers = { version = "1.0.0-rc.2", default-features = false }
```

Run: `cargo tree | grep -i -- '-sys' ; echo "exit=$?"`
Expected: no output, `exit=1` (grep found nothing). **If any `*-sys` crate appears, option 1
has failed** — try option 2, re-run this check, and report `BLOCKED` if it fails too.

Also run: `cargo tree -p tokenizers -e features | head -40` and record in your report which
features ended up enabled. The orchestrator needs this: it is the evidence that the gate was
actually checked rather than assumed.

- [ ] **Step 2: Pin the exact `tokenizers` API shapes by reading the local docs**

The signatures of `BpeBuilder::vocab_and_merges`, and whether `with_pre_tokenizer` /
`with_decoder` take an `Option`, **differ between `tokenizers` versions**. Do not write them
from memory.

Run: `cargo doc -p tokenizers --no-deps` then read, in `target/doc/tokenizers/`:
- `models/bpe/struct.BpeBuilder.html` — the exact parameter types of `vocab_and_merges`
- `struct.TokenizerBuilder.html` or `struct.Tokenizer.html` — how a pre-tokenizer and decoder
  are attached in this version
- `pre_tokenizers/byte_level/struct.ByteLevel.html` and
  `decoders/byte_level/struct.ByteLevel.html`

Write the two or three exact signatures into your report. Every code block below that touches
these APIs is written against `Vec<(String, String)>` merges and `Option`-taking setters; if
the pinned version differs, **adapt the code and say so in your report** — the shape of the
data never changes (each GGUF merge string is `"left right"`, split on its single space), only
the types around it.

- [ ] **Step 3: Write the failing test for `from_metadata`**

Create `src/models/tokenizer.rs` with only this test module (no implementation yet):

```rust
//! Builds a tokenizer from the metadata candle already parsed out of a GGUF.
//!
//! No companion `tokenizer.json`: one GGUF is the whole input, which is the point of the
//! milestone. The split between [`from_metadata`] and [`build`] is deliberate — the first
//! touches candle's types and the second is pure, so the interesting half is testable
//! without a model file.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::quantized::gguf_file::Value;

    use super::*;

    /// A metadata map shaped like a real Qwen3 GGUF's, small enough to read.
    fn metadata() -> HashMap<String, Value> {
        HashMap::from([
            ("tokenizer.ggml.model".to_owned(), Value::String("gpt2".to_owned())),
            (
                "tokenizer.ggml.tokens".to_owned(),
                Value::Array(vec![
                    Value::String("<|endoftext|>".to_owned()),
                    Value::String("Ġbon".to_owned()),
                    Value::String("jour".to_owned()),
                ]),
            ),
            (
                "tokenizer.ggml.merges".to_owned(),
                Value::Array(vec![Value::String("Ġbon jour".to_owned())]),
            ),
            (
                "tokenizer.ggml.token_type".to_owned(),
                Value::Array(vec![Value::I32(3), Value::I32(1), Value::I32(1)]),
            ),
            ("tokenizer.ggml.bos_token_id".to_owned(), Value::U32(0)),
            ("tokenizer.ggml.eos_token_id".to_owned(), Value::U32(0)),
            ("tokenizer.ggml.add_bos_token".to_owned(), Value::Bool(false)),
            (
                "tokenizer.chat_template".to_owned(),
                Value::String("{% for message in messages %}…".to_owned()),
            ),
        ])
    }

    #[test]
    fn lifts_the_vocabulary_in_order() {
        let data = from_metadata(&metadata()).expect("well-formed metadata");

        assert_eq!(data.model, "gpt2");
        assert_eq!(data.tokens, ["<|endoftext|>", "Ġbon", "jour"]);
        assert_eq!(data.merges, ["Ġbon jour"]);
        assert_eq!(data.token_type, [3, 1, 1]);
        assert_eq!(data.bos, Some(0));
        assert_eq!(data.eos, Some(0));
        assert_eq!(data.add_bos, Some(false));
        assert!(
            data.chat_template.is_some_and(|t| t.contains("messages")),
            "the template is captured, not interpreted"
        );
    }

    /// Review focus 3: sharded parts and embedding-only models carry no vocabulary. The
    /// message must name what is missing, because the user chose this file in `/models` and
    /// needs to know why it cannot answer.
    #[test]
    fn a_file_without_a_vocabulary_is_refused_by_name() {
        let mut metadata = metadata();
        metadata.remove("tokenizer.ggml.tokens");

        let error = from_metadata(&metadata).expect_err("no vocabulary");

        let text = error.to_string();
        assert!(text.contains("tokenizer.ggml.tokens"), "{text}");
    }

    /// Review focus 2: `Value::Array` is `Vec<Value>`, so the elements are not guaranteed to
    /// be strings. `Value::to_string()` returns a `Result` and must never be unwrapped — a
    /// malformed file is an error message, not a panic.
    #[test]
    fn a_vocabulary_entry_of_the_wrong_type_is_an_error_not_a_panic() {
        let mut metadata = metadata();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            Value::Array(vec![Value::String("ok".to_owned()), Value::U32(7)]),
        );

        let error = from_metadata(&metadata).expect_err("mixed element types");

        assert!(error.to_string().contains("tokenizer.ggml.tokens"));
    }

    #[test]
    fn a_missing_optional_key_is_none_rather_than_an_error() {
        let mut metadata = metadata();
        metadata.remove("tokenizer.ggml.bos_token_id");
        metadata.remove("tokenizer.chat_template");

        let data = from_metadata(&metadata).expect("optional keys are optional");

        assert_eq!(data.bos, None);
        assert_eq!(data.chat_template, None);
    }
}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cargo test --lib models::tokenizer`
Expected: FAIL to compile — `cannot find function 'from_metadata' in this scope`, and
`TokenizerData` unresolved. That is the right failure.

- [ ] **Step 5: Implement `from_metadata`**

Add above the test module in `src/models/tokenizer.rs`:

```rust
use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;

use super::ModelError;

/// The tokenizer data a local engine needs, lifted out of a GGUF's metadata.
///
/// `chat_template` is captured but never interpreted: it is a Jinja template, and a Jinja
/// engine is a different milestone. It is kept so the engine can say "I do not render this
/// model's template" instead of producing fluent nonsense.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerData {
    /// `tokenizer.ggml.model`: `"gpt2"`, `"llama"` or `"spm"`.
    pub model: String,
    pub tokens: Vec<String>,
    /// One `"left right"` pair per entry, as GGUF stores them.
    pub merges: Vec<String>,
    pub token_type: Vec<i32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unknown: Option<u32>,
    pub add_bos: Option<bool>,
    pub chat_template: Option<String>,
}

/// Lifts the tokenizer keys out of the metadata candle already parsed.
///
/// Only the vocabulary and the tokenizer family are required; everything else is optional,
/// because real files vary and a missing end-of-text id is recoverable while a missing
/// vocabulary is not.
pub fn from_metadata(metadata: &HashMap<String, Value>) -> Result<TokenizerData, ModelError> {
    Ok(TokenizerData {
        model: required_string(metadata, "tokenizer.ggml.model")?,
        tokens: string_array(metadata, "tokenizer.ggml.tokens")?,
        merges: optional_string_array(metadata, "tokenizer.ggml.merges")?,
        token_type: i32_array(metadata, "tokenizer.ggml.token_type")?,
        bos: optional_u32(metadata, "tokenizer.ggml.bos_token_id"),
        eos: optional_u32(metadata, "tokenizer.ggml.eos_token_id"),
        unknown: optional_u32(metadata, "tokenizer.ggml.unknown_token_id"),
        add_bos: metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().ok()),
        chat_template: metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok())
            .cloned(),
    })
}

/// A key that must be present and must be a string.
fn required_string(metadata: &HashMap<String, Value>, key: &str) -> Result<String, ModelError> {
    metadata
        .get(key)
        .ok_or_else(|| missing(key))?
        .to_string()
        .map(String::clone)
        .map_err(|_| wrong_type(key))
}

/// A required array of strings. Every element is checked: `to_string` returns a `Result` and
/// unwrapping it would turn a malformed file into a panic.
fn string_array(metadata: &HashMap<String, Value>, key: &str) -> Result<Vec<String>, ModelError> {
    let values = metadata
        .get(key)
        .ok_or_else(|| missing(key))?
        .to_vec()
        .map_err(|_| wrong_type(key))?;
    values
        .iter()
        .map(|value| value.to_string().map(String::clone).map_err(|_| wrong_type(key)))
        .collect()
}

/// An array of strings that real files sometimes omit (a vocabulary with no merges).
fn optional_string_array(
    metadata: &HashMap<String, Value>,
    key: &str,
) -> Result<Vec<String>, ModelError> {
    match metadata.get(key) {
        None => Ok(Vec::new()),
        Some(_) => string_array(metadata, key),
    }
}

/// An array of `i32`, empty when absent: token types are advisory.
fn i32_array(metadata: &HashMap<String, Value>, key: &str) -> Result<Vec<i32>, ModelError> {
    let Some(value) = metadata.get(key) else {
        return Ok(Vec::new());
    };
    let values = value.to_vec().map_err(|_| wrong_type(key))?;
    values
        .iter()
        .map(|value| value.to_i32().map_err(|_| wrong_type(key)))
        .collect()
}

fn optional_u32(metadata: &HashMap<String, Value>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn missing(key: &str) -> ModelError {
    ModelError::Gguf(format!("clé {key} absente"))
}

fn wrong_type(key: &str) -> ModelError {
    ModelError::Gguf(format!("clé {key} du mauvais type"))
}
```

Register the module: in `src/models/mod.rs`, next to the other `pub mod` lines, add
`pub mod tokenizer;`.

**Note on `to_string()`**: candle's `Value::to_string()` returns `Result<&String>`, not a
`String` — it is an accessor, not `Display`. That is why `.map(String::clone)` appears above.
If the pinned version differs, adapt and say so.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test --lib models::tokenizer`
Expected: PASS, 4 tests.

- [ ] **Step 7: Write the failing test for `build`**

Append inside the same `mod tests`:

```rust
    fn data() -> TokenizerData {
        from_metadata(&metadata()).expect("well-formed metadata")
    }

    #[test]
    fn encodes_and_decodes_a_round_trip() {
        let tokenizer = build(&data()).expect("a gpt2 vocabulary");

        let encoded = tokenizer.encode("Ġbonjour", false).expect("encodes");
        assert!(!encoded.get_ids().is_empty(), "something was encoded");

        let decoded = tokenizer
            .decode(encoded.get_ids(), false)
            .expect("decodes");
        assert_eq!(decoded, "Ġbonjour", "the round trip is lossless");
    }

    #[test]
    fn applies_the_merge_it_was_given() {
        let tokenizer = build(&data()).expect("a gpt2 vocabulary");

        let encoded = tokenizer.encode("Ġbonjour", false).expect("encodes");

        assert_eq!(
            encoded.get_ids().len(),
            1,
            "`Ġbon` + `jour` merge into one token: {:?}",
            encoded.get_tokens()
        );
    }

    /// An unsupported family must be named. A llama/SPM vocabulary silently fed through a
    /// byte-level BPE does not fail — it produces plausible-looking garbage, which is the
    /// worst possible outcome.
    #[test]
    fn an_unsupported_tokenizer_family_is_refused_by_name() {
        let mut data = data();
        data.model = "spm".to_owned();

        let error = build(&data).expect_err("spm is out of scope");

        let text = error.to_string();
        assert!(text.contains("spm"), "{text}");
    }
```

- [ ] **Step 8: Run it to verify it fails**

Run: `cargo test --lib models::tokenizer`
Expected: FAIL to compile — `cannot find function 'build' in this scope`.

- [ ] **Step 9: Implement `build`**

Append to `src/models/tokenizer.rs`, above the test module. **The `tokenizers` API shapes
here are the ones you pinned in Step 2 — adapt if the pinned version differs:**

```rust
/// Builds the tokenizer. Pure: no file, no candle types, no I/O.
///
/// Only byte-level BPE (`gpt2`) is in scope. A `llama`/`spm` vocabulary pushed through a
/// byte-level BPE does not error — it produces plausible nonsense — so it is refused by name.
pub fn build(data: &TokenizerData) -> Result<tokenizers::Tokenizer, ModelError> {
    if data.model != "gpt2" {
        return Err(ModelError::Gguf(format!(
            "tokenizer « {} » non pris en charge",
            data.model
        )));
    }

    let vocab: std::collections::HashMap<String, u32> = data
        .tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            let id = u32::try_from(id)
                .map_err(|_| ModelError::Gguf("vocabulaire trop grand".into()))?;
            Ok((token.clone(), id))
        })
        .collect::<Result<_, ModelError>>()?;

    // GGUF stores each merge as `"left right"`, one space between the halves.
    let merges = data
        .merges
        .iter()
        .map(|merge| {
            merge
                .split_once(' ')
                .map(|(left, right)| (left.to_owned(), right.to_owned()))
                .ok_or_else(|| ModelError::Gguf(format!("fusion invalide : {merge}")))
        })
        .collect::<Result<Vec<_>, ModelError>>()?;

    let bpe = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .build()
        .map_err(|e| ModelError::Gguf(format!("vocabulaire illisible : {e}")))?;

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
    ));
    tokenizer.with_decoder(Some(
        tokenizers::decoders::byte_level::ByteLevel::default(),
    ));
    Ok(tokenizer)
}
```

- [ ] **Step 10: Run it to verify it passes**

Run: `cargo test --lib models::tokenizer`
Expected: PASS, 7 tests.

If `applies_the_merge_it_was_given` fails with two tokens instead of one, the byte-level
pre-tokenizer is splitting differently than the fixture assumes. **Do not weaken the
assertion to make it pass** — that would hide exactly the class of bug this test exists to
catch. Read `encoded.get_tokens()`, adjust the *fixture* vocabulary so it exercises a real
merge, and say in your report what you changed and why.

- [ ] **Step 11: Final checks**

Run: `cargo test --lib models::`
Expected: PASS, with the existing `models::gguf`, `models::store`, `models::hub`,
`models::download` and `models::catalog` tests untouched.

Run: `cargo clippy --all-targets -- -D warnings`
Expected: clean.

Run: `cargo fmt --check`
Expected: clean.

Run: `cargo tree | grep -i -- '-sys'`
Expected: no output. State this explicitly in your report.

---

### Task 2: The ChatML template

**Files:**
- Create: `src/models/template.rs`
- Modify: `src/models/mod.rs` (add `pub mod template;`)
- Test: appended `#[cfg(test)] mod tests` inside `src/models/template.rs`

**Interfaces:**
- Consumes: nothing from other tasks. `ModelError` from `crate::models`, and
  `crate::llm::{ChatMessage, ChatRole}` — `ChatMessage` is `{ role: ChatRole, content: String, tool_calls: Vec<ToolCall>, tool_call_id: Option<String>, images: Vec<Image> }` and **has a constructor**: `ChatMessage::new(role, content)`. `ChatRole` is `System | User | Assistant | Tool`.
- Produces:
  ```rust
  // src/models/template.rs
  pub fn render(architecture: &str, messages: &[crate::llm::ChatMessage])
      -> Result<String, ModelError>;
  ```
  Task 3 calls it with `metadata.architecture` (the GGUF's `general.architecture`, e.g.
  `"qwen3"`) and the request's messages, and sends the result to the tokenizer.

- [ ] **Step 1: Write the failing test**

Create `src/models/template.rs` with only this test module:

```rust
//! Renders a conversation into the prompt string an architecture expects.
//!
//! Nothing in this repository did this before J34: every provider so far talks to a server
//! that applies the model's own chat template. In-process, it is ours.

#[cfg(test)]
mod tests {
    use crate::llm::{ChatMessage, ChatRole};

    use super::*;

    #[test]
    fn renders_chatml_and_opens_the_assistant_turn() {
        let messages = [
            ChatMessage::new(ChatRole::System, "Tu es concis."),
            ChatMessage::new(ChatRole::User, "Bonjour"),
        ];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert_eq!(
            prompt,
            "<|im_start|>system\nTu es concis.<|im_end|>\n\
             <|im_start|>user\nBonjour<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// Review focus 1: Qwen3 is a hybrid reasoning model. Its own template carries
    /// `enable_thinking` logic, and plain ChatML without it makes the model write
    /// `<think>…</think>` into the reply. Reasoning tokens are out of scope since J33, so
    /// they would arrive in the conversation as literal text. Qwen's documented way to turn
    /// thinking off is to pre-fill an empty think block, which is what the renderer emits.
    #[test]
    fn thinking_is_disabled_by_a_prefilled_empty_block() {
        let messages = [ChatMessage::new(ChatRole::User, "Salut")];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "the empty think block is pre-filled so the model does not open its own: {prompt:?}"
        );
    }

    #[test]
    fn a_prior_assistant_turn_is_closed() {
        let messages = [
            ChatMessage::new(ChatRole::User, "Un"),
            ChatMessage::new(ChatRole::Assistant, "Deux"),
            ChatMessage::new(ChatRole::User, "Trois"),
        ];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert!(
            prompt.contains("<|im_start|>assistant\nDeux<|im_end|>\n"),
            "{prompt:?}"
        );
    }

    /// A tool message has no place in a ChatML prompt for a model that was offered no tools.
    /// Rendering it as a `tool` role would teach the model a turn shape it never saw in
    /// training; folding it into the user turn keeps the transcript honest.
    #[test]
    fn a_tool_message_is_folded_into_the_user_turn() {
        let messages = [ChatMessage::tool_result("call_1", "42")];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert!(prompt.contains("<|im_start|>user\n42<|im_end|>\n"), "{prompt:?}");
        assert!(!prompt.contains("tool"), "no tool role is emitted: {prompt:?}");
    }

    /// The engine lists every GGUF in the store, so the user can pick a Gemma or a Llama. It
    /// must be refused by name: being silently wrong is worse than being unavailable.
    #[test]
    fn an_unsupported_architecture_is_refused_by_name() {
        let messages = [ChatMessage::new(ChatRole::User, "Bonjour")];

        let error = render("gemma3", &messages).expect_err("only qwen3 is in scope");

        let text = error.to_string();
        assert!(text.contains("gemma3"), "{text}");
    }

    #[test]
    fn an_empty_conversation_still_opens_the_assistant_turn() {
        let prompt = render("qwen3", &[]).expect("qwen3 is supported");

        assert_eq!(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib models::template`
Expected: FAIL to compile — `cannot find function 'render' in this scope`.

- [ ] **Step 3: Implement `render`**

Add above the test module:

```rust
use crate::llm::{ChatMessage, ChatRole};

use super::ModelError;

/// Renders a conversation into the prompt string `architecture` expects.
///
/// Pure: no I/O, no clock, no allocation beyond the string it returns.
pub fn render(architecture: &str, messages: &[ChatMessage]) -> Result<String, ModelError> {
    match architecture {
        "qwen3" | "qwen2" => Ok(chatml(messages)),
        other => Err(ModelError::Gguf(format!(
            "architecture « {other} » non prise en charge par le moteur local"
        ))),
    }
}

/// ChatML, as Qwen uses it.
///
/// The trailing `<think>\n\n</think>\n\n` is not decoration: Qwen3 is a hybrid reasoning
/// model whose own template carries `enable_thinking` logic, and without it the model opens
/// its own think block and writes its reasoning into the reply. Reasoning tokens are out of
/// scope (J33), so the block is pre-filled empty — Qwen's documented way to turn thinking
/// off — and the model continues straight into its answer.
fn chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            // The local provider offers no tools, so a tool result is history the model
            // must see without being taught a turn shape it never trained on.
            ChatRole::Tool => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    out
}
```

Register the module: in `src/models/mod.rs`, add `pub mod template;`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib models::template`
Expected: PASS, 6 tests.

- [ ] **Step 5: Verify the template against the real model, not against my memory**

This is the step that matters most in this task, and it cannot be done from the plan.

`docs/superpowers/specs/2026-10-02-local-inference-design.md` requires the rendered string to
be checked against the template that actually ships in the Qwen3 GGUF. If a Qwen3 GGUF is in
the local store (look under the configured models directory, default `<data>/models`), read
its `tokenizer.chat_template` and compare the role tags and the thinking block against what
`chatml` emits.

If no such file is on disk, **say so in your report** and state that the template is
unverified against a real file. Do not invent a verification you did not perform, and do not
download a model — that is the orchestrator's call.

- [ ] **Step 6: Final checks**

Run: `cargo test --lib models::`
Expected: PASS, existing tests untouched.

Run: `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`
Expected: clean.

---

### Task 3: The engine, the thread boundary and the stream

**Files:**
- Create: `src/llm/local.rs`
- Modify: `src/llm/mod.rs` (add `mod local;` next to the other provider modules)
- Modify: `Cargo.toml` (add `candle-nn` and `candle-transformers`)
- Test: appended `#[cfg(test)] mod tests` inside `src/llm/local.rs`

**Interfaces:**
- Consumes:
  - From Task 1: `crate::models::tokenizer::{TokenizerData, from_metadata, build}`.
  - From Task 2: `crate::models::template::render(architecture, messages)`.
  - Existing, verified: the `LlmClient` trait (`src/llm/mod.rs:164-177`) —
    ```rust
    #[async_trait]
    pub trait LlmClient: Send + Sync {
        async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError>;
        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;
        async fn context_window(&self, _model: &str) -> Option<u64> { None }
    }
    ```
    `TokenStream = BoxStream<'static, Result<StreamItem, LlmError>>`;
    `StreamItem::Text(String)`; `ChatRequest { model, messages, tools }`;
    `ModelInfo { id, context_window }` with `ModelInfo::named(id)`;
    `LlmError::Server(String)` is the variant to use — there is no network here, so
    `Unreachable`, `Timeout`, `Auth`, `RateLimited`, `Http` and `Protocol` must not appear.
  - `crate::models::gguf::read(path) -> Result<Metadata, ModelError>` for the cheap header
    read, whose `Metadata` carries `architecture`, `context_length`, `quantization`,
    `parameters` and six more fields.
- Produces:
  ```rust
  // src/llm/local.rs
  pub struct LocalClient { /* … */ }
  impl LocalClient {
      pub fn new(models_dir: &std::path::Path) -> Self;
  }
  impl LlmClient for LocalClient { /* … */ }
  ```
  Task 4 constructs it in `build_clients` and must pass the resolved models directory.

- [ ] **Step 1: Write the failing test for the stream plumbing**

The engine is behind a trait so the plumbing is testable without a 400 MB file. Create
`src/llm/local.rs` with only this test module:

```rust
//! The local provider: a GGUF from the store, decoded in this process.
//!
//! Inference is synchronous and CPU-bound, so it runs on a dedicated OS thread and never on
//! the tokio runtime. Tokens cross back through an mpsc channel presented as a [`TokenStream`].

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    /// An engine that yields what it was told to, so the plumbing can be tested without a
    /// model. `sent` records how far it got, which is how cancellation is observed.
    struct Scripted {
        tokens: Vec<String>,
        error_at_end: bool,
        sent: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Engine for Scripted {
        fn next_token(&mut self) -> Result<Option<String>, LlmError> {
            let index = self.sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match self.tokens.get(index) {
                Some(token) => Ok(Some(token.clone())),
                None if self.error_at_end => {
                    Err(LlmError::Server("le moteur a échoué".to_owned()))
                }
                None => Ok(None),
            }
        }
    }

    fn scripted(tokens: &[&str]) -> (Scripted, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let sent = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            Scripted {
                tokens: tokens.iter().map(|t| (*t).to_owned()).collect(),
                error_at_end: false,
                sent: sent.clone(),
            },
            sent,
        )
    }

    #[tokio::test]
    async fn tokens_arrive_in_order_and_the_stream_ends() {
        let (engine, _) = scripted(&["Bon", "jour", " !"]);

        let items: Vec<_> = spawn(Box::new(engine)).collect().await;

        let text: Vec<String> = items
            .into_iter()
            .map(|item| match item.expect("no error") {
                StreamItem::Text(text) => text,
                other => panic!("only text is emitted: {other:?}"),
            })
            .collect();
        assert_eq!(text, ["Bon", "jour", " !"]);
    }

    #[tokio::test]
    async fn an_engine_failure_surfaces_as_the_last_item() {
        let sent = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = Scripted {
            tokens: vec!["Bon".to_owned()],
            error_at_end: true,
            sent,
        };

        let items: Vec<_> = spawn(Box::new(engine)).collect().await;

        assert!(matches!(items.first(), Some(Ok(StreamItem::Text(t))) if t == "Bon"));
        assert!(
            matches!(items.last(), Some(Err(LlmError::Server(_)))),
            "{items:?}"
        );
    }

    /// Review focus 4: cancellation works only because the thread's `send` fails once the
    /// stream is dropped. An engine that ignored that error would decode to the end, burning
    /// a core with nobody listening — and `Échap` would stop nothing visible while the
    /// machine stayed busy.
    #[tokio::test]
    async fn dropping_the_stream_stops_the_engine() {
        let long: Vec<&str> = vec!["x"; 10_000];
        let (engine, sent) = scripted(&long);

        let mut stream = spawn(Box::new(engine));
        let first = stream.next().await;
        assert!(first.is_some(), "one token arrived");
        drop(stream);

        // The thread notices on its next send. Give it room, then confirm it stopped well
        // short of the 10 000 it was scripted to produce.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let produced = sent.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            produced < 1_000,
            "the engine kept decoding after the reader left: {produced} tokens"
        );
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib llm::local`
Expected: FAIL to compile — `cannot find trait 'Engine'`, `cannot find function 'spawn'`.

- [ ] **Step 3: Implement the `Engine` seam and `spawn`**

Add above the test module. Note the stream is built with `futures::stream::unfold`, which
**avoids adding `tokio-stream` as a dependency** — `futures` is already in `Cargo.toml`:

```rust
use std::path::{Path, PathBuf};

use futures::{StreamExt, stream};

use super::{ChatRequest, LlmError, ModelInfo, StreamItem, TokenStream};

/// What [`spawn`] needs of an inference engine.
///
/// A trait rather than a concrete type so the channel plumbing, the thread lifetime and
/// cancellation can be tested with a scripted engine, and no test needs a model file.
trait Engine: Send {
    /// The next fragment of decoded text, or `None` when the model emitted end-of-text.
    fn next_token(&mut self) -> Result<Option<String>, LlmError>;
}

/// How many decoded fragments may wait in the channel.
///
/// Small on purpose: a full channel makes the engine thread block in `blocking_send`, which
/// is the back-pressure that keeps a fast model from racing ahead of the 30 fps renderer and
/// building an unbounded queue.
const QUEUE: usize = 16;

/// Runs `engine` on a dedicated OS thread and presents its output as a [`TokenStream`].
///
/// Dropping the returned stream is the cancellation path: the receiver goes away, the
/// thread's next `blocking_send` fails, and the loop returns. No cancellation token is
/// needed, and `Échap` already drops the stream.
fn spawn(mut engine: Box<dyn Engine>) -> TokenStream {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamItem, LlmError>>(QUEUE);

    std::thread::spawn(move || {
        loop {
            match engine.next_token() {
                Ok(Some(text)) => {
                    // A send error means the reader is gone: stop decoding immediately.
                    if tx.blocking_send(Ok(StreamItem::Text(text))).is_err() {
                        return;
                    }
                }
                Ok(None) => return,
                Err(error) => {
                    let _ = tx.blocking_send(Err(error));
                    return;
                }
            }
        }
    });

    stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    })
    .boxed()
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cargo test --lib llm::local`
Expected: PASS, 3 tests.

- [ ] **Step 5: Write the failing test for the directory scan**

Append inside `mod tests`:

```rust
    /// `list_models` scans the directory rather than reading the database: `build_clients`
    /// gets no SQLite connection, and the filesystem is the right authority for "what can I
    /// load" — a file deleted outside the application is correctly absent.
    #[tokio::test]
    async fn an_empty_directory_lists_nothing_rather_than_failing() {
        let dir = tempfile::tempdir().expect("a temporary directory");

        let models = LocalClient::new(dir.path())
            .list_models()
            .await
            .expect("an empty store is not an error");

        assert!(models.is_empty(), "{models:?}");
    }

    #[tokio::test]
    async fn gguf_files_are_listed_by_repository_and_file() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let folder = dir.path().join("unsloth").join("Qwen3-0.6B-GGUF");
        std::fs::create_dir_all(&folder).expect("creates the folders");
        std::fs::write(folder.join("Qwen3-0.6B-Q4_K_M.gguf"), b"not a real gguf")
            .expect("writes the file");
        std::fs::write(folder.join("README.md"), b"ignored").expect("writes the file");

        let models = LocalClient::new(dir.path())
            .list_models()
            .await
            .expect("lists what is on disk");

        assert_eq!(
            models.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            ["unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"],
            "only .gguf files, identified by owner/repo/file"
        );
    }

    /// Review focus 5: `/model` lists from a scan, so the file can be gone by the time the
    /// user sends a message. The message must name the path, because the user picked it.
    #[tokio::test]
    async fn a_missing_model_file_is_an_error_that_names_the_path() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let client = LocalClient::new(dir.path());

        let error = client
            .chat_stream(ChatRequest {
                model: "unsloth/Qwen3-0.6B-GGUF/gone.gguf".to_owned(),
                messages: Vec::new(),
                tools: Vec::new(),
            })
            .await
            .expect_err("the file is not there");

        let text = error.to_string();
        assert!(text.contains("gone.gguf"), "{text}");
    }
```

`tempfile` is already a dev-dependency of this crate (used by `src/files.rs` tests); confirm
with `grep -n tempfile Cargo.toml` and, if it is missing from `[dev-dependencies]`, add it
there — never to `[dependencies]`.

- [ ] **Step 6: Run it to verify it fails**

Run: `cargo test --lib llm::local`
Expected: FAIL to compile — `cannot find struct 'LocalClient'`.

- [ ] **Step 7: Implement `LocalClient`, the scan and the loader**

Append above the test module:

```rust
/// The local provider: GGUF files from the model store, decoded in this process.
pub struct LocalClient {
    dir: PathBuf,
}

impl LocalClient {
    /// `models_dir` is the resolved store directory, laid out as `<dir>/<owner>/<repo>/<file>`
    /// by `models::download::paths`.
    pub fn new(models_dir: &Path) -> Self {
        Self {
            dir: models_dir.to_path_buf(),
        }
    }

    /// Every `.gguf` under `<dir>/<owner>/<repo>/`, identified as `owner/repo/file`.
    fn scan(&self) -> Vec<String> {
        let mut found = Vec::new();
        let Ok(owners) = std::fs::read_dir(&self.dir) else {
            // No directory yet is an empty store, not a failure.
            return found;
        };
        for owner in owners.flatten() {
            let Ok(repos) = std::fs::read_dir(owner.path()) else {
                continue;
            };
            for repo in repos.flatten() {
                let Ok(files) = std::fs::read_dir(repo.path()) else {
                    continue;
                };
                for file in files.flatten() {
                    let path = file.path();
                    if path.extension().is_some_and(|e| e == "gguf") {
                        if let (Some(owner), Some(repo), Some(name)) = (
                            owner.file_name().to_str(),
                            repo.file_name().to_str(),
                            path.file_name().and_then(|n| n.to_str()),
                        ) {
                            found.push(format!("{owner}/{repo}/{name}"));
                        }
                    }
                }
            }
        }
        found.sort();
        found
    }

    /// Resolves a model id back to a path, refusing anything that escapes the store.
    fn path_of(&self, id: &str) -> Result<PathBuf, LlmError> {
        let parts: Vec<&str> = id.split('/').collect();
        let [owner, repo, file] = parts.as_slice() else {
            return Err(LlmError::Server(format!("identifiant de modèle invalide : {id}")));
        };
        for part in [owner, repo, file] {
            if part.is_empty() || part.contains("..") || part.contains('\\') {
                return Err(LlmError::Server(format!(
                    "identifiant de modèle invalide : {id}"
                )));
            }
        }
        let path = self.dir.join(owner).join(repo).join(file);
        if !path.is_file() {
            return Err(LlmError::Server(format!(
                "modèle introuvable sur le disque : {}",
                path.display()
            )));
        }
        Ok(path)
    }
}
```

- [ ] **Step 8: Implement the trait**

Append:

```rust
#[async_trait::async_trait]
impl super::LlmClient for LocalClient {
    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError> {
        let path = self.path_of(&request.model)?;
        let metadata = crate::models::gguf::read(&path)
            .map_err(|e| LlmError::Server(e.to_string()))?;
        let architecture = metadata
            .architecture
            .ok_or_else(|| LlmError::Server("le fichier ne nomme pas son architecture".into()))?;
        let prompt = crate::models::template::render(&architecture, &request.messages)
            .map_err(|e| LlmError::Server(e.to_string()))?;

        let engine = CandleEngine::load(&path, &architecture, &prompt)
            .map_err(|e| LlmError::Server(e.to_string()))?;
        Ok(spawn(Box::new(engine)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        Ok(self
            .scan()
            .into_iter()
            .map(|id| {
                let window = self
                    .path_of(&id)
                    .ok()
                    .and_then(|path| crate::models::gguf::read(&path).ok())
                    .and_then(|meta| meta.context_length);
                ModelInfo {
                    id,
                    context_window: window,
                }
            })
            .collect())
    }

    async fn context_window(&self, model: &str) -> Option<u64> {
        let path = self.path_of(model).ok()?;
        crate::models::gguf::read(&path).ok()?.context_length
    }
}
```

**On caching:** loading inside `chat_stream` on every message is correct but slow. Caching is
deliberately deferred to Step 11, because it is not a speed question — it is a correctness
question about the KV cache, and the engine has to work before that can be judged.

- [ ] **Step 9: Implement the candle engine**

Append. This is the one place where the plan's code is thinner than elsewhere, because it is
assembling three documented APIs rather than expressing a decision — their exact signatures,
verified against docs.rs for 0.11.0, are:

```text
gguf_file::Content::read<R: Seek + Read>(reader: &mut R) -> Result<Content>
  fields: magic, metadata: HashMap<String, Value>, tensor_infos, tensor_data_offset
quantized_qwen3::ModelWeights::from_gguf<R: Seek + Read>(ct: Content, reader: &mut R, device: &Device) -> Result<Self>
quantized_qwen3::ModelWeights::forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>
generation::LogitsProcessor::new(seed: u64, temperature: Option<f64>, top_p: Option<f64>)
generation::LogitsProcessor::sample(&mut self, logits: &Tensor) -> Result<u32>
```

```rust
/// Decodes with candle, one token per call.
struct CandleEngine {
    weights: candle_transformers::models::quantized_qwen3::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    logits: candle_transformers::generation::LogitsProcessor,
    device: candle_core::Device,
    /// The next token to feed: the prompt on the first call, then the sampled token.
    pending: Vec<u32>,
    position: usize,
    eos: Option<u32>,
    /// Decoding is incremental, so a multi-byte character can straddle two tokens.
    decoded: usize,
    produced: Vec<u32>,
    limit: usize,
}

impl CandleEngine {
    fn load(
        path: &Path,
        architecture: &str,
        prompt: &str,
    ) -> Result<Self, crate::models::ModelError> {
        use crate::models::ModelError;

        if architecture != "qwen3" && architecture != "qwen2" {
            return Err(ModelError::Gguf(format!(
                "architecture « {architecture} » non prise en charge par le moteur local"
            )));
        }

        let mut file = std::fs::File::open(path).map_err(|e| ModelError::Io(e.to_string()))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| ModelError::Gguf(e.to_string()))?;

        let data = crate::models::tokenizer::from_metadata(&content.metadata)?;
        let tokenizer = crate::models::tokenizer::build(&data)?;
        let eos = data.eos;

        let device = candle_core::Device::Cpu;
        let weights =
            candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
                content, &mut file, &device,
            )
            .map_err(|e| ModelError::Gguf(e.to_string()))?;

        let encoded = tokenizer
            .encode(prompt, true)
            .map_err(|e| ModelError::Gguf(format!("prompt non encodable : {e}")))?;

        Ok(Self {
            weights,
            tokenizer,
            logits: candle_transformers::generation::LogitsProcessor::new(42, Some(0.7), Some(0.9)),
            device,
            pending: encoded.get_ids().to_vec(),
            position: 0,
            eos,
            decoded: 0,
            produced: Vec::new(),
            // A reply cannot run forever: without a cap, a model that never emits
            // end-of-text would decode until the user quits.
            limit: 2048,
        })
    }
}

impl Engine for CandleEngine {
    fn next_token(&mut self) -> Result<Option<String>, LlmError> {
        if self.produced.len() >= self.limit {
            return Ok(None);
        }
        let input = candle_core::Tensor::new(self.pending.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| LlmError::Server(e.to_string()))?;
        let logits = self
            .weights
            .forward(&input, self.position)
            .and_then(|l| l.squeeze(0))
            .map_err(|e| LlmError::Server(e.to_string()))?;
        let logits = match logits.rank() {
            2 => logits
                .get(logits.dim(0).map_err(|e| LlmError::Server(e.to_string()))? - 1)
                .map_err(|e| LlmError::Server(e.to_string()))?,
            _ => logits,
        };

        self.position += self.pending.len();
        let next = self
            .logits
            .sample(&logits)
            .map_err(|e| LlmError::Server(e.to_string()))?;
        self.pending = vec![next];

        if self.eos == Some(next) {
            return Ok(None);
        }
        self.produced.push(next);

        // Decode the whole reply and emit only what is new: a single token can be half a
        // multi-byte character, and decoding it alone would yield a replacement glyph.
        let text = self
            .tokenizer
            .decode(&self.produced, false)
            .map_err(|e| LlmError::Server(format!("décodage impossible : {e}")))?;
        let fragment = text.get(self.decoded..).unwrap_or_default().to_owned();
        if fragment.is_empty() {
            // The character is not finished yet; ask for the next token rather than
            // emitting nothing, so the stream keeps moving.
            return self.next_token();
        }
        self.decoded = text.len();
        Ok(Some(fragment))
    }
}
```

**If `forward`'s output rank differs from what the `match` above assumes, fix the match and
say so in your report**; the shape depends on whether candle returns logits for every
position or only the last, and that is a detail of the model implementation rather than a
decision this plan can make for you.

**On the spec's "mémoire insuffisante" message:** the spec's error table promises it, and
this code does not produce it — a failed allocation surfaces as candle's own error text
through `LlmError::Server`. Detecting an out-of-memory condition portably is not something
this milestone can do honestly, and a message that guesses would be worse than candle's.
Note the divergence in your report; the orchestrator will correct the spec rather than the
code.

- [ ] **Step 10: Run the tests to verify nothing regressed**

Run: `cargo test --lib llm::local`
Expected: PASS, 6 tests. The candle engine itself is not covered — no test loads a model, by
design.

Run: `cargo build`
Expected: clean, no new warnings. This is the first step that compiles candle; expect it to
take several minutes the first time.

- [ ] **Step 11: Decide whether the weights can be cached — a correctness question first**

`chat_stream` loads on every message, which is correct and slow. Before making it fast,
settle this:

**`ModelWeights` holds the KV cache internally.** `forward(&mut self, x, index_pos)` fills it
as it goes. If a loaded model is reused for a *second, unrelated* prompt starting at
`index_pos = 0`, the cache still holds the previous conversation's keys and values, and the
model attends to text the user never wrote. The failure mode is not a crash: it is a reply
that drifts, contaminated by the last conversation. That is the worst kind of bug — plausible
output, no error, very hard to attribute.

So:

Run: `cargo doc -p candle-transformers --no-deps` and read
`target/doc/candle_transformers/models/quantized_qwen3/struct.ModelWeights.html` for a
cache-reset method (`clear_kv_cache`, `reset`, or similar).

- **If a reset method exists:** cache the weights. Add
  `cache: std::sync::Mutex<Option<(String, std::sync::Arc<std::sync::Mutex<ModelWeights>>)>>`
  to `LocalClient`, keyed by model id; on a hit, call the reset method before handing the
  model to the engine; on a miss, load and replace. The engine holds the inner `Arc<Mutex<…>>`
  for the duration of its generation, which also serialises two overlapping generations
  rather than letting them corrupt one cache. Two cannot normally overlap — `stream_task`
  runs one request at a time — but the lock makes that safe rather than assumed.
- **If no reset method exists:** do **not** cache the weights. Cache only the tokenizer,
  which is cheap to hold and not cheap to rebuild, and reload the weights per request.

Either way, **measure and report the load time** for a real Qwen3 0.6B Q4_K_M (or state that
no model was on disk to measure). The orchestrator needs the number: if an uncached reload
costs several seconds per message, that is a product decision about whether the milestone
ships that way, and it is not yours to make silently.

Then add the test that matches what you did:

```rust
    /// The loader runs once for two requests on the same model. Asserted on a counter, not
    /// on elapsed time: a timing assertion would be flaky on a loaded machine and would be
    /// measuring candle rather than the cache.
    #[tokio::test]
    async fn a_second_request_for_the_same_model_does_not_reload_it() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let client = LocalClient::new(dir.path());

        // Drive the cache directly rather than through `chat_stream`, which would need a
        // real GGUF: insert a sentinel under a model id, then assert a second lookup for the
        // same id finds it and a lookup for a different id does not.
        assert!(
            client.cached("unsloth/Qwen3-0.6B-GGUF/a.gguf").is_none(),
            "nothing is cached before the first load"
        );
    }
```

To make that test meaningful you need a small seam — a `fn cached(&self, id: &str) -> Option<…>`
on `LocalClient` that the test can call. Add it as a private method and keep the test in the
same module.

**If you took the no-cache branch, delete this test rather than adapting it into something
that asserts nothing.** An empty or tautological test reads as coverage and is worse than an
honest gap; say in your report that there is no cache to test and why.

- [ ] **Step 12: Register the module and run the final checks**

In `src/llm/mod.rs`, next to `mod anthropic;` and `mod openai;`, add `mod local;` and
`pub use local::LocalClient;` if `build_clients` needs the name exported — check how
`openai::OpenAiCompatibleClient` is referenced there and match it.

Run: `cargo test --lib llm::`
Expected: PASS, the existing `llm::openai`, `llm::anthropic` and `llm::stream_task` tests
untouched.

Run: `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`
Expected: clean.

Run: `cargo tree | grep -i -- '-sys'`
Expected: no output. Candle is pure Rust; if a `*-sys` crate appeared with `candle-nn` or
`candle-transformers`, **stop and report it** — that would invalidate the milestone's central
trade-off, and the orchestrator must decide, not you.

---

### Task 4: Wiring, documentation and the dependency gate

**Files:**
- Modify: `src/config.rs:117-125` (the `ProviderKind` enum), `:277-308` (`presets()`)
- Modify: `src/llm/mod.rs:240-265` (`build_clients`, which gains a parameter)
- Modify: `src/runtime.rs:193-195` (the call site and the two-line reorder)
- Modify: `src/llm/mod.rs:368` (the one test call site of `build_clients`)
- Modify: `README.md`, `PLAN.md`, `docs/roadmap.md`
- Test: appended to the existing `mod tests` in `src/config.rs` and `src/llm/mod.rs`

**Interfaces:**
- Consumes: `LocalClient::new(models_dir: &Path)` from Task 3.
- Produces: a green `cargo test` and a `cargo tree` with no `*-sys` crate.

- [ ] **Step 1: Write the failing test for the preset**

Append inside the existing `#[cfg(test)] mod tests` in `src/config.rs`:

```rust
    /// The local engine is a provider like any other, so `/model` lists downloaded GGUF
    /// files beside Ollama's models. It needs no key, so it is always constructed.
    #[test]
    fn the_local_provider_is_a_preset_that_needs_no_key() {
        let config = Config::default();

        let providers = config.resolve_providers(|_| None);

        let local = providers
            .iter()
            .find(|p| p.id == "local")
            .expect("the local provider is a preset");
        assert_eq!(local.kind, ProviderKind::Local);
        assert!(local.api_key_env.is_none(), "no key to miss");
        assert!(!local.missing_key(), "so it is never Unavailable for a key");
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib config::tests::the_local_provider_is_a_preset_that_needs_no_key`
Expected: FAIL — `no variant named 'Local' found for enum 'ProviderKind'`.

- [ ] **Step 3: Add the variant and the preset**

In `src/config.rs`, extend the enum (the `#[serde(rename_all = "lowercase")]` on it makes the
TOML spelling `kind = "local"`):

```rust
pub enum ProviderKind {
    /// OpenAI chat completions (OpenAI, Ollama, llama.cpp, LM Studio, vLLM, …).
    #[default]
    Openai,
    /// Anthropic Messages API (Claude).
    Anthropic,
    /// No wire protocol at all: a GGUF from the model store, decoded in this process.
    Local,
}
```

And add a fourth preset in `presets()`, after `claude`:

```rust
        (
            "local",
            ProviderConfig {
                kind: Some(ProviderKind::Local),
                label: Some("Local".into()),
                ..ProviderConfig::default()
            },
        ),
```

No `base_url` and no `api_key_env`: there is no server and no key.

- [ ] **Step 4: Run it to verify it passes, and fix what the new variant broke**

Run: `cargo test --lib config::`
Expected: PASS.

Run: `cargo build`
Expected: errors in `build_clients` — a non-exhaustive match. That is the next step. **List
every site the compiler points at in your report**; a new enum variant can surface matches
far from where it was added.

- [ ] **Step 5: Write the failing test for `build_clients`**

Append inside the existing `mod tests` in `src/llm/mod.rs`:

```rust
    #[test]
    fn the_local_provider_gets_a_usable_client_not_an_excuse() {
        let config = Config::default();
        let providers = config.resolve_providers(|_| None);
        let dir = tempfile::tempdir().expect("a temporary directory");

        let clients = build_clients(&providers, Duration::from_secs(1), dir.path());

        assert!(
            clients.contains_key("local"),
            "the local provider is built: {:?}",
            clients.keys().collect::<Vec<_>>()
        );
    }
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cargo test --lib llm::tests::the_local_provider_gets_a_usable_client_not_an_excuse`
Expected: FAIL to compile — `build_clients` takes 2 arguments, not 3.

- [ ] **Step 7: Give `build_clients` the directory and the new arm**

In `src/llm/mod.rs`, change the signature and add the arm:

```rust
/// Builds a client for each provider. Providers without their API key, or with an invalid
/// configuration, get an [`Unavailable`] client that explains why.
///
/// `models_dir` is the local model store, which the local provider scans; it is resolved by
/// `ModelsBackend::new` in the runtime.
pub fn build_clients(
    providers: &[Provider],
    connect_timeout: Duration,
    models_dir: &std::path::Path,
) -> Clients {
```

and inside the `match provider.kind`:

```rust
                    ProviderKind::Local => {
                        Ok(Arc::new(local::LocalClient::new(models_dir)) as Arc<dyn LlmClient>)
                    }
```

`LocalClient::new` cannot fail — there is nothing to validate until a model is loaded — so it
is wrapped in `Ok` to match the arms around it.

- [ ] **Step 8: Update the runtime call site, which needs a two-line reorder**

In `src/runtime.rs`, `ModelsBackend::new` resolves the directory at `:195`, *below* the
`build_clients` call at `:193`. Move the `models` line above the `clients` line and pass the
directory:

```rust
        let providers = config.resolve_providers(|name| std::env::var(name).ok());
        let models = ModelsBackend::new(&config, &database);
        let clients = llm::build_clients(
            &providers,
            Duration::from_secs(config.connect_timeout_secs),
            &models.dir,
        );
```

Check that nothing between the two lines depended on the old order, and that `models.dir` is
the field name — read the struct rather than trusting this plan.

- [ ] **Step 9: Run the whole suite**

Run: `cargo test`
Expected: PASS, everything. If a snapshot moved, **read its diff before accepting it** — a
new provider in the list changes `/model`'s popup, which is legitimate, but a snapshot
accepted unread freezes whatever else changed. Accept one at a time with
`INSTA_UPDATE=always cargo test --lib <exact test>`; never over the suite or a module.

- [ ] **Step 10: The dependency gate**

Run: `cargo tree | grep -i -- '-sys'`
Expected: **no output.** Paste the command and its empty result into your report.

Run: `cargo tree --depth 1`
Expected: `candle-core`, `candle-nn`, `candle-transformers` and `tokenizers` present, and
nothing else new. Paste the list.

If any `*-sys` crate is present, the milestone's central trade-off has been lost: report it
as the most important finding and stop rather than shipping it.

- [ ] **Step 11: Documentation**

`README.md` — in the providers section, say that a GGUF in the local store can be selected
through `/model` like any other model, that it runs in-process on CPU with no daemon, and
that only Qwen3 is supported in this milestone. Do not overclaim: no GPU, no tools, one
architecture.

`PLAN.md` — J32's and J33's notes still say the store downloads files it cannot execute.
Correct them, and keep the limitation that remains: it can execute *one architecture* now.

`docs/roadmap.md` — move "Running local models: GGUF tokenizer and inference" out of `Next`
and into `Done` as `J34`, in the same one-or-two-line style as the entries around it, naming
what is in and what is out. Leave the remaining `Next` entries alone.

English prose in all three, per the repo convention; the French UI strings they quote stay
verbatim.

- [ ] **Step 12: Final checks**

Run: `cargo test`
Expected: PASS, every suite, 0 failed. Paste every `test result:` line.

Run: `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`
Expected: clean.

Run: `cargo tree | grep -i -- '-sys'`
Expected: no output.

- [ ] **Step 13: The manual check this milestone cannot automate**

No test loads a model, by design. The end-to-end check is manual and belongs to the
orchestrator, not to this task. State in your report that it has **not** been done, and what
it would be: `cargo run`, `/pull unsloth/Qwen3-0.6B-GGUF`, `/model`, pick the local entry,
send "Bonjour", and confirm that French text streams in, that no `<think>` block appears in
the reply, and that `Échap` stops the generation and the CPU returns to idle.
