# J34 — Local inference: running the GGUF files the store already downloads

## Why

J32 gave chatatui a model store: `/pull` fetches a GGUF from HuggingFace, verifies it,
reads its metadata and records it. `/models` browses what is on disk alongside a catalogue
of known repositories. The milestone shipped with one honest limitation, stated in the
application itself: **nothing runs those files.** The store downloads models it cannot
execute.

This milestone closes that gap for one model. The goal is stated as independence, not
performance and not completeness: **chatatui alone suffices.** You run the binary, you
`/pull` a model, it answers. No daemon to install, no second file to fetch — one GGUF is
the whole input.

That framing matters because the default provider today is `ollama`, which wraps llama.cpp.
A user who already runs Ollama can already run models. What this milestone adds is the
removal of that requirement.

## Scope

**In scope.** Qwen3 0.6B in Q4_K_M, loaded from the local store, answering in streaming in
the TUI, on CPU, selected through `/model` like any other provider's model.

**Out of scope, deliberately:**

- **GPU acceleration.** CPU only. A later milestone, behind a feature flag.
- **Any architecture other than Qwen3.** The engine names the architecture it was given and
  refuses the rest, in words, rather than producing noise.
- **Tool calling from a local model.** The local provider offers no tool specifications, so
  the model never emits a call. `/tools on` with a local model simply means the model has no
  tools — which is true and visible, not a silent failure.
- **Exact token counting.** `src/tokens.rs` keeps its four-characters-per-token estimate.
  A real tokenizer makes exact counts *possible* and that is a later, separate change.
- **More than one model resident in memory.** One at a time.
- **Interpreting the GGUF's own `tokenizer.chat_template`.** It is a Jinja template; a Jinja
  engine is its own milestone. See "The template" below for what we do instead.

## What already exists, precisely

This section is the inventory the design argues from. Every claim here was read out of the
code, not remembered.

### The provider contract is small

`src/llm/mod.rs:164-177`:

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;
    async fn context_window(&self, _model: &str) -> Option<u64> { None }
}
```

- `TokenStream = BoxStream<'static, Result<StreamItem, LlmError>>` (`src/llm/mod.rs:134`).
- `StreamItem` (`:118-131`) is `Text(String)`, `Usage(Usage)`, or
  `ToolCallDelta { index, id, name, arguments }`.
- `ModelInfo { id: String, context_window: Option<u64> }` (`:137-152`).
- `context_window` has a default returning `None`, so it is optional.

Three facts make a local backend cheap to add:

1. **`Usage` is optional.** `src/llm/stream_task.rs` forwards it if it arrives and does
   nothing if it never does; no consumer requires it.
2. **The waiting phases are not the backend's job.** `Phase` (`src/llm/mod.rs:274-283`) is
   synthesized by `stream_task.rs` around its own `.await` points — `Retrieving` before
   context, `Connecting` before `chat_stream`, `Waiting` once the stream opens. A backend
   emits none of them and still animates correctly. **This is J33 paying for itself:** model
   loading takes seconds, and the waiting line already covers it with no new code.
3. **Tool-call ids are synthesized if absent** (`stream_task.rs:242-246`), and
   `MAX_TOOL_ROUNDS` is 8 (`:82`).

The one structural obstacle: **`ProviderKind` is a closed two-variant enum** — `Openai`,
`Anthropic` (`src/config.rs:117-125`) — and `build_clients` (`src/llm/mod.rs:240-265`)
matches on it exhaustively. A backend that speaks neither wire format must add a variant
and an arm. (An OpenAI-compatible *server* needs neither: that is already the documented
`[providers.<id>]` path. We are not adding a server.)

### The GGUF parser stops short of what a tokenizer needs

`src/models/gguf.rs` walks the header and the full tensor-descriptor table, but:

- **The vocabulary is located, not stored.** `tokenizer.ggml.tokens`, `.merges`, `.scores`
  and `.token_type` are GGUF arrays; `read_unsigned` returns `None` for the array kind, so
  they reach `skip_value` (`:221-238`), where string elements are stepped over one at a time
  (`:228-234`) and fixed-width arrays are a single seek (`:224-226`). The cursor moves; no
  bytes are copied.
- **`tokenizer.chat_template` is never read into memory at all** — it is a string value that
  falls to `skip_value`'s string branch (`:217-220`).
- **Special-token ids are read and thrown away.** `tokenizer.ggml.bos_token_id`,
  `eos_token_id`, `unknown_token_id`, `padding_token_id` and `add_bos_token` are captured by
  the generic integer arm into a `suffixed: Vec<(String, u64)>` (`:90-93`) which is only
  consulted for five `{arch}.*` suffixes — so they are parsed and then dropped.
- **Tensor names and data offsets are read and discarded** (`:100`, `:110`), and
  `general.alignment` is dropped like any other non-`{arch}` integer. The absolute start of
  the tensor-data section is therefore never computed. This does not block us: candle has its
  own GGUF reader for weights (see "Two GGUF readers" below).

The allocation bounds — `MAX_COUNT = 1<<20`, `MAX_STRING = 1<<20`, `MAX_ARRAY = 1<<24`
(`:21-26`, enforced by `bounded()` at `:255-260`) — are checked **before** any allocation,
and already cover the arrays we are about to start reading.

`Metadata` (`:30-47`) computes ten fields. `src/models/download.rs:163-166` keeps four of
them, so **`LocalModel` does not record which tokenizer family a model uses**, even though
`gguf.rs` captured it. An engine reading `LocalModel` alone could not know whether to build
a BPE or a SentencePiece tokenizer.

### There is no chat template anywhere in the repository

`src/prompt.rs` assembles a `Vec<ChatMessage>` for an HTTP request. It does not render a
token string: no Jinja, no per-architecture role tags, nothing that emits `<|im_start|>` or
`[INST]`. Today the server does this. In-process, it becomes ours.

`src/tokens.rs` counts by estimate — `chars.div_ceil(4)` plus a flat four per message
(`:10-21`) — and says so in its own doc comment.

## Approach

### Why candle, and what it removes from the work

`candle-transformers` 0.11.0 ships, among others, `quantized_qwen3`, `quantized_qwen2`,
`quantized_qwen3_moe`, `quantized_lfm2`, `quantized_gemma3`, `quantized_phi3`,
`quantized_llama`, `quantized_mistral` and `quantized_glm4` — which is to say, the
architectures of the `/models` catalogue. `quantized_qwen3::ModelWeights` offers:

```rust
pub fn from_gguf<R: Seek + Read>(ct: Content, reader: &mut R, device: &Device) -> Result<Self>
pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>
```

`from_gguf` loads straight off the file. `forward` takes a position index and keeps the KV
cache in the struct, so incremental decoding is a loop, not a design problem.

**The forward pass is therefore not ours to write.** Neither are the k-quant dequantization
kernels: `candle_core::quantized` carries both a `gguf_file` module and a `k_quants` module,
in pure Rust.

This was the decisive finding. It moves the milestone from "implement a transformer" to
"wire one up, and solve the two things candle does not do for us".

The build cost is the reason candle was chosen over `llama-cpp-2`, which is more capable in
every respect (1.46M downloads against candle's smaller share, monthly releases tracking
llama.cpp, every quantization, every architecture, CUDA/Metal/Vulkan, a GGUF tokenizer and
chat templates already built). `llama-cpp-2` depends on `llama-cpp-sys-2`: a C/C++ build.

What that costs here is specific, not abstract. `.github/workflows/release.yml` builds four
targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`
and `x86_64-pc-windows-msvc`. The second of those is **cross-compiled**, so a `-sys`
dependency would require a cross C toolchain for aarch64 Linux, not merely cmake on the
build host. Keeping `cargo build --release --locked --target <t>` working unchanged on all
four was judged worth more than the capability, given that the scope is one small model
on CPU.

### The two things candle does not do

**The tokenizer.** candle's own quantized examples use the HuggingFace `tokenizers` crate
with a separate `tokenizer.json`, and `tokenizers` is **not** a dependency of
`candle-transformers` (its dependencies are byteorder, candle-core, candle-nn, fancy-regex,
num-traits, rand, rayon, serde, serde_json, serde_plain, tracing). Requiring a second file
alongside the GGUF would defeat the milestone's stated intent, and many GGUF repositories
publish no `tokenizer.json` at all.

So: extend our parser to read the vocabulary out of the GGUF, then build a
`tokenizers::Tokenizer` from it **in memory**. `BpeBuilder::vocab_and_merges()` exists for
exactly this — vocab as a `String -> u32` map, merges as the crate's `Merges` type — so no
BPE encoder has to be written. That matters beyond effort: a hand-written BPE that is
slightly wrong does not crash, it silently degrades every answer, and the failure is very
hard to see.

**The template.** Qwen3 uses ChatML. The renderer is a pure function from
`&[ChatMessage]` to `String`:

```
<|im_start|>system\n{content}<|im_end|>\n
<|im_start|>user\n{content}<|im_end|>\n
<|im_start|>assistant\n
```

The GGUF's own `tokenizer.chat_template` is read but **not interpreted**. It is read for one
purpose: so the engine can say "this model's template is not one I render" instead of
producing fluent nonsense. The plan must verify the rendered string against the template
string that ships in the actual Qwen3 GGUF.

## Components

### `src/models/gguf.rs` — untouched

This milestone does not change the existing parser at all, and the first draft of this spec
was wrong to plan an extension of it.

The reason: `candle_core::quantized::gguf_file::Content` — which must be read anyway to load
the weights — exposes `metadata: HashMap<String, Value>` with `Value::Array(Vec<Value>)` and
accessors `to_vec()`, `to_string()`, `to_i32()`, `to_u32()`, `to_bool()`. **candle already
parses the vocabulary arrays our parser steps over.** Writing array readers to recover data
candle hands us would be duplicate code, duplicate bounds checks and duplicate bugs.

`src/models/gguf.rs` therefore keeps its single job, and keeps it well: inspecting a freshly
downloaded file for `/models` without loading a tensor or a 150 000-entry vocabulary.

### `src/models/tokenizer.rs` — new

Two functions, split so that the part worth testing is testable:

```rust
/// The tokenizer data a local engine needs, lifted out of a GGUF's metadata.
pub struct TokenizerData {
    pub model: String,                 // tokenizer.ggml.model: "gpt2" | "llama" | "spm"
    pub tokens: Vec<String>,           // tokenizer.ggml.tokens
    pub merges: Vec<String>,           // tokenizer.ggml.merges, "a b" per entry
    pub token_type: Vec<i32>,          // tokenizer.ggml.token_type
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unknown: Option<u32>,
    pub add_bos: Option<bool>,
    pub chat_template: Option<String>, // read, not interpreted
}

/// Lifts the tokenizer keys out of the metadata candle already parsed.
pub fn from_metadata(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
) -> Result<TokenizerData, ModelError>;

/// Builds the tokenizer. Pure: no file, no candle types, no I/O.
pub fn build(data: &TokenizerData) -> Result<tokenizers::Tokenizer, ModelError>;
```

The split matters for testing: `Value`'s variants are public, so a test builds a metadata map
literally and drives `from_metadata` with no file at all, and `build` never sees a candle type.

`build` handles `model == "gpt2"` (byte-level BPE) and returns
`ModelError::Gguf("tokenizer <name> non pris en charge")` for anything else. Byte-level
pre-tokenization and decoding are configured to match what the GGUF declares.

### `src/models/template.rs` — new

```rust
/// Renders a conversation into the prompt string an architecture expects.
pub fn render(architecture: &str, messages: &[ChatMessage]) -> Result<String, ModelError>;
```

Pure. No I/O, no clock. Qwen3 → ChatML; every other architecture → an error that names it.

### `src/llm/local.rs` — new

`LocalClient` implements `LlmClient`.

- `list_models()` reads the store and returns one `ModelInfo` per downloaded GGUF, with
  `context_window` from `LocalModel.context_length`. It never fails because a model is
  missing; an empty store is `Ok(vec![])`.
- `context_window(model)` answers from the store rather than probing anything.
- `chat_stream(request)`:
  1. Resolve `request.model` to a `LocalModel` and a path, through the store.
  2. Render the prompt with `template::render`.
  3. Load or reuse the model (see below), encode the prompt, and hand both to the engine
     thread.
  4. Return a `TokenStream` over the receiving end of an `mpsc`.

**The model stays loaded between messages**, behind a `Mutex<Option<Loaded>>` keyed by path.
Reloading several hundred megabytes per message would make the feature unusable. One model
resident at a time; selecting a different one drops the first.

### The thread boundary

Inference is synchronous and CPU-bound. It runs on a **dedicated thread**, never on the
tokio runtime: a thread that would otherwise block the executor for the whole generation.
Tokens cross back through a `tokio::sync::mpsc`, whose receiver is adapted into the
`BoxStream` the trait requires.

Cancellation falls out of this for free and matches what the application already does:
`Échap` drops the stream, the receiver drops, the thread's next `send` fails, and the decode
loop stops. No cancellation token, no new mechanism.

### Configuration

- `ProviderKind::Local` added to `src/config.rs:117-125` (`kind = "local"` in TOML, given
  the enum's `rename_all = "lowercase"`).
- A `local` preset in `presets()` (`src/config.rs:277-308`) with no `base_url` and no
  `api_key_env`, so `Provider::missing_key()` never gates it and it is always constructed.
- One new arm in `build_clients` (`src/llm/mod.rs:240-265`).

**No schema migration.** The GGUF is re-read when a model is loaded — which must happen
anyway for the weights — so there is nothing to persist. The six metadata fields
`download.rs` currently discards stay discarded; this milestone does not need them, and
widening `local_models` for data we read from the file regardless would be storage for its
own sake.

### Two GGUF readers, kept on purpose

After this milestone the crate contains two: ours (`src/models/gguf.rs`) and candle's
(`candle_core::quantized::gguf_file`). This is deliberate, and the split is clean because
each is used for exactly one thing.

Ours is bounds-checked against hostile input and reads a freshly downloaded file without
loading a single tensor or a single vocabulary entry — which is what `/pull` and `/models`
need, on every model in the store, every time the popup opens.

Candle's reads everything into memory, which is correct when a model is actually being
loaded, and it is then the single source for both the weights **and** the tokenizer data.
Nothing is parsed twice on the loading path.

Replacing ours with candle's would mean reading a whole vocabulary to render a list row.
Replacing candle's with ours would mean reimplementing its tensor loader. Both stay.

## Error handling

`LlmError` already exists and its `Display` text is shown verbatim in the UI, in French
(`src/llm/mod.rs:180-206`). Five new conditions, all mapped onto `LlmError::Server`, since
there is no network and therefore no status code:

| Condition | Message |
|---|---|
| File missing or unreadable | `modèle introuvable sur le disque : <chemin>` |
| GGUF corrupt or truncated | `fichier GGUF illisible : <raison>` |
| Architecture not supported | `architecture « <nom> » non prise en charge par le moteur local` |
| Tokenizer family not supported | `tokenizer « <nom> » non pris en charge` |
| Allocation failure while loading | `mémoire insuffisante pour charger <nom>` |

`Unreachable`, `Timeout`, `Auth`, `RateLimited`, `Http` and `Protocol` are unreachable for
this backend and must not appear in it.

## Testing

Everything is unit-testable **except the forward pass**, and the split is the point.

- **`tokenizer::from_metadata`** — a metadata map built literally in the test, since `Value`'s
  variants are public. Cases: the vocabulary and merges lifted in order; special-token ids
  recovered; `chat_template` captured; a missing `tokenizer.ggml.tokens` rejected with a
  message naming the key rather than panicking; a `tokens` entry of the wrong `Value` type
  rejected rather than silently skipped.
- **`tokenizer::build`** — a tiny hand-written vocabulary with known encode/decode pairs,
  including a round trip, a byte-fallback case, and an unsupported family rejected. Takes no
  candle type, so it needs no fixture at all.
- **`template::render`** — exact string assertions, character for character, including the
  trailing `<|im_start|>assistant\n` that must be present for generation to start, a
  system-prompt-only conversation, and an unsupported architecture rejected by name.
- **`LocalClient`** — the engine sits behind a trait so the streaming plumbing is tested
  with a fake that yields scripted tokens. This covers: tokens arriving in order, an error
  mid-stream surfacing as `LlmError`, cancellation stopping the thread when the receiver
  drops, and `list_models` on an empty store returning `Ok(vec![])`. **No test loads a real
  model**, so the suite stays fast and needs no fixture of several hundred megabytes.

**What is not tested, stated plainly:** the forward pass is candle's, and the end-to-end
check — a real Qwen3 0.6B GGUF producing coherent French — is manual, once, like the
HuggingFace check J32 left open. We do not test other people's matrix multiplication.

## Dependencies

Three new direct dependencies, or four if `tokenizers` survives the feature audit below.
Either way this is the largest tree the project has taken:

| Crate | Version | Why |
|---|---|---|
| `candle-core` | 0.11 | Tensors, `quantized::gguf_file`, `k_quants` |
| `candle-nn` | 0.11 | Required by candle-transformers |
| `candle-transformers` | 0.11 | `quantized_qwen3::ModelWeights`, `generation::LogitsProcessor` |
| `tokenizers` | see the risk below | `BpeBuilder::vocab_and_merges` |

### The `tokenizers` feature risk, and the escape hatch

This is the one open risk in the design, and it can send the tokenizer back to being
hand-written. It is recorded here rather than discovered during implementation.

`tokenizers` 0.23.2 — the stable line — has default features `["progressbar", "onig",
"esaxx_fast"]`. **`onig` is a C library**, and pulling it would defeat the entire reason
candle was chosen over `llama-cpp-2`, cross-compilation included. Worse, 0.23.2's pure-Rust
regex alternative (`fancy-regex`) is reachable only through a feature named
`unstable_wasm` — depending on an explicitly unstable, wasm-oriented feature in a native
build, to dodge a C library, is a bodge, not a design.

`tokenizers` 1.0.0-rc.2 (21 September 2026) has default features `["progressbar"]` only: no
`onig`, no `esaxx_fast`. It is clean — and it is a release candidate, in a project that
otherwise pins stable versions.

The plan resolves this in order, and stops at the first that works:

1. **`tokenizers` 1.0.0-rc.2**, `default-features = false`. Preferred. The pre-release status
   is an accepted, recorded risk: the crate is used here for one narrow purpose (a byte-level
   BPE built from an in-memory vocabulary), not for its breadth.
2. **`tokenizers` 0.23.2 with `default-features = false`**, *if* the byte-level BPE path
   compiles and passes the round-trip tests without any regex engine — plausible, because
   `onig` serves pretokenizers and normalizers we do not use, but it must be proven, not
   assumed.
3. **Write the byte-level BPE by hand**, which was the second-ranked option when this was
   decided and remains a sound fallback. The GPT-2 pretokenizer pattern needs a negative
   lookahead that the `regex` crate cannot express, so this means a small hand-rolled
   scanner rather than a regex — about two hundred lines, tested against known token
   sequences for real Qwen3 text.

**`cargo tree` must show no `*-sys` crate anywhere in the build, and this check is a gate,
not a nicety.** It is the one verification that proves the milestone's central trade-off was
actually honoured. If options 1 and 2 both fail it, the plan takes option 3 — the milestone's
scope does not change, only the amount of code in `src/models/tokenizer.rs`.

## Accepted limitations

- One architecture. A Gemma or Llama GGUF in the store is listed by `/models` and refused by
  name when selected. That is better than the alternative — being silently wrong — but it is
  a limitation, and `/models` should not imply otherwise.
- CPU only, so a 7B model is technically loadable and practically unusable. The milestone
  does not stop you; it just does not pretend.
- No tool calling, no RAG-shaped structured output from a local model.
- Token counts for a local model stay estimates, even though an exact tokenizer is now
  present. Wiring it into `src/tokens.rs` is a later change.
- A second local model selected mid-session unloads the first, with a visible pause.

## Suggested task breakdown

Four tasks, one per layer, in dependency order:

1. **`src/models/tokenizer.rs`** — `TokenizerData`, `from_metadata`, `build`. Owns the
   `candle-core` and `tokenizers` dependencies. **This task opens with the feature audit**,
   because its outcome decides how much code the task contains: a thin adapter over
   `tokenizers` if resolution option 1 or 2 holds, a hand-rolled scanner if option 3 is
   forced. Its tests — encode, decode, round trip, byte fallback, unsupported family — are
   the same either way, which is what makes the fallback cheap to take.
2. **`src/models/template.rs`** — ChatML rendering. Shares no file with task 1 and depends on
   nothing it produces; the two can start together.
3. **`src/llm/local.rs`** — the `Engine` seam, the candle engine, the thread boundary and the
   streaming adapter, plus `candle-nn` and `candle-transformers`. Depends on tasks 1 and 2.
4. **Wiring and documentation** — `ProviderKind::Local`, the preset, the `build_clients` arm
   and its new parameter, `README.md`, `PLAN.md`, `docs/roadmap.md`, and the `cargo tree`
   gate. Depends on task 3.

Tasks 1 and 2 share no file and can start together. In Rust the crate is a single
compilation unit, so parallel tasks must run in **isolated worktrees**: a task's TDD red
phase leaves the lib test binary uncompilable, which would break its neighbour's
`cargo test`. This was learnt the hard way in J32 and confirmed in J33.

### One interface the first draft missed

`store::list` requires a `&rusqlite::Connection`, and `build_clients(providers,
connect_timeout)` receives neither a connection nor the models configuration — so a
`LocalClient` built there cannot read the inventory, as the first draft assumed.

Resolution: `LocalClient` does not read the database at all. It **scans the models
directory**, whose layout `download::paths` fixes as `<dir>/<owner>/<repo>/<file>`, and reads
each file's header with the existing `gguf::read` for the context window. The filesystem is
the right authority for "what can I load" — a file deleted outside the application is
correctly absent — and it avoids a second SQLite connection.

That requires the resolved directory to reach `build_clients`, which is a three-line change:
`ModelsBackend::new(&config, &database)` already resolves it at `src/runtime.rs:195`, just
below the `build_clients` call at `:193`. The two are reordered, `build_clients` gains a
`models_dir: &Path` parameter, and its one test call site (`src/llm/mod.rs:368`) is updated.
Task 4 owns this.
