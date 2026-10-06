//! The local provider: a GGUF from the store, decoded in this process.
//!
//! Inference is synchronous and CPU-bound, so it runs on a dedicated OS thread and never on
//! the tokio runtime. Tokens cross back through an mpsc channel presented as a [`TokenStream`].

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

/// The three states the one model slot can be in.
///
/// [`Busy`](Slot::Busy) is the state the first version lacked, and the whole point of the
/// enum: it is not the same thing as [`Empty`](Slot::Empty), and answering a second request
/// for the same model as though the slot were empty means telling it to load its own copy.
enum Slot<T> {
    /// Nothing is loaded and nobody is loading.
    Empty,
    /// `id`'s value is loaded and free to take.
    Free(String, T),
    /// A generation holds `id`'s value, or is loading it right now.
    Busy(String),
}

/// Holds at most one value, labelled with the model id it was loaded for.
///
/// [`acquire`](Self::acquire) *moves* the value out instead of lending it: a generation owns
/// its weights outright for its whole run, so two overlapping generations can never interleave
/// `forward` calls on one KV cache. The engine puts the value back when it is dropped.
///
/// A request for another id evicts whatever is resident, because the caller is about to load
/// its replacement and the spec keeps one model in memory at a time. A request for the id
/// already in use waits for it instead, because loading it twice is what would run the
/// machine out of memory.
struct Resident<T> {
    slot: std::sync::Mutex<Slot<T>>,
    /// Signalled by [`release`](Self::release) and [`abandon`](Self::abandon), so a waiting
    /// request wakes as soon as the model comes back rather than on a timer.
    returned: std::sync::Condvar,
}

impl<T> Default for Resident<T> {
    fn default() -> Self {
        Self {
            slot: std::sync::Mutex::new(Slot::Empty),
            returned: std::sync::Condvar::new(),
        }
    }
}

impl<T> Resident<T> {
    /// The slot, recovered even if a previous holder panicked while holding the lock.
    ///
    /// Poisoning carries no information here — the slot is a value or a label, and both are
    /// self-consistent whatever a panic interrupted — while refusing to look would turn one
    /// panic into a provider that can never load again.
    fn locked(&self) -> std::sync::MutexGuard<'_, Slot<T>> {
        self.slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether `id` is loaded and free to take right now.
    ///
    /// A seam for the tests, not a production query: the only production question is "give me
    /// this model if you have it", which [`acquire`](Self::acquire) answers in one step.
    #[cfg(test)]
    fn holds(&self, id: &str) -> bool {
        matches!(&*self.locked(), Slot::Free(held, _) if held == id)
    }

    /// Claims the slot for `id`, handing back the resident value when it is the one wanted.
    ///
    /// `Ok(None)` means "the slot is yours, load it"; `Err` names the model that holds it and
    /// did not hand it back in time. Either way the caller owes the slot a
    /// [`release`](Self::release) or an [`abandon`](Self::abandon).
    fn acquire(&self, id: &str) -> Result<Option<T>, String> {
        self.acquire_within(id, HAND_BACK_TIMEOUT)
    }

    fn acquire_within(&self, id: &str, wait: std::time::Duration) -> Result<Option<T>, String> {
        let deadline = std::time::Instant::now() + wait;
        let mut slot = self.locked();
        loop {
            match std::mem::replace(&mut *slot, Slot::Empty) {
                Slot::Free(held, value) if held == id => {
                    *slot = Slot::Busy(held);
                    return Ok(Some(value));
                }
                // Another model's weights, dropped here: that is the eviction, and it is why
                // only one model is ever resident.
                Slot::Free(..) | Slot::Empty => {
                    *slot = Slot::Busy(id.to_owned());
                    return Ok(None);
                }
                Slot::Busy(held) => {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        let message = format!(
                            "{held} est déjà en cours d'utilisation : attendez la fin de la \
                             réponse en cours"
                        );
                        *slot = Slot::Busy(held);
                        return Err(message);
                    }
                    *slot = Slot::Busy(held);
                    slot = match self.returned.wait_timeout(slot, remaining) {
                        Ok((guard, _)) => guard,
                        Err(poisoned) => poisoned.into_inner().0,
                    };
                }
            }
        }
    }

    /// Hands `value` back, free for the next request.
    fn release(&self, id: String, value: T) {
        *self.locked() = Slot::Free(id, value);
        self.returned.notify_all();
    }

    /// Gives the slot up without a value: a load that failed holds nothing to hand back.
    fn abandon(&self) {
        *self.locked() = Slot::Empty;
        self.returned.notify_all();
    }
}

/// How long a request waits for the generation holding the model to hand it back.
///
/// Cancellation is asynchronous — `Échap` drops the stream and the engine thread only notices
/// on its next send — and a load in `spawn_blocking` cannot be cancelled at all, so a
/// re-send arriving straight after a cancellation finds the model still in use. Waiting is
/// right and loading a second copy is not: a 4 B model is some 2.5 GB of weights plus
/// candle's pre-allocated KV cache, and two of those sink an 8 GB machine.
const HAND_BACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// A model loaded in memory, kept between messages.
///
/// Several hundred megabytes: reloading per message would make the feature unusable, which is
/// why it is cached at all. The KV cache lives inside `weights`, so it is cleared on the way
/// back into [`Resident`] — see [`CandleEngine::drop`].
struct Loaded {
    weights: candle_transformers::models::quantized_qwen3::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    /// Token ids that end the reply: the file's end-of-text id when it declares one, plus
    /// the ChatML terminators our own template taught the model to answer with.
    stops: Vec<u32>,
}

/// The local provider: GGUF files from the model store, decoded in this process.
pub struct LocalClient {
    dir: PathBuf,
    /// The one resident model, shared with the engine thread that borrowed it.
    cache: std::sync::Arc<Resident<Loaded>>,
}

impl LocalClient {
    /// `models_dir` is the resolved store directory, laid out as `<dir>/<owner>/<repo>/<file>`
    /// by `models::download::paths`.
    pub fn new(models_dir: &Path) -> Self {
        Self {
            dir: models_dir.to_path_buf(),
            cache: std::sync::Arc::default(),
        }
    }

    /// Whether `id` is loaded and ready to answer without a reload.
    ///
    /// The seam the cache test drives. It answers `bool` rather than handing the model out:
    /// a `Loaded` cannot outlive the lock that guards it, and taking it would be a side
    /// effect a predicate has no business having.
    #[cfg(test)]
    fn cached(&self, id: &str) -> bool {
        self.cache.holds(id)
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
                    if path.extension().is_some_and(|e| e == "gguf")
                        && let (Some(owner), Some(repo), Some(name)) = (
                            owner.file_name().to_str(),
                            repo.file_name().to_str(),
                            path.file_name().and_then(|n| n.to_str()),
                        )
                    {
                        found.push(format!("{owner}/{repo}/{name}"));
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
            return Err(LlmError::Local(format!(
                "identifiant de modèle invalide : {id}"
            )));
        };
        for part in [owner, repo, file] {
            if part.is_empty() || part.contains("..") || part.contains('\\') {
                return Err(LlmError::Local(format!(
                    "identifiant de modèle invalide : {id}"
                )));
            }
        }
        let path = self.dir.join(owner).join(repo).join(file);
        if !path.is_file() {
            return Err(LlmError::Local(format!(
                "modèle introuvable sur le disque : {}",
                path.display()
            )));
        }
        Ok(path)
    }
}

/// Whether candle refuses this quantization outright.
///
/// candle 0.11's `GgmlDType` holds `F32`, `F16`, `BF16`, the legacy `Q4_0`…`Q8_1` and the
/// K-quants `Q2_K`…`Q8_K` — and no i-quant whatsoever. Every i-quant name begins with `IQ` and
/// no supported one does, so the prefix is the whole test. Asking the header costs nothing and
/// happens before the file is opened, which is what lets the refusal name the quantization
/// instead of surfacing candle's `unknown dtype for tensor 16`.
fn is_i_quant(quantization: &str) -> bool {
    quantization.starts_with("IQ")
}

#[async_trait::async_trait]
impl super::LlmClient for LocalClient {
    fn preparing(&self) -> super::Phase {
        super::Phase::Loading
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<TokenStream, LlmError> {
        let path = self.path_of(&request.model)?;
        let metadata =
            crate::models::gguf::read(&path).map_err(|e| LlmError::Local(e.to_string()))?;
        let architecture = metadata
            .architecture
            .ok_or_else(|| LlmError::Local("le fichier ne nomme pas son architecture".into()))?;
        if let Some(quantization) = metadata.quantization.as_deref()
            && is_i_quant(quantization)
        {
            return Err(LlmError::Local(format!(
                "quantization « {quantization} » non prise en charge par le moteur local : \
                 il faut un K-quant, par exemple Q4_K_M"
            )));
        }
        let prompt = crate::models::template::render(&architecture, &request.messages)
            .map_err(|e| LlmError::Local(e.to_string()))?;

        // Loading several hundred megabytes is blocking CPU work: it belongs on the blocking
        // pool, not on a runtime worker that also drives the terminal event loop.
        let cache = std::sync::Arc::clone(&self.cache);
        let id = request.model.clone();
        let engine = tokio::task::spawn_blocking(move || {
            CandleEngine::load(cache, id, &path, &architecture, &prompt)
        })
        .await
        .map_err(|e| LlmError::Local(format!("le chargement du modèle a échoué : {e}")))?
        .map_err(|e| LlmError::Local(e.to_string()))?;
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

/// Decodes with candle, one token per call.
struct CandleEngine {
    /// The borrowed model. `Some` for the engine's whole life; it is an `Option` only so
    /// [`Drop`] can move it back into the cache.
    model: Option<Loaded>,
    /// Where the model goes when this engine ends, however it ends.
    cache: std::sync::Arc<Resident<Loaded>>,
    /// The id the model is filed under.
    id: String,
    logits: candle_transformers::generation::LogitsProcessor,
    device: candle_core::Device,
    /// The next token to feed: the prompt on the first call, then the sampled token.
    pending: Vec<u32>,
    position: usize,
    /// Turns the sampled ids into the text the reader sees.
    stream: Incremental,
    limit: usize,
}

/// Turns sampled token ids into text, one token at a time.
///
/// Separate from [`CandleEngine`] because this is the half that can be tested: it needs a
/// tokenizer and nothing else, where the rest needs several hundred megabytes of weights.
#[derive(Default)]
struct Incremental {
    produced: Vec<u32>,
    /// Byte length of the prefix of the current decode already handed to the reader.
    emitted: usize,
}

impl Incremental {
    /// How many tokens have been accepted so far.
    fn len(&self) -> usize {
        self.produced.len()
    }

    /// Accepts `token` and returns the text it completed, or `None` when it completed none.
    ///
    /// The whole reply is decoded each time and only the new tail is returned, because a
    /// token decoded alone can be a fraction of a character. The subtlety is that
    /// `ByteLevel`'s decoder ends in `String::from_utf8_lossy`: an incomplete trailing UTF-8
    /// sequence does not come back as nothing, it comes back as U+FFFD — three bytes that the
    /// next token then *replaces* with the real character, shrinking the decoded text. So a
    /// decode whose tail is incomplete is never committed: neither emitted nor counted, so
    /// that `emitted` stays a prefix of the current decode and the character is handed over
    /// whole once its last byte arrives.
    fn push(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        token: u32,
    ) -> Result<Option<String>, LlmError> {
        self.produced.push(token);
        let text = tokenizer
            .decode(&self.produced, false)
            .map_err(|e| LlmError::Local(format!("décodage impossible : {e}")))?;
        // A trailing replacement glyph is the only signal available that the tail is half a
        // character, the lossy conversion having already happened inside the decoder. A model
        // that genuinely ends its reply on U+FFFD loses that one character, which is a far
        // better trade than every accented reply losing two.
        if text.ends_with(char::REPLACEMENT_CHARACTER) {
            return Ok(None);
        }
        let Some(fragment) = text.get(self.emitted..) else {
            // Unreachable while the invariant above holds, and cheap insurance if it ever
            // does not: resynchronize on the current decode rather than repeat text the
            // reader has already seen.
            self.emitted = text.len();
            return Ok(None);
        };
        if fragment.is_empty() {
            return Ok(None);
        }
        let fragment = fragment.to_owned();
        self.emitted = text.len();
        Ok(Some(fragment))
    }
}

/// Most tokens a single reply may contain.
///
/// A reply cannot run forever: a GGUF that declares no end-of-text id would otherwise decode
/// until the user quits, since nothing else would ever stop the loop.
const MAX_REPLY_TOKENS: usize = 2048;

/// Turn terminators of the ChatML prompt `template::render` writes.
///
/// The tokenizer is built from the raw GGUF vocabulary, with no added-token registry, so
/// these are ordinary vocabulary entries rather than tokenizer-level special tokens: nothing
/// below `Tokenizer::decode` would hide them, and a reply ending on one would otherwise show
/// `<|im_end|>` to the user as text.
const CHATML_STOPS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

/// The control tokens `template::render` writes into the prompt.
///
/// A GGUF has no added-token registry: these are plain vocabulary entries, and a byte-level
/// BPE given `<|im_start|>` as text splits it into twelve one-character tokens. The model
/// would then never see ChatML at all — no error anywhere, just a prompt of plausible bytes
/// and a fluent, wrong reply. Marking them is what makes the rendered prompt the prompt the
/// model was trained on.
const CONTROL_TOKENS: [&str; 5] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
];

/// Sampling temperature and nucleus cutoff, Qwen's own recommended pair for chat.
const TEMPERATURE: f64 = 0.7;
const TOP_P: f64 = 0.9;

/// The seed one generation samples with.
///
/// A fresh one each time, because `LogitsProcessor::from_sampling` seeds
/// `StdRng::seed_from_u64`: with the same prompt, the same weights and a fixed seed, the reply
/// is the same tokens byte for byte — and `/retry` and `/compare` against the same local model
/// rebuild exactly that prompt. Reading a clock here is allowed: the purity rule binds
/// `App::update`, not an `LlmClient`.
///
/// The counter alone would restart with the process and the clock alone can read the same
/// value twice on a coarse timer, so the two are mixed. The odd multiplier spreads successive
/// counter values across the whole word instead of changing one bit at a time.
fn generation_seed() -> u64 {
    static STEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            since
                .as_secs()
                .wrapping_mul(1_000_000_000)
                .wrapping_add(u64::from(since.subsec_nanos()))
        });
    let step = STEP
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    elapsed ^ step
}

/// The two control tokens the ChatML prompt cannot be built without.
///
/// `template::render` opens and closes every turn with them, and `<|im_end|>` is also what
/// ends the reply. A vocabulary that lacks either — a fine-tune, a re-quantized derivative —
/// cannot be driven in ChatML at all: the model would never see a turn boundary, nothing would
/// fail, and a missing `<|im_end|>` would additionally leave `stops` empty so every reply ran
/// to the token cap. The rest of [`CONTROL_TOKENS`] is genuinely optional: `<|endoftext|>` is
/// a second terminator and `<think>`/`</think>` only matter to models that reason, so those
/// stay best-effort.
const REQUIRED_CONTROL_TOKENS: [&str; 2] = ["<|im_start|>", "<|im_end|>"];

/// Tells `tokenizer` that the prompt's control tokens are atomic.
///
/// Only entries the vocabulary already defines are marked: `add_special_tokens` would
/// otherwise mint a fresh id past the model's output dimension, which the model cannot read.
/// A file missing one of [`REQUIRED_CONTROL_TOKENS`] is refused by name instead, because
/// skipping it is the silent-degradation outcome this module refuses everywhere else.
fn mark_control_tokens(
    tokenizer: &mut tokenizers::Tokenizer,
) -> Result<(), crate::models::ModelError> {
    for token in REQUIRED_CONTROL_TOKENS {
        if tokenizer.token_to_id(token).is_none() {
            return Err(crate::models::ModelError::Unsupported(format!(
                "le vocabulaire du fichier ne contient pas {token} : le moteur local ne peut pas \
                 lui parler en ChatML"
            )));
        }
    }
    let known: Vec<tokenizers::AddedToken> = CONTROL_TOKENS
        .iter()
        .filter(|token| tokenizer.token_to_id(token).is_some())
        .map(|token| tokenizers::AddedToken::from(*token, true))
        .collect();
    tokenizer.add_special_tokens(&known);
    Ok(())
}

impl CandleEngine {
    /// Reuses the resident model when it is the one asked for, and loads it otherwise.
    ///
    /// Blocking: reads the whole file and dequantizes the embedding table. Call it off the
    /// tokio runtime's worker threads.
    fn load(
        cache: std::sync::Arc<Resident<Loaded>>,
        id: String,
        path: &Path,
        architecture: &str,
        prompt: &str,
    ) -> Result<Self, crate::models::ModelError> {
        use crate::models::ModelError;

        if architecture != "qwen3" {
            return Err(ModelError::Unsupported(format!(
                "architecture « {architecture} » non prise en charge par le moteur local"
            )));
        }

        // The slot is claimed from here on: every path out of this function either moves the
        // model into the engine, whose `Drop` hands it back, or gives the slot up explicitly.
        let model = match cache.acquire(&id).map_err(ModelError::Busy)? {
            Some(model) => model,
            None => match Self::read(path) {
                Ok(model) => model,
                Err(error) => {
                    cache.abandon();
                    return Err(error);
                }
            },
        };

        let encoded = match model.tokenizer.encode(prompt, true) {
            Ok(encoded) => encoded,
            Err(error) => {
                cache.release(id, model);
                return Err(ModelError::Gguf(format!("prompt non encodable : {error}")));
            }
        };

        Ok(Self {
            model: Some(model),
            cache,
            id,
            logits: candle_transformers::generation::LogitsProcessor::new(
                generation_seed(),
                Some(TEMPERATURE),
                Some(TOP_P),
            ),
            device: candle_core::Device::Cpu,
            pending: encoded.get_ids().to_vec(),
            position: 0,
            stream: Incremental::default(),
            limit: MAX_REPLY_TOKENS,
        })
    }

    /// Reads the weights and the tokenizer out of one GGUF, candle's `Content` serving both.
    fn read(path: &Path) -> Result<Loaded, crate::models::ModelError> {
        use crate::models::ModelError;

        let mut file = std::fs::File::open(path).map_err(|e| ModelError::Io(e.to_string()))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| ModelError::Gguf(e.to_string()))?;

        let data = crate::models::tokenizer::from_metadata(&content.metadata)?;
        let mut tokenizer = crate::models::tokenizer::build(&data)?;
        mark_control_tokens(&mut tokenizer)?;
        let mut stops: Vec<u32> = data.eos.into_iter().collect();
        for token in CHATML_STOPS {
            if let Some(id) = tokenizer.token_to_id(token)
                && !stops.contains(&id)
            {
                stops.push(id);
            }
        }

        let weights = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
            content,
            &mut file,
            &candle_core::Device::Cpu,
        )
        .map_err(|e| ModelError::Gguf(e.to_string()))?;

        Ok(Loaded {
            weights,
            tokenizer,
            stops,
        })
    }
}

impl Drop for CandleEngine {
    /// Returns the model to the cache, whatever ended the generation — end-of-text, the token
    /// cap, an error, or the reader dropping the stream.
    ///
    /// The KV cache is cleared here, which is the single place it is cleared: what sits in
    /// [`Resident`] is therefore always clean, and the next prompt — which starts again at
    /// position 0 — cannot attend to this conversation's keys. Skipping this would not fail
    /// loudly; it would quietly contaminate the next reply.
    fn drop(&mut self) {
        if let Some(mut model) = self.model.take() {
            model.weights.clear_kv_cache();
            self.cache.release(std::mem::take(&mut self.id), model);
        }
    }
}

impl Engine for CandleEngine {
    fn next_token(&mut self) -> Result<Option<String>, LlmError> {
        let Some(model) = self.model.as_mut() else {
            return Err(LlmError::Local(
                "le moteur local a déjà rendu son modèle".to_owned(),
            ));
        };
        // A loop rather than recursion: a token can complete no character on its own, and
        // recursing would hold one set of tensors per attempt.
        loop {
            if self.stream.len() >= self.limit {
                return Ok(None);
            }
            let input = candle_core::Tensor::new(self.pending.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| LlmError::Local(e.to_string()))?;
            // `forward` narrows to the last position itself, so the result is
            // `(batch, vocabulary)` whatever the prompt's length: one `squeeze` is enough.
            let logits = model
                .weights
                .forward(&input, self.position)
                .and_then(|l| l.squeeze(0))
                .map_err(|e| LlmError::Local(e.to_string()))?;

            self.position += self.pending.len();
            let next = self
                .logits
                .sample(&logits)
                .map_err(|e| LlmError::Local(e.to_string()))?;
            self.pending = vec![next];

            if model.stops.contains(&next) {
                return Ok(None);
            }
            if let Some(fragment) = self.stream.push(&model.tokenizer, next)? {
                return Ok(Some(fragment));
            }
            // This token completed no character: ask for the next one rather than emitting
            // half of one.
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::llm::LlmClient;

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
                None if self.error_at_end => Err(LlmError::Local("le moteur a échoué".to_owned())),
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

    /// The waiting line shows whatever this returns while the weights are read. For an engine
    /// with no server, the inherited `Connecting` was simply false.
    #[test]
    fn the_local_engine_prepares_by_loading_not_connecting() {
        let dir = tempfile::tempdir().expect("a temporary directory");

        let phase = LocalClient::new(dir.path()).preparing();

        assert_eq!(phase, crate::llm::Phase::Loading);
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
            matches!(items.last(), Some(Err(LlmError::Local(_)))),
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

        // `TokenStream` is a boxed stream and not `Debug`, so `expect_err` cannot be used.
        let outcome = client
            .chat_stream(ChatRequest {
                model: "unsloth/Qwen3-0.6B-GGUF/gone.gguf".to_owned(),
                messages: Vec::new(),
                tools: Vec::new(),
            })
            .await;
        let Err(error) = outcome else {
            panic!("the file is not there");
        };

        let text = error.to_string();
        assert!(text.contains("gone.gguf"), "{text}");
    }

    #[test]
    fn the_resident_slot_hands_back_only_what_it_was_asked_for() {
        let slot: Resident<String> = Resident::default();
        assert!(!slot.holds("a"), "nothing is resident before a load");
        assert!(slot.acquire("a").expect("free").is_none());
        slot.abandon();

        slot.release("a".to_owned(), "les poids de a".to_owned());
        assert!(slot.holds("a"));
        assert_eq!(
            slot.acquire("a").expect("free").as_deref(),
            Some("les poids de a")
        );
        assert!(!slot.holds("a"), "taking it leaves nothing resident");
    }

    /// One model resident at a time: asking for another frees the first rather than keeping
    /// both, which is what makes a 4 B model and a 0.6 B model safe to alternate between.
    #[test]
    fn asking_for_another_model_evicts_the_resident_one() {
        let slot: Resident<String> = Resident::default();
        slot.release("a".to_owned(), "les poids de a".to_owned());

        assert!(
            slot.acquire("b").expect("free").is_none(),
            "b was never loaded"
        );
        assert!(!slot.holds("a"), "a was evicted to make room for b");
    }

    /// The finding: `Resident` could not tell "nobody has it" from "a generation is using it
    /// right now", and answered both with "load your own". `Échap` during a load then a
    /// re-send does exactly that — cancellation is asynchronous and `spawn_blocking` is not
    /// cancellable at all — so the same file was read twice into memory.
    #[test]
    fn a_model_already_in_use_is_not_offered_for_a_second_load() {
        let slot: Resident<String> = Resident::default();
        slot.release("a".to_owned(), "les poids de a".to_owned());
        let _first = slot.acquire("a").expect("free").expect("resident");

        let outcome = slot.acquire_within("a", std::time::Duration::from_millis(20));

        let error = outcome.expect_err("a second copy must not be loaded");
        assert!(error.contains('a'), "the model is named: {error}");
    }

    /// `/retry` and `/compare` against the same local model rebuild the identical prompt, and
    /// `LogitsProcessor::from_sampling` seeds `StdRng::seed_from_u64`: one fixed seed means
    /// the same tokens byte for byte. The user asks for another answer and is handed the same
    /// one, with nothing on screen to explain why.
    #[test]
    fn each_generation_samples_with_its_own_seed() {
        let seeds: std::collections::HashSet<u64> = (0..64).map(|_| generation_seed()).collect();

        assert_eq!(
            seeds.len(),
            64,
            "generations share a seed, so /retry cannot change the answer"
        );
    }

    /// Refusing is the fallback, not the behaviour: the generation holding the model hands it
    /// back when it ends, and the waiting request then reuses it rather than loading.
    #[test]
    fn a_second_request_waits_for_the_model_to_come_back() {
        let slot = std::sync::Arc::new(Resident::<String>::default());
        slot.release("a".to_owned(), "les poids de a".to_owned());
        let first = slot.acquire("a").expect("free").expect("resident");

        let waiter = {
            let slot = std::sync::Arc::clone(&slot);
            std::thread::spawn(move || slot.acquire("a"))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        slot.release("a".to_owned(), first);

        let acquired = waiter
            .join()
            .expect("the waiting thread finished")
            .expect("it was not refused");
        assert_eq!(
            acquired.as_deref(),
            Some("les poids de a"),
            "it waited for the resident copy instead of loading another"
        );
    }

    /// The cache seam as `LocalClient` exposes it, asserted on the slot rather than on
    /// elapsed time: a timing assertion would be flaky on a loaded machine and would be
    /// measuring candle rather than the cache. Reuse itself needs a real GGUF and is checked
    /// by hand, since no test loads a model.
    #[tokio::test]
    async fn a_failed_load_leaves_nothing_resident() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let client = LocalClient::new(dir.path());
        let id = "unsloth/Qwen3-0.6B-GGUF/a.gguf";

        assert!(
            !client.cached(id),
            "nothing is cached before the first load"
        );

        let outcome = client
            .chat_stream(ChatRequest {
                model: id.to_owned(),
                messages: Vec::new(),
                tools: Vec::new(),
            })
            .await;
        assert!(outcome.is_err(), "the file is not there");
        assert!(!client.cached(id), "a failed load caches nothing");
    }

    /// A GGUF carries no added-token registry, so the ChatML control tokens the prompt is
    /// built from are ordinary vocabulary entries. Left at that, `tokenizers` shreds
    /// `<|im_start|>` into twelve one-character tokens: the model never sees ChatML, there is
    /// no error anywhere, and the reply is fluent nonsense.
    #[test]
    fn the_prompt_s_control_tokens_encode_as_single_tokens() {
        let mut tokenizer = chatml_vocabulary();
        let id = tokenizer
            .token_to_id("<|im_start|>")
            .expect("the vocabulary holds it");

        let shredded = tokenizer
            .encode("<|im_start|>user", true)
            .expect("encodes")
            .get_ids()
            .to_vec();
        assert!(
            shredded.len() > 2,
            "before marking, the control token is split: {shredded:?}"
        );

        mark_control_tokens(&mut tokenizer).expect("the vocabulary holds the ChatML pair");

        let ids = tokenizer
            .encode("<|im_start|>user", true)
            .expect("encodes")
            .get_ids()
            .to_vec();
        assert_eq!(ids.first(), Some(&id), "{ids:?}");
    }

    /// A control token the file does not define must not be invented: `add_special_tokens`
    /// would mint a fresh id past the model's output dimension, and the model cannot read it.
    #[test]
    fn a_control_token_absent_from_the_vocabulary_is_not_invented() {
        let mut tokenizer = chatml_vocabulary();
        assert!(
            tokenizer.token_to_id("<think>").is_none(),
            "this vocabulary has no thinking tokens"
        );
        let before = tokenizer.get_vocab_size(true);

        mark_control_tokens(&mut tokenizer).expect("the vocabulary holds the ChatML pair");

        assert_eq!(
            tokenizer.get_vocab_size(true),
            before,
            "the vocabulary grew by a token the model never saw"
        );
    }

    /// A fine-tune or a re-quantized derivative can ship a vocabulary missing one half of the
    /// ChatML pair. Skipping it silently means the model never sees ChatML, there is no error
    /// anywhere, and the reply is fluent nonsense — and a missing `<|im_end|>` also leaves
    /// `stops` empty, so every reply then runs to the token cap.
    #[test]
    fn a_vocabulary_without_the_chatml_pair_is_refused_by_name() {
        for missing in ["<|im_start|>", "<|im_end|>"] {
            let mut tokenizer = chatml_vocabulary_without(missing);

            let error = mark_control_tokens(&mut tokenizer)
                .expect_err("the prompt is built out of this token");

            let text = error.to_string();
            assert!(text.contains(missing), "{text}");
        }
    }

    /// A GGUF the store lists but the engine cannot run is refused by name rather than
    /// loaded and misread. `qwen2` is the sharp case: `template::render` accepts it, so the
    /// engine is the only thing standing between the user and a reply decoded by the wrong
    /// architecture.
    ///
    /// The whole string is asserted, not a substring: this message used to arrive wrapped in
    /// `erreur du serveur : fichier GGUF illisible :`, two claims that are both false — there
    /// is no server on this path, and the file reads perfectly well.
    #[tokio::test]
    async fn an_architecture_the_engine_does_not_run_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let folder = dir.path().join("Qwen").join("Qwen2.5-GGUF");
        std::fs::create_dir_all(&folder).expect("creates the folders");
        std::fs::write(folder.join("q.gguf"), gguf_header("qwen2")).expect("writes the file");

        let outcome = LocalClient::new(dir.path())
            .chat_stream(ChatRequest {
                model: "Qwen/Qwen2.5-GGUF/q.gguf".to_owned(),
                messages: Vec::new(),
                tools: Vec::new(),
            })
            .await;
        let Err(error) = outcome else {
            panic!("the engine only runs qwen3");
        };

        assert_eq!(
            error.to_string(),
            "architecture « qwen2 » non prise en charge par le moteur local"
        );
    }

    /// Every message on this path reaches the user verbatim. A prefix naming a server would
    /// send them looking for an Ollama that is not involved — the milestone's whole pitch is
    /// that there is no daemon.
    #[tokio::test]
    async fn no_local_failure_claims_a_server() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let client = LocalClient::new(dir.path());

        let outcome = client
            .chat_stream(ChatRequest {
                model: "unsloth/Qwen3-0.6B-GGUF/gone.gguf".to_owned(),
                messages: Vec::new(),
                tools: Vec::new(),
            })
            .await;
        let Err(error) = outcome else {
            panic!("the file is not there");
        };

        let text = error.to_string();
        assert!(!text.contains("serveur"), "{text}");
        assert!(!text.contains("illisible"), "{text}");
    }

    /// The smallest GGUF header our parser accepts: the magic, version 3, no tensor, and one
    /// string key naming the architecture.
    fn gguf_header(architecture: &str) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        // Value type 8 is a string, as the GGUF specification numbers them.
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        bytes.extend_from_slice(architecture.as_bytes());
        bytes
    }

    /// The same header plus `general.file_type`, the key that names the quantization.
    fn gguf_header_quantized(architecture: &str, file_type: u32) -> Vec<u8> {
        let mut bytes = gguf_header(architecture);
        // Correct the key count the single-pair header declared, then append the second pair.
        bytes[16..24].copy_from_slice(&2u64.to_le_bytes());
        let key = b"general.file_type";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        // Value type 4 is a uint32, as the GGUF specification numbers them.
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&file_type.to_le_bytes());
        bytes
    }

    /// A vocabulary of the 256 byte-level characters and nothing else, so every text encodes
    /// and every multi-byte character necessarily straddles several tokens — which is what a
    /// real vocabulary does too for any character it does not hold merged.
    fn byte_level_tokenizer() -> tokenizers::Tokenizer {
        let tokens: Vec<String> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect();
        let data = crate::models::tokenizer::TokenizerData {
            model: "gpt2".to_owned(),
            pre: Some("qwen2".to_owned()),
            tokens,
            merges: Vec::new(),
            token_type: Vec::new(),
            bos: None,
            eos: None,
            unknown: None,
            add_bos: None,
            chat_template: None,
        };
        crate::models::tokenizer::build(&data).expect("a byte-level tokenizer")
    }

    /// Feeds `text`'s own token ids through the incremental decoder and returns what the
    /// reader would have seen.
    fn streamed(tokenizer: &tokenizers::Tokenizer, text: &str) -> String {
        let ids = tokenizer
            .encode(text, false)
            .expect("encodes")
            .get_ids()
            .to_vec();
        assert!(
            ids.len() > text.chars().count(),
            "the fixture must split a character across tokens: {ids:?}"
        );
        let mut stream = Incremental::default();
        let mut seen = String::new();
        for id in ids {
            if let Some(fragment) = stream.push(tokenizer, id).expect("decodes") {
                seen.push_str(&fragment);
            }
        }
        seen
    }

    /// The interface is French, so accented output is the normal case. `é` arrives as two
    /// byte-level tokens; the decoder must not show the first one alone — `ByteLevel` decodes
    /// lossily, so a half character comes back as a 3-byte replacement glyph, and the decoded
    /// text then *shrinks* when the character completes.
    #[test]
    fn an_accented_character_split_across_tokens_arrives_whole() {
        let tokenizer = byte_level_tokenizer();

        let seen = streamed(&tokenizer, "éxy");

        assert_eq!(seen, "éxy");
        assert!(
            !seen.contains(char::REPLACEMENT_CHARACTER),
            "a half-decoded character reached the reader: {seen:?}"
        );
    }

    /// An emoji is four bytes, so it straddles four tokens: the same failure, longer.
    #[test]
    fn an_emoji_split_across_tokens_arrives_whole() {
        let tokenizer = byte_level_tokenizer();

        let seen = streamed(&tokenizer, "ok 😀 !");

        assert_eq!(seen, "ok 😀 !");
        assert!(
            !seen.contains(char::REPLACEMENT_CHARACTER),
            "a half-decoded character reached the reader: {seen:?}"
        );
    }

    /// A vocabulary shaped like a real GGUF's: the control tokens are plain entries, and no
    /// merge reaches them.
    fn chatml_vocabulary() -> tokenizers::Tokenizer {
        chatml_vocabulary_except(None)
    }

    /// The same vocabulary with one control token left out, as a derivative GGUF can ship.
    fn chatml_vocabulary_without(absent: &str) -> tokenizers::Tokenizer {
        chatml_vocabulary_except(Some(absent))
    }

    fn chatml_vocabulary_except(absent: Option<&str>) -> tokenizers::Tokenizer {
        let mut tokens: Vec<String> =
            "<|im_start_end>user"
                .chars()
                .fold(Vec::new(), |mut acc, c| {
                    let s = c.to_string();
                    if !acc.contains(&s) {
                        acc.push(s);
                    }
                    acc
                });
        for token in ["<|im_start|>", "<|im_end|>"] {
            if absent != Some(token) {
                tokens.push(token.to_owned());
            }
        }
        let data = crate::models::tokenizer::TokenizerData {
            model: "gpt2".to_owned(),
            pre: Some("qwen2".to_owned()),
            tokens,
            merges: Vec::new(),
            token_type: Vec::new(),
            bos: None,
            eos: None,
            unknown: None,
            add_bos: None,
            chat_template: None,
        };
        crate::models::tokenizer::build(&data).expect("a byte-level tokenizer")
    }

    /// candle implements no i-quant at all, so an `IQ*` file dies inside its tensor-info table
    /// with `unknown dtype for tensor 16` — an English sentence naming an internal index, which
    /// tells the user neither what is wrong nor what to do. The header already says `IQ1_S`
    /// before candle opens anything, so the refusal happens here, in French, naming both the
    /// quantization the file carries and one that would work.
    #[tokio::test]
    async fn a_quantization_the_engine_cannot_read_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let folder = dir.path().join("unsloth").join("Qwen3-4B-GGUF");
        std::fs::create_dir_all(&folder).expect("creates the folders");
        // 24 is `IQ1_S`, the value the user's own Qwen3-4B file carries.
        std::fs::write(folder.join("q.gguf"), gguf_header_quantized("qwen3", 24))
            .expect("writes the file");

        let outcome = LocalClient::new(dir.path())
            .chat_stream(ChatRequest {
                model: "unsloth/Qwen3-4B-GGUF/q.gguf".to_owned(),
                messages: Vec::new(),
                tools: Vec::new(),
            })
            .await;
        let Err(error) = outcome else {
            panic!("candle cannot read an i-quant");
        };

        assert_eq!(
            error.to_string(),
            "quantization « IQ1_S » non prise en charge par le moteur local : \
             il faut un K-quant, par exemple Q4_K_M"
        );
    }
}
