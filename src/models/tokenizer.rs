//! Builds a tokenizer from the metadata candle already parsed out of a GGUF.
//!
//! No companion `tokenizer.json`: one GGUF is the whole input, which is the point of the
//! milestone. The split between [`from_metadata`] and [`build`] is deliberate — the first
//! touches candle's types and the second is pure, so the interesting half is testable
//! without a model file.

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
    /// `tokenizer.ggml.pre`: which pre-tokenizer the family's vocabulary was trained with
    /// (`"qwen2"`, `"llama3"`, `"default"`…). Two files can both say `gpt2` here and split
    /// text differently, so this key decides the regex, not `model`.
    pub pre: Option<String>,
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
        pre: metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok())
            .cloned(),
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
        .cloned()
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
        .map(|value| value.to_string().cloned().map_err(|_| wrong_type(key)))
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

/// The pre-tokenizer pattern Qwen2 and Qwen3 were trained with, as their own
/// `tokenizer.json` spells it.
///
/// It differs from GPT-2's in two ways that matter: `\p{N}` splits every digit on its own
/// where GPT-2's `\p{N}+` groups a whole run, and `[^\r\n\p{L}\p{N}]?\p{L}+` keeps a newline
/// out of a word's leading character. `(?i:…)` and the `\s+(?!\S)` lookahead both need a
/// backtracking engine; `tokenizers`' `SysRegex` is onig here, because `candle-core` declares
/// `tokenizers` with the `onig` feature, so the pattern is expressible as written.
const QWEN2_PRE_TOKENIZER: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Builds the tokenizer. Pure: no file, no candle types, no I/O.
///
/// Only byte-level BPE (`gpt2`) is in scope. A `llama`/`spm` vocabulary pushed through a
/// byte-level BPE does not error — it produces plausible nonsense — so it is refused by name.
///
/// The same applies one level down, which is why `pre` is checked as strictly as `model`:
/// `tokenizer.ggml.model` only says the vocabulary is byte-level BPE, and GPT-2 and Qwen2
/// split the same text differently under that one name. An absent `pre` is refused rather
/// than guessed, because the only safe guess would be the GPT-2 default and a Qwen file that
/// forgot to declare itself would then be tokenized wrongly in silence — the same reasoning
/// that refuses an unsupported `model` and an unsupported architecture: unavailable beats
/// silently wrong.
pub fn build(data: &TokenizerData) -> Result<tokenizers::Tokenizer, ModelError> {
    if data.model != "gpt2" {
        return Err(ModelError::Unsupported(format!(
            "tokenizer « {} » non pris en charge",
            data.model
        )));
    }
    match data.pre.as_deref() {
        Some("qwen2") => {}
        Some(other) => {
            return Err(ModelError::Unsupported(format!(
                "pré-tokeniseur « {other} » non pris en charge"
            )));
        }
        None => {
            return Err(ModelError::Unsupported(
                "pré-tokeniseur non déclaré (tokenizer.ggml.pre) : non pris en charge".to_owned(),
            ));
        }
    }

    // `tokenizers` keys its vocabulary with its own `AHashMap` alias, and a `std` `HashMap`
    // does not convert into it: collect straight into the crate's own type.
    let vocab: tokenizers::models::bpe::Vocab = data
        .tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            let id =
                u32::try_from(id).map_err(|_| ModelError::Gguf("vocabulaire trop grand".into()))?;
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
        .build()
        .map_err(|e| ModelError::Gguf(format!("vocabulaire illisible : {e}")))?;
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);

    // Qwen's own two-stage pre-tokenizer: its pattern splits the text, then `ByteLevel` maps
    // the bytes with no regex of its own and no prefix space. `ByteLevel::default()` would be
    // GPT-2's — `add_prefix_space: true, use_regex: true` — which inserts a space after every
    // control token and groups digits.
    let split = tokenizers::pre_tokenizers::split::Split::new(
        // `&str` would be taken as a literal and escaped: the pattern must be named as one.
        tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN2_PRE_TOKENIZER.to_owned()),
        tokenizers::SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| ModelError::Gguf(format!("motif de pré-tokenisation invalide : {e}")))?;
    // `trim_offsets` only moves the offsets reported alongside the ids, which nothing here
    // reads: the engine uses `get_ids` and `decode`.
    let bytes = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::sequence::Sequence::new(
        vec![split.into(), bytes.into()],
    )));
    // The decoder is the same type and must match, or decoding re-reads the mapping with
    // GPT-2's settings.
    tokenizer.with_decoder(Some(bytes));
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::quantized::gguf_file::Value;

    use super::*;

    /// A metadata map shaped like a real Qwen3 GGUF's, small enough to read.
    ///
    /// Shaped for the lifting tests, which care about which keys are read and what happens
    /// when one is absent or of the wrong type. It is deliberately *not* a runnable
    /// vocabulary — see [`runnable_metadata`] for that.
    fn metadata() -> HashMap<String, Value> {
        HashMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                Value::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre".to_owned(),
                Value::String("qwen2".to_owned()),
            ),
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
            (
                "tokenizer.ggml.add_bos_token".to_owned(),
                Value::Bool(false),
            ),
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
        assert_eq!(
            data.pre.as_deref(),
            Some("qwen2"),
            "the file names its own pre-tokenizer and it decides the split"
        );
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

    /// A vocabulary that can actually tokenize, which [`metadata`] cannot.
    ///
    /// Byte-level BPE is bottom-up: it starts from the individual characters of a pretoken
    /// and applies merges by rank, so every starting character must be a vocabulary entry
    /// and every merge result must be one too. A vocabulary of whole words with no
    /// single-character entries encodes to nothing at all — there is no symbol to begin
    /// from — which is why this fixture spells the chain out in full.
    ///
    /// `Ġ` is the byte-level spelling of byte 0x20, so these entries describe `" bonjour"`:
    /// `Ġ`+`b` → `Ġb`, `o`+`n` → `on`, `Ġb`+`on` → `Ġbon`, `j`+`o` → `jo`, `u`+`r` → `ur`,
    /// `jo`+`ur` → `jour`, and finally the plan's own `Ġbon`+`jour` → `Ġbonjour`.
    fn runnable_metadata() -> HashMap<String, Value> {
        let tokens = [
            "<|endoftext|>",
            // The starting alphabet.
            "Ġ",
            "b",
            "o",
            "n",
            "j",
            "u",
            "r",
            // Every merge result, which BPE requires to be in the vocabulary.
            "Ġb",
            "on",
            "Ġbon",
            "jo",
            "ur",
            "jour",
            "Ġbonjour",
        ];
        // Rank order matters: BPE applies the lowest-ranked applicable merge first.
        let merges = ["Ġ b", "o n", "Ġb on", "j o", "u r", "jo ur", "Ġbon jour"];

        HashMap::from([
            (
                "tokenizer.ggml.model".to_owned(),
                Value::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre".to_owned(),
                Value::String("qwen2".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                Value::Array(
                    tokens
                        .iter()
                        .map(|t| Value::String((*t).to_owned()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges".to_owned(),
                Value::Array(
                    merges
                        .iter()
                        .map(|m| Value::String((*m).to_owned()))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.eos_token_id".to_owned(), Value::U32(0)),
        ])
    }

    fn data() -> TokenizerData {
        from_metadata(&runnable_metadata()).expect("well-formed metadata")
    }

    /// The text is `" bonjour"` with a leading space, not the literal `"Ġbonjour"`: `Ġ` is
    /// the byte-level *spelling* of byte 0x20, so feeding the glyph itself would encode the
    /// two UTF-8 bytes of `Ġ` and never reach the token the vocabulary holds.
    #[test]
    fn encodes_and_decodes_a_round_trip() {
        let tokenizer = build(&data()).expect("a gpt2 vocabulary");

        let encoded = tokenizer.encode(" bonjour", false).expect("encodes");
        assert!(!encoded.get_ids().is_empty(), "something was encoded");

        let decoded = tokenizer.decode(encoded.get_ids(), false).expect("decodes");
        assert_eq!(decoded, " bonjour", "the round trip is lossless");
    }

    #[test]
    fn applies_the_merge_it_was_given() {
        let tokenizer = build(&data()).expect("a gpt2 vocabulary");

        let encoded = tokenizer.encode(" bonjour", false).expect("encodes");

        assert_eq!(
            encoded.get_ids().len(),
            1,
            "`Ġbon` + `jour` merge into one token: {:?}",
            encoded.get_tokens()
        );
    }

    /// A vocabulary holding every byte-level character, so that any text at all encodes.
    ///
    /// The single merge is the instrument: GPT-2's `\p{N}+` hands BPE the whole run of digits
    /// and the merge fires, while Qwen's `\p{N}` hands it one digit at a time and it cannot.
    fn byte_level_data() -> TokenizerData {
        let mut tokens: Vec<String> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect();
        tokens.sort();
        tokens.push("12".to_owned());
        tokens.push("<|im_start|>".to_owned());
        tokens.push("<|im_end|>".to_owned());
        TokenizerData {
            model: "gpt2".to_owned(),
            pre: Some("qwen2".to_owned()),
            tokens,
            merges: vec!["1 2".to_owned()],
            token_type: Vec::new(),
            bos: None,
            eos: None,
            unknown: None,
            add_bos: None,
            chat_template: None,
        }
    }

    /// The prompt is ChatML, so a control token is immediately followed by a role name with
    /// no space between them. GPT-2's pre-tokenizer sets `add_prefix_space`, which inserts
    /// one into every piece that follows an added token — the model then reads
    /// `<|im_start|> system` where it trained on `<|im_start|>system`. Nothing fails; the
    /// replies are just quietly worse.
    #[test]
    fn no_space_is_inserted_after_a_control_token() {
        let mut tokenizer = build(&byte_level_data()).expect("a gpt2 vocabulary");
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("<|im_start|>", true),
            tokenizers::AddedToken::from("<|im_end|>", true),
        ]);

        let encoded = tokenizer
            .encode("<|im_start|>system", false)
            .expect("encodes");

        assert!(
            !encoded.get_tokens().iter().any(|t| t.starts_with('Ġ')),
            "a space was inserted after the control token: {:?}",
            encoded.get_tokens()
        );
        assert_eq!(
            tokenizer.decode(encoded.get_ids(), false).expect("decodes"),
            "<|im_start|>system"
        );
    }

    /// Qwen splits every digit on its own (`\p{N}`, not `\p{N}+`). Grouping them is the
    /// difference between the arithmetic the model was trained on and a plausible guess.
    #[test]
    fn digits_are_split_one_by_one() {
        let tokenizer = build(&byte_level_data()).expect("a gpt2 vocabulary");

        let encoded = tokenizer.encode("123", false).expect("encodes");

        assert_eq!(encoded.get_tokens(), ["1", "2", "3"]);
    }

    /// `tokenizer.ggml.model` says `gpt2` for every byte-level BPE, Qwen's included, so it
    /// cannot decide the regex. A Llama-3 or GPT-2 vocabulary split by Qwen's pattern would
    /// not fail either — it is the same silent degradation, one level down.
    #[test]
    fn a_pre_tokenizer_from_another_family_is_refused_by_name() {
        let mut data = byte_level_data();
        data.pre = Some("llama3".to_owned());

        let error = build(&data).expect_err("only qwen2 is in scope");

        let text = error.to_string();
        assert!(text.contains("llama3"), "{text}");
    }

    /// A file that declares no pre-tokenizer could only be guessed at, and the only guess
    /// available is GPT-2's — which is wrong for the one family in scope.
    #[test]
    fn a_file_that_declares_no_pre_tokenizer_is_refused() {
        let mut data = byte_level_data();
        data.pre = None;

        let error = build(&data).expect_err("the split cannot be guessed");

        let text = error.to_string();
        assert!(text.contains("tokenizer.ggml.pre"), "{text}");
    }

    /// An unsupported family must be named. A llama/SPM vocabulary silently fed through a
    /// byte-level BPE does not fail — it produces plausible-looking garbage, which is the
    /// worst possible outcome.
    #[test]
    fn an_unsupported_tokenizer_family_is_refused_by_name() {
        let mut data = data();
        data.model = "spm".to_owned();

        let error = build(&data).expect_err("spm is out of scope");

        // The whole string: an SPM vocabulary in a perfectly readable file must not be
        // announced as `fichier GGUF illisible`.
        assert_eq!(error.to_string(), "tokenizer « spm » non pris en charge");
    }
}
