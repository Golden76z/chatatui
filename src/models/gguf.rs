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
    // Seeking past the end of a stream succeeds, so skipping a value is only safe once the
    // end is known: without this a file whose last value is missing would parse as `Ok`.
    let end = stream_end(reader)?;
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
                None => skip_value(reader, kind, end)?,
            },
            _ => match read_unsigned(reader, kind)? {
                Some(value) => suffixed.push((key, value)),
                None => skip_value(reader, kind, end)?,
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
        // 19 through 31 are the i-quants, except 21, which llama.cpp slotted a K-quant into.
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
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
        // The i-quants, whose tensor-type numbering is its own and not the `file_type` one.
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
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
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(kind, _)| kind)
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
fn skip_value<R: Read + Seek>(reader: &mut R, kind: u32, end: u64) -> Result<(), ModelError> {
    match kind {
        6 => skip(reader, 4, end),
        12 => skip(reader, 8, end),
        8 => {
            let length = bounded(read_u64(reader)?, MAX_STRING, "longueur de chaîne")?;
            skip(reader, length, end)
        }
        9 => {
            let element = read_u32(reader)?;
            let count = bounded(read_u64(reader)?, MAX_ARRAY, "taille de tableau")?;
            match fixed_width(element) {
                // Fixed-width elements: one seek over the lot.
                Some(width) => skip(reader, count.saturating_mul(width), end),
                // Strings are length-prefixed, so each one has to be stepped over.
                None if element == 8 => {
                    for _ in 0..count {
                        let length = bounded(read_u64(reader)?, MAX_STRING, "longueur de chaîne")?;
                        skip(reader, length, end)?;
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
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn bounded(value: u64, limit: u64, what: &str) -> Result<u64, ModelError> {
    if value > limit {
        return Err(ModelError::Gguf(format!("{what} invalide ({value})")));
    }
    Ok(value)
}

/// Length of the whole stream, leaving the cursor where it was.
fn stream_end<R: Seek>(reader: &mut R) -> Result<u64, ModelError> {
    let here = position(reader)?;
    let end = reader
        .seek(SeekFrom::End(0))
        .map_err(|e| ModelError::Gguf(e.to_string()))?;
    reader
        .seek(SeekFrom::Start(here))
        .map_err(|e| ModelError::Gguf(e.to_string()))?;
    Ok(end)
}

fn position<R: Seek>(reader: &mut R) -> Result<u64, ModelError> {
    reader
        .stream_position()
        .map_err(|e| ModelError::Gguf(e.to_string()))
}

/// Moves `bytes` forward, refusing to land past `end`: a value the file does not actually
/// hold is a truncated file, not a value to ignore.
fn skip<R: Read + Seek>(reader: &mut R, bytes: u64, end: u64) -> Result<(), ModelError> {
    let from = position(reader)?;
    let target = from
        .checked_add(bytes)
        .filter(|target| *target <= end)
        .ok_or_else(|| {
            ModelError::Gguf(format!(
                "fichier tronqué (valeur de {bytes} octets au-delà de la fin)"
            ))
        })?;
    reader
        .seek(SeekFrom::Start(target))
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

    #[test]
    fn accepts_version_2_and_skips_the_vocabulary() {
        let bytes = gguf(
            2,
            &[
                ("general.architecture", 8, string("qwen2")),
                (
                    "tokenizer.ggml.tokens",
                    9,
                    string_array(&["<s>", "hello", "world"]),
                ),
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

    /// Review Focus 5: a value whose bytes are simply not there must fail, even though
    /// seeking past the end of a file is not an error by itself.
    #[test]
    fn rejects_a_value_that_runs_past_the_end_of_the_file() {
        // A string of 1000 bytes announced, and not one byte of it present.
        let declared = 1000u64.to_le_bytes().to_vec();
        let bytes = gguf(3, &[("general.license", 8, declared)], &[]);

        let error = parse(&mut Cursor::new(bytes)).expect_err("rejected");

        assert!(matches!(error, ModelError::Gguf(_)), "{error:?}");
    }

    /// Same class, through the one-seek path arrays take.
    #[test]
    fn rejects_an_array_that_runs_past_the_end_of_the_file() {
        // A million u32 announced, and none of them present.
        let mut declared = 4u32.to_le_bytes().to_vec();
        declared.extend(1_000_000u64.to_le_bytes());
        let bytes = gguf(3, &[("general.tags", 9, declared)], &[]);

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
        let bytes = gguf(
            3,
            &[("general.quantization_version", 99, vec![0u8; 4])],
            &[],
        );

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

    /// `general.file_type` 24 is `IQ1_S`. The i-quants occupy 19 through 31 in llama.cpp's
    /// `llama_ftype`, a range this table used to skip entirely — so an i-quant file came back
    /// with no quantization at all, and neither `/models` nor the local engine could name what
    /// they were looking at. Verified against a real file: `Qwen3-4B-UD-IQ1_S.gguf` carries 24.
    #[test]
    fn an_i_quant_names_itself_rather_than_coming_back_blank() {
        let bytes = gguf(
            3,
            &[
                ("general.architecture", 8, string("qwen3")),
                ("general.file_type", 4, u32_value(24)),
            ],
            &[],
        );

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        assert_eq!(meta.quantization.as_deref(), Some("IQ1_S"));
    }

    /// The fallback table has its own numbering: `IQ1_S` is 19 as a tensor type and 24 as a
    /// `general.file_type`. A file that omits `file_type` — some conversion tools do — is named
    /// from its dominant tensor type, so that table needs the i-quants as much as the other one.
    #[test]
    fn an_i_quant_is_named_from_its_tensors_when_the_file_type_is_missing() {
        let bytes = gguf(
            3,
            &[("general.architecture", 8, string("qwen3"))],
            &[
                ("blk.0.attn_norm.weight", vec![2560], 0),
                ("blk.0.attn_q.weight", vec![2560, 2560], 19),
                ("blk.0.attn_k.weight", vec![2560, 2560], 19),
            ],
        );

        let meta = parse(&mut Cursor::new(bytes)).expect("parses");

        assert_eq!(meta.quantization.as_deref(), Some("IQ1_S"));
    }
}
