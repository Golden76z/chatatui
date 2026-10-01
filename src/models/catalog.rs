//! A short, hand-picked list of GGUF repositories to offer in `/models`.
//!
//! Sorting HuggingFace by download count does not produce a list worth showing: the top of
//! the GGUF charts is embedding, speech and image models, mirrors, and "uncensored" or
//! quantization-experiment forks of a handful of base models. So this list is chosen rather
//! than measured, kept short, and ordered smallest first — the smallest model is the one
//! most likely to run on the machine at hand.
//!
//! No file names here: the repositories rename and re-quantize their files, so `/models`
//! asks the Hub for the real list and lets the user pick (see [`crate::models::hub`]).

/// One offered model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// `owner/name` on HuggingFace.
    pub repo: &'static str,
    /// What to call it on screen.
    pub name: &'static str,
    /// Parameter count, written for a human: `0,6 B`.
    pub parameters: &'static str,
    /// One line on what it is for.
    pub note: &'static str,
}

/// The offered models, smallest first.
pub fn entries() -> &'static [Entry] {
    ENTRIES
}

/// Whether `repo` is one of the offered models.
pub fn contains(repo: &str) -> bool {
    ENTRIES.iter().any(|entry| entry.repo == repo)
}

const ENTRIES: &[Entry] = &[
    Entry {
        repo: "LiquidAI/LFM2.5-350M-GGUF",
        name: "LFM2.5 350M",
        parameters: "0,35 B",
        note: "tient partout, même sans GPU",
    },
    Entry {
        repo: "unsloth/Qwen3-0.6B-GGUF",
        name: "Qwen3 0.6B",
        parameters: "0,6 B",
        note: "le plus petit Qwen3",
    },
    Entry {
        repo: "lmstudio-community/gemma-3-1b-it-GGUF",
        name: "Gemma 3 1B",
        parameters: "1 B",
        note: "petit modèle de Google",
    },
    Entry {
        repo: "unsloth/Qwen3-1.7B-GGUF",
        name: "Qwen3 1.7B",
        parameters: "1,7 B",
        note: "bon compromis sans GPU",
    },
    Entry {
        repo: "LiquidAI/LFM2.5-2.6B-GGUF",
        name: "LFM2.5 2.6B",
        parameters: "2,6 B",
        note: "rapide sur matériel modeste",
    },
    Entry {
        repo: "unsloth/Llama-3.2-3B-Instruct-GGUF",
        name: "Llama 3.2 3B Instruct",
        parameters: "3 B",
        note: "le petit Llama de Meta",
    },
    Entry {
        repo: "unsloth/Qwen3-4B-GGUF",
        name: "Qwen3 4B",
        parameters: "4 B",
        note: "polyvalent, ~3 Go en Q4_K_M",
    },
    Entry {
        repo: "unsloth/Phi-4-mini-instruct-GGUF",
        name: "Phi-4 mini Instruct",
        parameters: "3,8 B",
        note: "le petit Phi de Microsoft",
    },
    Entry {
        repo: "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF",
        name: "Qwen2.5 Coder 7B",
        parameters: "7 B",
        note: "orienté code",
    },
    Entry {
        repo: "unsloth/Qwen3-8B-GGUF",
        name: "Qwen3 8B",
        parameters: "8 B",
        note: "le Qwen3 à tout faire, ~5 Go",
    },
    Entry {
        repo: "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        name: "Llama 3.1 8B Instruct",
        parameters: "8 B",
        note: "référence stable pour comparer",
    },
    Entry {
        repo: "lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        name: "DeepSeek-R1 Qwen3 8B",
        parameters: "8 B",
        note: "raisonne avant de répondre",
    },
    Entry {
        repo: "lmstudio-community/Qwen3-14B-GGUF",
        name: "Qwen3 14B",
        parameters: "14 B",
        note: "meilleur, demande un GPU",
    },
    Entry {
        repo: "unsloth/gpt-oss-20b-GGUF",
        name: "gpt-oss 20B",
        parameters: "20 B",
        note: "le modèle ouvert d'OpenAI",
    },
    Entry {
        repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        name: "Qwen3 Coder 30B A3B",
        parameters: "30 B",
        note: "code, rapide pour sa taille",
    },
    Entry {
        repo: "unsloth/Qwen3-32B-GGUF",
        name: "Qwen3 32B",
        parameters: "32 B",
        note: "le plus capable, ~20 Go",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalog_is_not_empty() {
        assert!(entries().len() >= 10, "{} entries", entries().len());
    }

    #[test]
    fn every_repository_is_owner_slash_name() {
        for entry in entries() {
            let parts: Vec<&str> = entry.repo.split('/').collect();
            assert_eq!(parts.len(), 2, "{}", entry.repo);
            assert!(parts.iter().all(|part| !part.is_empty()), "{}", entry.repo);
            // The store joins these onto a directory; anything else would escape it.
            assert!(!entry.repo.contains(".."), "{}", entry.repo);
            assert!(!entry.repo.contains('\\'), "{}", entry.repo);
        }
    }

    #[test]
    fn no_repository_appears_twice() {
        let mut seen: Vec<&str> = entries().iter().map(|entry| entry.repo).collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total);
    }

    #[test]
    fn every_entry_is_described() {
        for entry in entries() {
            assert!(!entry.name.is_empty(), "{}", entry.repo);
            assert!(!entry.parameters.is_empty(), "{}", entry.repo);
            assert!(!entry.note.is_empty(), "{}", entry.repo);
        }
    }

    #[test]
    fn a_catalog_repository_is_recognised() {
        assert!(contains(entries()[0].repo));
        assert!(!contains("someone/not-in-the-list"));
    }
}
