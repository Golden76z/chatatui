//! Local model store: GGUF files downloaded from HuggingFace, kept on disk with an
//! inventory in SQLite.
//!
//! - [`hub`]: the HuggingFace client (list a repository's GGUF files, download one);
//! - [`gguf`]: the metadata in a GGUF file's header;
//! - [`store`]: the inventory of downloaded models;
//! - [`download`]: the background job behind `/pull`;
//! - [`catalog`]: the models `/models` offers when they are not downloaded yet;
//! - [`tokenizer`]: the tokenizer a local engine needs, built from a GGUF's own metadata;
//! - [`template`]: the prompt string an architecture expects, rendered from a conversation.
//!
//! Running a model is not part of this module yet.

pub mod catalog;
pub mod download;
pub mod gguf;
pub mod hub;
pub mod store;
pub mod template;
pub mod tokenizer;

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
    /// The file itself could not be read: wrong magic, truncated, a descriptor that does not
    /// parse. Reserved for that, because the prefix asserts it.
    #[error("fichier GGUF illisible : {0}")]
    Gguf(String),
    /// The file reads perfectly well and the engine cannot run it: an architecture, a
    /// tokenizer family or a pre-tokenizer that is out of scope. Shown verbatim — calling such
    /// a file unreadable would send the user looking for a corrupt download.
    #[error("{0}")]
    Unsupported(String),
    /// The model is loaded and a generation is still using it. Shown verbatim: nothing is
    /// wrong with the file, and loading a second copy would double several gigabytes.
    #[error("{0}")]
    Busy(String),
    #[error("{0}")]
    Io(String),
    #[error("le fichier téléchargé est corrompu (empreinte sha256 incorrecte)")]
    Checksum,
}
