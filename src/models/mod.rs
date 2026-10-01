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
