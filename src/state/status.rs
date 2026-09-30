//! High-level state shown in the status bar.

/// What the app is currently doing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Status {
    /// Waiting for user input.
    #[default]
    Ready,
    /// An assistant reply is being streamed.
    Generating,
    /// Confirmation of a user action (e.g. "modèle : qwen2.5").
    Info(String),
    /// The last operation failed; the message is user-facing.
    Error(String),
}
