//! Plain data describing what the app is showing. No I/O, no rendering.

pub mod conversation;
pub mod find;
pub mod gguf_picker;
pub mod model_picker;
pub mod models_picker;
pub mod overlay;
pub mod palette;
pub mod scroll;
pub mod sidebar;
pub mod status;

pub use conversation::{Citation, Conversation, Image, Message, MessageId, MessageStatus, Role};
pub use find::{Find, FindMatch};
pub use gguf_picker::GgufPicker;
pub use model_picker::{ModelChoice, ModelList, ModelPicker};
pub use models_picker::{ModelRow, ModelsPicker};
pub use overlay::{Overlay, OverlayKind};
pub use palette::Palette;
pub use scroll::ScrollState;
pub use sidebar::Sidebar;
pub use status::Status;
