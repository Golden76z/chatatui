//! Markdown to terminal lines.
//!
//! [`render`] turns a markdown string into lines already wrapped for a given width, so the
//! caller knows exactly how many rows a message takes (needed for scrolling).

mod highlight;
mod render;
mod wrap;

pub use highlight::warm_up;
pub use render::render;
pub use wrap::{display_width, wrap_plain, wrap_spans};
