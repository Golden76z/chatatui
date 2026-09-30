//! Modal popups drawn over the screen. Only one is open at a time and it has the focus.

use super::{ModelPicker, Palette};

/// The open popup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Overlay {
    ModelPicker(ModelPicker),
    Palette(Palette),
    /// Help screen, scrolled by this many lines.
    Help {
        scroll: u16,
    },
    /// Context usage details (/context).
    Context {
        scroll: u16,
    },
    /// The prompt sent to the model (/prompt).
    Prompt {
        scroll: u16,
    },
    /// Indexed document collections (/collections).
    Collections {
        scroll: u16,
    },
    /// MCP servers and their tools (/mcp).
    Mcp {
        scroll: u16,
    },
    /// Two replies side by side (/compare).
    Compare {
        scroll: u16,
    },
    /// The model asks to run a tool: allow, always allow or refuse.
    ToolConfirm {
        /// What the call does (`lire ~/notes.md`).
        description: String,
        tool: String,
        /// The result will be sent to a cloud provider.
        cloud: bool,
    },
}

/// How the keymap should treat the open popup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayKind {
    /// Filterable list: arrows move, letters filter, Enter selects.
    List,
    /// Read-only text: arrows scroll, most keys close.
    Text,
    /// A question: Enter / `o` yes, `t` always, Esc / `n` no.
    Confirm,
    /// Two replies: ← / 1 and → / 2 keep one, arrows scroll.
    Compare,
}

impl Overlay {
    /// Keyboard behaviour of this popup.
    pub fn kind(&self) -> OverlayKind {
        match self {
            Self::ModelPicker(_) | Self::Palette(_) => OverlayKind::List,
            Self::Help { .. }
            | Self::Context { .. }
            | Self::Prompt { .. }
            | Self::Collections { .. }
            | Self::Mcp { .. } => OverlayKind::Text,
            Self::ToolConfirm { .. } => OverlayKind::Confirm,
            Self::Compare { .. } => OverlayKind::Compare,
        }
    }

    /// Scroll position of a text popup.
    pub fn scroll_mut(&mut self) -> Option<&mut u16> {
        match self {
            Self::Help { scroll }
            | Self::Context { scroll }
            | Self::Prompt { scroll }
            | Self::Collections { scroll }
            | Self::Mcp { scroll }
            | Self::Compare { scroll } => Some(scroll),
            Self::ModelPicker(_) | Self::Palette(_) | Self::ToolConfirm { .. } => None,
        }
    }
}
