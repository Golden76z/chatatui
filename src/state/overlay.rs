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
}

/// How the keymap should treat the open popup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayKind {
    /// Filterable list: arrows move, letters filter, Enter selects.
    List,
    /// Read-only text: arrows scroll, most keys close.
    Text,
}

impl Overlay {
    /// Keyboard behaviour of this popup.
    pub fn kind(&self) -> OverlayKind {
        match self {
            Self::ModelPicker(_) | Self::Palette(_) => OverlayKind::List,
            Self::Help { .. } | Self::Context { .. } | Self::Prompt { .. } => OverlayKind::Text,
        }
    }

    /// Scroll position of a text popup.
    pub fn scroll_mut(&mut self) -> Option<&mut u16> {
        match self {
            Self::Help { scroll } | Self::Context { scroll } | Self::Prompt { scroll } => {
                Some(scroll)
            }
            Self::ModelPicker(_) | Self::Palette(_) => None,
        }
    }
}
