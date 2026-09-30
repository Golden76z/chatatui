//! Colours of the interface, for dark or light terminals.
//!
//! Rendering code asks [`palette`] for semantic colours (`dim`, `accent`, `warn`…) instead
//! of naming terminal colours, so that one setting (`theme` in the configuration) adapts
//! the whole interface. The palette is chosen once at startup ([`set`]); until then, and
//! in tests, the dark one is used.

use std::sync::OnceLock;

use ratatui::style::Color;

/// Which palette to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemeName {
    /// Guess from the terminal (`COLORFGBG`), dark when unknown.
    #[default]
    Auto,
    Dark,
    Light,
}

/// Semantic colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette {
    /// Secondary text: hints, details, separators.
    pub dim: Color,
    /// Titles, borders, the user's messages.
    pub accent: Color,
    /// Warnings, cloud markers, inline code, attachments.
    pub warn: Color,
    /// Errors.
    pub error: Color,
    /// Positive state (ready).
    pub ok: Color,
    /// Links, sources, the document search.
    pub info: Color,
    /// The assistant's messages, indexing progress.
    pub assistant: Color,
    /// Background of the highlighted row of a list.
    pub selection_bg: Color,
    /// Background of the status bar.
    pub bar_bg: Color,
    /// Small badges drawn over the conversation (`↓ Ctrl+Fin`).
    pub badge_fg: Color,
    pub badge_bg: Color,
    /// syntect theme for code blocks.
    pub code_theme: &'static str,
}

/// For dark backgrounds (the default).
pub const DARK: Palette = Palette {
    dim: Color::DarkGray,
    accent: Color::Cyan,
    warn: Color::Yellow,
    error: Color::Red,
    ok: Color::Green,
    info: Color::Blue,
    assistant: Color::Magenta,
    selection_bg: Color::DarkGray,
    bar_bg: Color::Black,
    badge_fg: Color::Black,
    badge_bg: Color::Gray,
    code_theme: "base16-ocean.dark",
};

/// For light backgrounds: yellow and cyan are replaced by darker tones that stay
/// readable on white.
pub const LIGHT: Palette = Palette {
    dim: Color::Indexed(244),
    accent: Color::Indexed(25),
    warn: Color::Indexed(130),
    error: Color::Indexed(160),
    ok: Color::Indexed(28),
    info: Color::Indexed(31),
    assistant: Color::Indexed(90),
    selection_bg: Color::Indexed(254),
    bar_bg: Color::Indexed(254),
    badge_fg: Color::White,
    badge_bg: Color::Indexed(244),
    code_theme: "InspiredGitHub",
};

static CHOSEN: OnceLock<Palette> = OnceLock::new();

/// Chooses the palette (first call wins). `colorfgbg` is the `COLORFGBG` variable.
pub fn set(name: ThemeName, colorfgbg: Option<&str>) {
    let _ = CHOSEN.set(resolve(name, colorfgbg));
}

/// The palette in use.
pub fn palette() -> &'static Palette {
    CHOSEN.get().unwrap_or(&DARK)
}

/// The palette for `name`; `Auto` reads the background colour from `COLORFGBG`
/// (`"15;0"`: light text on black; `"0;15"`: dark text on white).
pub fn resolve(name: ThemeName, colorfgbg: Option<&str>) -> Palette {
    match name {
        ThemeName::Dark => DARK,
        ThemeName::Light => LIGHT,
        ThemeName::Auto => {
            let background = colorfgbg
                .and_then(|v| v.rsplit(';').next())
                .and_then(|b| b.trim().parse::<u8>().ok());
            match background {
                // White or light grey backgrounds (7 = light grey, 9-15 = bright colours).
                Some(7 | 9..=15) => LIGHT,
                _ => DARK,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_follows_the_terminal_background() {
        assert_eq!(resolve(ThemeName::Auto, Some("0;15")), LIGHT);
        assert_eq!(resolve(ThemeName::Auto, Some("15;default;0")), DARK);
        assert_eq!(resolve(ThemeName::Auto, None), DARK);
        assert_eq!(
            resolve(ThemeName::Light, Some("15;0")),
            LIGHT,
            "explicit wins"
        );
    }

    #[test]
    fn code_themes_exist() {
        let themes = syntect::highlighting::ThemeSet::load_defaults().themes;
        assert!(themes.contains_key(DARK.code_theme));
        assert!(themes.contains_key(LIGHT.code_theme));
    }
}
