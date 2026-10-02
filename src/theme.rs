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
///
/// Every tone is indexed on purpose. The eight basic ANSI colours are not colours: their
/// actual tone is whatever the user's terminal palette says, so `Color::Cyan` is a
/// different blue under every popular theme. Monochrome plus one accent: body text uses the
/// terminal's default foreground and never appears here.
pub const DARK: Palette = Palette {
    dim: Color::Indexed(245),
    accent: Color::Indexed(110),
    warn: Color::Indexed(179),
    error: Color::Indexed(167),
    ok: Color::Indexed(108),
    info: Color::Indexed(110),
    assistant: Color::Indexed(110),
    selection_bg: Color::Indexed(238),
    bar_bg: Color::Indexed(234),
    badge_fg: Color::Indexed(234),
    badge_bg: Color::Indexed(245),
    code_theme: "base16-ocean.dark",
};

/// For light backgrounds: the same reduced vocabulary in tones that stay readable on white.
pub const LIGHT: Palette = Palette {
    dim: Color::Indexed(245),
    accent: Color::Indexed(25),
    warn: Color::Indexed(130),
    error: Color::Indexed(160),
    ok: Color::Indexed(28),
    info: Color::Indexed(25),
    assistant: Color::Indexed(25),
    selection_bg: Color::Indexed(254),
    bar_bg: Color::Indexed(254),
    badge_fg: Color::Indexed(255),
    badge_bg: Color::Indexed(245),
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

    /// A basic ANSI colour is not a colour: its tone comes from the terminal's own palette,
    /// so the same build looks different under Gruvbox, Dracula or Solarized. Both palettes
    /// must pin indexed tones.
    ///
    /// This only inspects the eleven `Palette` constants. It says nothing about the call
    /// sites, where a renderer can name `Color::Cyan` or call `.dark_gray()` and never come
    /// near a palette — three lines of the welcome screen did exactly that for a whole
    /// milestone. Those are covered by [`no_basic_ansi_colour_is_named_in_src`].
    #[test]
    fn every_colour_is_pinned_not_inherited() {
        for (name, palette) in [("DARK", DARK), ("LIGHT", LIGHT)] {
            for (field, colour) in [
                ("dim", palette.dim),
                ("accent", palette.accent),
                ("warn", palette.warn),
                ("error", palette.error),
                ("ok", palette.ok),
                ("info", palette.info),
                ("assistant", palette.assistant),
                ("selection_bg", palette.selection_bg),
                ("bar_bg", palette.bar_bg),
                ("badge_fg", palette.badge_fg),
                ("badge_bg", palette.badge_bg),
            ] {
                assert!(
                    matches!(colour, Color::Indexed(_)),
                    "{name}.{field} is {colour:?}, not an indexed tone"
                );
            }
        }
    }

    /// Monochrome plus one accent: `accent`, `info` and `assistant` are the same family, so
    /// the interface does not read as three competing hues.
    #[test]
    fn the_accent_family_is_one_hue() {
        assert_eq!(DARK.accent, DARK.assistant);
        assert_eq!(LIGHT.accent, LIGHT.assistant);
    }

    /// The basic ANSI colours, as `Color::` variants and as the `Stylize` shorthands that
    /// set them.
    const BASIC: [(&str, &str); 16] = [
        ("Black", "black"),
        ("Red", "red"),
        ("Green", "green"),
        ("Yellow", "yellow"),
        ("Blue", "blue"),
        ("Magenta", "magenta"),
        ("Cyan", "cyan"),
        ("Gray", "gray"),
        ("DarkGray", "dark_gray"),
        ("LightRed", "light_red"),
        ("LightGreen", "light_green"),
        ("LightYellow", "light_yellow"),
        ("LightBlue", "light_blue"),
        ("LightMagenta", "light_magenta"),
        ("LightCyan", "light_cyan"),
        ("White", "white"),
    ];

    /// Every `.rs` file under `src/`, recursively.
    fn sources(dir: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                sources(&path, found);
            } else if path.extension().is_some_and(|e| e == "rs") {
                found.push(path);
            }
        }
    }

    /// Guards the call sites, which no other test can see.
    ///
    /// `every_colour_is_pinned_not_inherited` checks the `Palette` constants, and no
    /// snapshot in this repository records colour at all — a terminal-dependent tone used
    /// directly in a renderer is therefore invisible to the whole suite. Reading the sources
    /// is the only thing left that can see it, so that is what this does: it greps `src/`
    /// for the basic ANSI colours, in both the `Color::Cyan` and the `.dark_gray()` spelling,
    /// outside comments. Use `theme::palette()` instead; add a tone to `Palette` if none fits.
    #[test]
    fn no_basic_ansi_colour_is_named_in_src() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        sources(&root, &mut files);
        assert!(files.len() > 20, "no sources read from {}", root.display());

        let mut offences = Vec::new();
        for file in &files {
            let text = std::fs::read_to_string(file).expect("a source file reads back");
            for (number, line) in text.lines().enumerate() {
                // Comments may name them: this file's own documentation does.
                let code = line.split("//").next().unwrap_or_default();
                for (variant, method) in BASIC {
                    let named = code
                        .split(&format!("Color::{variant}"))
                        .skip(1)
                        .any(|rest| !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_'));
                    if named
                        || code.contains(&format!(".{method}()"))
                        || code.contains(&format!(".on_{method}()"))
                    {
                        let path = file.strip_prefix(&root).unwrap_or(file);
                        offences.push(format!("{}:{}: {variant}", path.display(), number + 1));
                    }
                }
            }
        }
        assert!(
            offences.is_empty(),
            "basic ANSI colours are terminal-dependent; use theme::palette():\n{}",
            offences.join("\n")
        );
    }

    #[test]
    fn code_themes_exist() {
        let themes = syntect::highlighting::ThemeSet::load_defaults().themes;
        assert!(themes.contains_key(DARK.code_theme));
        assert!(themes.contains_key(LIGHT.code_theme));
    }
}
