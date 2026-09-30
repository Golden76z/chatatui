//! Syntax highlighting of code blocks with syntect.
//!
//! The syntax and theme sets are loaded once, on first use (see [`warm_up`]). Highlighted
//! blocks are cached: while a reply streams, its completed code blocks are not highlighted
//! again on every refresh.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock, PoisonError},
};

use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};
use syntect::{
    easy::HighlightLines,
    highlighting::{FontStyle, Theme, ThemeSet},
    parsing::{SyntaxReference, SyntaxSet},
    util::LinesWithEndings,
};

/// Maximum number of cached blocks; the cache is simply emptied when full.
const CACHE_CAPACITY: usize = 256;

type Highlighted = Vec<Vec<Span<'static>>>;

fn cache() -> &'static Mutex<HashMap<(String, String), Highlighted>> {
    static CACHE: OnceLock<Mutex<HashMap<(String, String), Highlighted>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Loads the syntax and theme sets. Called on a background thread at startup so the first
/// code block does not stall the UI.
pub fn warm_up() {
    syntaxes();
    theme();
}

/// Like [`highlight`], but reuses the result for identical blocks. Use it only for blocks
/// that will not change (closed fences), or the cache fills with partial versions.
pub fn highlight_cached(code: &str, lang: &str) -> Highlighted {
    let key = (lang.to_owned(), code.to_owned());
    if let Some(lines) = cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return lines.clone();
    }
    let lines = highlight(code, lang);
    let mut cache = cache().lock().unwrap_or_else(PoisonError::into_inner);
    if cache.len() >= CACHE_CAPACITY {
        cache.clear();
    }
    cache.insert(key, lines.clone());
    lines
}

fn syntaxes() -> &'static SyntaxSet {
    static SYNTAXES: OnceLock<SyntaxSet> = OnceLock::new();
    SYNTAXES.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme() -> Option<&'static Theme> {
    static THEMES: OnceLock<ThemeSet> = OnceLock::new();
    THEMES
        .get_or_init(ThemeSet::load_defaults)
        .themes
        .get(crate::theme::palette().code_theme)
}

fn find_syntax(lang: &str) -> Option<&'static SyntaxReference> {
    let set = syntaxes();
    let token = lang.split([',', ' ']).next().unwrap_or_default().trim();
    if token.is_empty() {
        return None;
    }
    set.find_syntax_by_token(token)
        .or_else(|| set.find_syntax_by_token(&token.to_ascii_lowercase()))
}

/// Highlights `code` as `lang`. Returns one span list per source line (without newlines).
///
/// Unknown languages and highlighting failures fall back to unstyled text.
pub fn highlight(code: &str, lang: &str) -> Highlighted {
    let plain = || {
        code.lines()
            .map(|l| vec![Span::raw(l.to_owned())])
            .collect()
    };
    let (Some(syntax), Some(theme)) = (find_syntax(lang), theme()) else {
        return plain();
    };

    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines = Vec::new();
    for line in LinesWithEndings::from(code) {
        let Ok(ranges) = highlighter.highlight_line(line, syntaxes()) else {
            return plain();
        };
        let spans = ranges
            .into_iter()
            .filter_map(|(style, text)| {
                let text = text.trim_end_matches(['\n', '\r']);
                (!text.is_empty()).then(|| Span::styled(text.to_owned(), convert(style)))
            })
            .collect();
        lines.push(spans);
    }
    lines
}

fn convert(style: syntect::highlighting::Style) -> Style {
    let fg = style.foreground;
    let mut out = Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b));
    if style.font_style.contains(FontStyle::BOLD) {
        out = out.add_modifier(Modifier::BOLD);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        out = out.add_modifier(Modifier::ITALIC);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        out = out.add_modifier(Modifier::UNDERLINED);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Vec<Span<'static>>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn known_language_gets_colors_and_keeps_text() {
        let code = "fn main() {\n    println!(\"hi\");\n}\n";
        let lines = highlight(code, "rust");
        assert_eq!(
            text(&lines),
            vec!["fn main() {", "    println!(\"hi\");", "}"]
        );
        let colors: Vec<_> = lines.iter().flatten().filter_map(|s| s.style.fg).collect();
        assert!(colors.len() > 3, "several colored tokens");
        assert!(colors.iter().all(|c| matches!(c, Color::Rgb(..))));
    }

    #[test]
    fn language_aliases_and_attributes() {
        assert!(find_syntax("rs").is_some());
        assert!(find_syntax("Python").is_some());
        assert!(find_syntax("rust,ignore").is_some());
    }

    #[test]
    fn cached_result_matches() {
        let code = "x = 1\n";
        assert_eq!(highlight_cached(code, "python"), highlight(code, "python"));
        assert_eq!(highlight_cached(code, "python"), highlight(code, "python"));
    }

    #[test]
    fn unknown_language_is_plain() {
        let lines = highlight("a\nb", "no-such-language");
        assert_eq!(text(&lines), vec!["a", "b"]);
        assert!(lines.iter().flatten().all(|s| s.style == Style::default()));
    }
}
