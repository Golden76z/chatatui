//! Wrapping of styled text to a fixed display width.

use ratatui::text::Span;
use unicode_width::UnicodeWidthStr;

/// Display width of a string in terminal columns.
pub fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Word-wraps a logical line made of styled spans.
///
/// Breaks at whitespace; words wider than `width` are split. Whitespace at a break is
/// dropped. Always returns at least one (possibly empty) line.
pub fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
    let mut line: Vec<Span<'static>> = Vec::new();
    let mut used = 0;

    for span in spans {
        for (segment, is_space) in segments(&span.content) {
            let w = display_width(segment);
            if is_space {
                if used > 0 && used + w <= width {
                    line.push(Span::styled(segment.to_owned(), span.style));
                    used += w;
                }
                continue;
            }
            if used + w <= width {
                line.push(Span::styled(segment.to_owned(), span.style));
                used += w;
                continue;
            }
            if used > 0 && w <= width {
                lines.push(finish_line(std::mem::take(&mut line)));
                line.push(Span::styled(segment.to_owned(), span.style));
                used = w;
                continue;
            }
            // Word too long for any line: fill the current line, then continue on new ones.
            for c in segment.chars() {
                let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
                if used + cw > width && used > 0 {
                    lines.push(finish_line(std::mem::take(&mut line)));
                    used = 0;
                }
                push_char(&mut line, c, span.style);
                used += cw;
            }
        }
    }
    lines.push(finish_line(line));
    lines
}

/// Hard-wraps (at any character) a line whose whitespace must be preserved, e.g. code.
pub fn wrap_plain(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut line: Vec<Span<'static>> = Vec::new();
    let mut used = 0;
    for span in spans {
        for c in span.content.chars() {
            let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
            if used + cw > width && used > 0 {
                lines.push(std::mem::take(&mut line));
                used = 0;
            }
            push_char(&mut line, c, span.style);
            used += cw;
        }
    }
    lines.push(line);
    lines
}

/// Appends a character, merging it into the last span when the style matches.
fn push_char(line: &mut Vec<Span<'static>>, c: char, style: ratatui::style::Style) {
    match line.last_mut() {
        Some(last) if last.style == style => last.content.to_mut().push(c),
        _ => line.push(Span::styled(c.to_string(), style)),
    }
}

/// Removes trailing whitespace spans left before a line break.
fn finish_line(mut line: Vec<Span<'static>>) -> Vec<Span<'static>> {
    while line.last().is_some_and(|s| s.content.trim().is_empty()) {
        line.pop();
    }
    line
}

/// Splits text into alternating runs of whitespace and non-whitespace.
fn segments(text: &str) -> impl Iterator<Item = (&str, bool)> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let first = rest.chars().next()?;
        let is_space = first.is_whitespace();
        let end = rest
            .char_indices()
            .find(|(_, c)| c.is_whitespace() != is_space)
            .map_or(rest.len(), |(i, _)| i);
        let (segment, tail) = rest.split_at(end);
        rest = tail;
        Some((segment, is_space))
    })
}

#[cfg(test)]
mod tests {
    use ratatui::style::{Style, Stylize};

    use super::*;

    fn text(lines: &[Vec<Span<'static>>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn wraps_at_word_boundaries() {
        let spans = [Span::raw("the quick brown fox jumps")];
        assert_eq!(
            text(&wrap_spans(&spans, 10)),
            vec!["the quick", "brown fox", "jumps"]
        );
    }

    #[test]
    fn keeps_styles_across_breaks() {
        let spans = [Span::raw("plain "), Span::raw("bold words here").bold()];
        let lines = wrap_spans(&spans, 11);
        assert_eq!(text(&lines), vec!["plain bold", "words here"]);
        assert_eq!(lines[1][0].style, Style::default().bold());
    }

    #[test]
    fn splits_words_longer_than_the_width() {
        let spans = [Span::raw("ab abcdefghij")];
        assert_eq!(
            text(&wrap_spans(&spans, 4)),
            vec!["ab a", "bcde", "fghi", "j"]
        );
    }

    #[test]
    fn counts_wide_characters() {
        let spans = [Span::raw("日本語 テキスト")];
        assert_eq!(text(&wrap_spans(&spans, 7)), vec!["日本語", "テキス", "ト"]);
    }

    #[test]
    fn empty_input_gives_one_empty_line() {
        assert_eq!(wrap_spans(&[], 10).len(), 1);
    }

    #[test]
    fn plain_wrap_preserves_indentation() {
        let spans = [Span::raw("    let x = 1;")];
        assert_eq!(text(&wrap_plain(&spans, 8)), vec!["    let ", "x = 1;"]);
    }
}
