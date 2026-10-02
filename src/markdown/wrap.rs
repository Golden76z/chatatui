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
///
/// French typography forbids a break before `:`, `;`, `!`, `?` and `»`, and after `«`, so
/// the space on that side is not a break opportunity: see [`units`].
pub fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
    let mut line: Vec<Span<'static>> = Vec::new();
    let mut used = 0;

    for unit in units(spans) {
        let w: usize = unit.spans.iter().map(|s| display_width(&s.content)).sum();
        if unit.is_space {
            if used > 0 && used + w <= width {
                line.extend(unit.spans);
                used += w;
            }
            continue;
        }
        if used + w <= width {
            line.extend(unit.spans);
            used += w;
            continue;
        }
        if used > 0 && w <= width {
            lines.push(finish_line(std::mem::take(&mut line)));
            line.extend(unit.spans);
            used = w;
            continue;
        }
        // Too long for any line: fill the current one, then continue on new ones. A glued
        // run (`pas :`) wider than a very narrow pane lands here too — there is no legal
        // break point left, so it is split like any over-long word rather than looped over.
        for span in &unit.spans {
            for c in span.content.chars() {
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

/// Marks that may not open a line in French; the space before them is not a break.
const NO_BREAK_BEFORE: [char; 5] = [':', ';', '!', '?', '»'];
/// Marks that may not end a line in French; the space after them is not a break.
const NO_BREAK_AFTER: [char; 1] = ['«'];

/// A run of spans the wrapper treats as one indivisible item.
struct Unit {
    spans: Vec<Span<'static>>,
    is_space: bool,
}

/// Splits `spans` into the items the wrapper may break between.
///
/// A word, or a stretch of whitespace — except that a space sitting between a word and one
/// of [`NO_BREAK_BEFORE`], or just after [`NO_BREAK_AFTER`], is glued into the surrounding
/// word instead of becoming a break opportunity. `n'importe pas :` must not leave its colon
/// alone on the next line; `« citation »` must not leave its guillemets behind.
fn units(spans: &[Span<'static>]) -> Vec<Unit> {
    let items: Vec<(Span<'static>, bool)> = spans
        .iter()
        .flat_map(|span| {
            segments(&span.content)
                .map(|(segment, is_space)| (Span::styled(segment.to_owned(), span.style), is_space))
                .collect::<Vec<_>>()
        })
        .collect();

    // Only a space between two words can be glued: a leading one has nothing to hold on
    // to, and dropping it is what every wrapper does.
    let glued: Vec<bool> = items
        .iter()
        .enumerate()
        .map(|(i, (_, is_space))| {
            let Some(previous) = (i > 0 && !items[i - 1].1).then(|| &items[i - 1].0) else {
                return false;
            };
            *is_space
                && (previous.content.ends_with(NO_BREAK_AFTER)
                    || items.get(i + 1).is_some_and(|(next, next_space)| {
                        !next_space && next.content.starts_with(NO_BREAK_BEFORE)
                    }))
        })
        .collect();

    let mut units: Vec<Unit> = Vec::new();
    for (i, (span, is_space)) in items.iter().enumerate() {
        // The glued space joins the word before it, and the mark after it joins them both.
        let append = glued[i] || (!*is_space && i > 0 && glued[i - 1]);
        match units.last_mut() {
            Some(last) if append && !last.is_space => last.spans.push(span.clone()),
            _ => units.push(Unit {
                spans: vec![span.clone()],
                is_space: *is_space,
            }),
        }
    }
    units
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

/// Removes trailing whitespace left before a line break.
///
/// A space is usually its own span, but a glued run split character by character (`pas :`
/// in a pane too narrow for it) can leave one at the end of a span that also holds text.
fn finish_line(mut line: Vec<Span<'static>>) -> Vec<Span<'static>> {
    while let Some(last) = line.last_mut() {
        let kept = last.content.trim_end().len();
        if kept == 0 {
            line.pop();
            continue;
        }
        if kept < last.content.len() {
            let trimmed = last.content[..kept].to_owned();
            last.content = trimmed.into();
        }
        break;
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

    /// French typography forbids a break before `:`, `;`, `!`, `?` and `»`: the space in
    /// front of the mark is not a break opportunity, so the mark travels with its word
    /// instead of opening the next line on its own.
    #[test]
    fn french_punctuation_does_not_open_a_line() {
        // Without the rule this breaks after "pas", leaving ":" alone.
        let spans = [Span::raw("si l'ordre n'importe pas :")];
        assert_eq!(
            text(&wrap_spans(&spans, 24)),
            vec!["si l'ordre n'importe", "pas :"]
        );
        for mark in [":", ";", "!", "?", "»"] {
            let spans = [Span::raw(format!("alpha bravo {mark}"))];
            assert_eq!(
                text(&wrap_spans(&spans, 11)),
                vec!["alpha".to_owned(), format!("bravo {mark}")],
                "no break before {mark}"
            );
        }
        // And symmetrically, no break after an opening guillemet.
        let spans = [Span::raw("il a dit « bonjour »")];
        assert_eq!(
            text(&wrap_spans(&spans, 11)),
            vec!["il a dit", "« bonjour »"]
        );
    }

    /// The glue spans styles, so `**gras** !` holds together too.
    #[test]
    fn the_glue_crosses_a_style_boundary() {
        let spans = [Span::raw("alpha bravo").bold(), Span::raw(" !")];
        let lines = wrap_spans(&spans, 11);
        assert_eq!(text(&lines), vec!["alpha", "bravo !"]);
        assert_eq!(lines[1][0].style, Style::default().bold());
    }

    /// A pane too narrow for the glued run has no legal break point left. It must split it
    /// like any over-long word — neither looping nor panicking — and leave no trailing space.
    #[test]
    fn a_pane_too_narrow_for_the_glued_run_still_terminates() {
        let spans = [Span::raw("alpha bravo :")];
        // Four columns: the run is split mid-word like any over-long one, and the colon does
        // end up alone — there is no other break point to prefer.
        assert_eq!(
            text(&wrap_spans(&spans, 4)),
            vec!["alph", "a br", "avo", ":"]
        );
        for width in 1..=8 {
            let lines = wrap_spans(&spans, width);
            for line in &lines {
                let rendered: String = line.iter().map(|s| s.content.as_ref()).collect();
                assert!(display_width(&rendered) <= width, "{width}: {rendered:?}");
                assert_eq!(rendered.trim_end(), rendered, "{width}: trailing space");
            }
        }
    }

    #[test]
    fn plain_wrap_preserves_indentation() {
        let spans = [Span::raw("    let x = 1;")];
        assert_eq!(text(&wrap_plain(&spans, 8)), vec!["    let ", "x = 1;"]);
    }
}
