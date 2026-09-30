//! Markdown event stream → wrapped, styled lines.

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::{
    highlight::{highlight, highlight_cached},
    wrap::{display_width, wrap_plain, wrap_spans},
};

const BULLETS: [&str; 3] = ["• ", "◦ ", "▪ "];

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn inline_code() -> Style {
    Style::default().fg(Color::Yellow)
}

fn heading_style(level: HeadingLevel) -> Style {
    let bold = Style::default().add_modifier(Modifier::BOLD);
    match level {
        HeadingLevel::H1 => bold.fg(Color::Cyan).add_modifier(Modifier::UNDERLINED),
        HeadingLevel::H2 => bold.fg(Color::Cyan),
        _ => bold,
    }
}

/// Renders markdown into lines no wider than `width` columns.
pub fn render(markdown: &str, width: usize) -> Vec<Line<'static>> {
    let options =
        Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
    let mut renderer = Renderer::new(width.max(1));
    let events: Vec<_> = Parser::new_ext(markdown, options)
        .into_offset_iter()
        .collect();
    // Blocks are numbered (for `/copy code N`) when there are several.
    renderer.number_code = events
        .iter()
        .filter(|(event, _)| matches!(event, Event::Start(Tag::CodeBlock(_))))
        .count()
        > 1;
    for (event, range) in events {
        if let Event::Start(Tag::CodeBlock(_)) = &event {
            renderer.code_closed = is_closed_fence(&markdown[range.clone()]);
        }
        renderer.event(event);
    }
    renderer.finish()
}

/// The code blocks of `markdown`, in order (their text, without the fences).
pub fn code_blocks(markdown: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;
    for event in Parser::new_ext(markdown, Options::empty()) {
        match event {
            Event::Start(Tag::CodeBlock(_)) => current = Some(String::new()),
            Event::Text(text) => {
                if let Some(block) = &mut current {
                    block.push_str(&text);
                }
            }
            Event::End(TagEnd::CodeBlock) => blocks.extend(current.take()),
            _ => {}
        }
    }
    blocks
}

/// `true` if a fenced code block's source ends with its closing fence (a block still being
/// streamed does not).
fn is_closed_fence(source: &str) -> bool {
    let trimmed = source.trim_end();
    let Some(fence) = trimmed.chars().next().filter(|c| *c == '`' || *c == '~') else {
        return true; // indented block: complete as far as the parser is concerned
    };
    let last_line = trimmed.lines().last().unwrap_or_default().trim();
    trimmed.lines().count() > 1 && last_line.len() >= 3 && last_line.chars().all(|c| c == fence)
}

/// A block that prefixes the lines inside it.
enum Container {
    Quote,
    /// List item; the marker is shown on its first line, spaces on the following ones.
    Item {
        marker: String,
        marker_shown: bool,
    },
}

struct CodeBlock {
    lang: String,
    text: String,
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<Vec<Span<'static>>>>,
    row: Vec<Vec<Span<'static>>>,
    cell: Vec<Span<'static>>,
    header_rows: usize,
}

struct Renderer {
    width: usize,
    lines: Vec<Line<'static>>,
    /// Current logical line (before wrapping).
    inline: Vec<Span<'static>>,
    styles: Vec<Style>,
    containers: Vec<Container>,
    /// Next number for each open list (`None` for bullet lists).
    lists: Vec<Option<u64>>,
    code: Option<CodeBlock>,
    table: Option<Table>,
    /// Destination and start index (in `inline`) of the open link.
    link: Option<(String, usize)>,
    /// A blank line must separate the next block from the previous one.
    pending_blank: bool,
    /// The current code block has its closing fence (its highlighting can be cached).
    code_closed: bool,
    /// Show each code block's number in its label.
    number_code: bool,
    /// Code blocks rendered so far.
    code_count: usize,
}

impl Renderer {
    fn new(width: usize) -> Self {
        Self {
            width,
            lines: Vec::new(),
            inline: Vec::new(),
            styles: Vec::new(),
            containers: Vec::new(),
            lists: Vec::new(),
            code: None,
            table: None,
            link: None,
            pending_blank: false,
            code_closed: false,
            number_code: false,
            code_count: 0,
        }
    }

    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn push_style(&mut self, patch: Style) {
        self.styles.push(self.style().patch(patch));
    }

    fn pop_style(&mut self) {
        self.styles.pop();
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => self.text(&text),
            Event::Code(code) => self.push_span(Span::styled(
                code.into_string(),
                self.style().patch(inline_code()),
            )),
            Event::InlineMath(math) | Event::DisplayMath(math) => {
                self.push_span(Span::styled(math.into_string(), inline_code()));
            }
            Event::Html(html) | Event::InlineHtml(html) => self.text(&html),
            Event::FootnoteReference(name) => {
                self.push_span(Span::styled(format!("[^{name}]"), dim()));
            }
            Event::SoftBreak => self.push_span(Span::styled(" ", self.style())),
            Event::HardBreak => self.flush_inline(),
            Event::Rule => {
                self.start_block();
                let width = self.width.saturating_sub(self.prefix_width()).max(1);
                let rule = vec![Span::styled("─".repeat(width), dim())];
                self.emit(rule);
                self.end_block();
            }
            Event::TaskListMarker(checked) => {
                let marker = if checked { "[x] " } else { "[ ] " };
                self.push_span(Span::styled(marker, dim()));
            }
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph | Tag::HtmlBlock => self.start_block(),
            Tag::Heading { level, .. } => {
                self.start_block();
                self.push_style(heading_style(level));
            }
            Tag::BlockQuote(_) => {
                self.start_block();
                self.containers.push(Container::Quote);
            }
            Tag::CodeBlock(kind) => {
                self.start_block();
                let lang = match kind {
                    CodeBlockKind::Fenced(lang) => lang.into_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                self.code = Some(CodeBlock {
                    lang,
                    text: String::new(),
                });
            }
            Tag::List(start) => {
                if self.in_item() {
                    // Nested list: end the parent item's text without a blank line.
                    self.flush_inline();
                } else {
                    self.start_block();
                }
                self.lists.push(start);
            }
            Tag::Item => {
                self.flush_inline();
                let depth = self.lists.len().saturating_sub(1);
                let marker = match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let marker = format!("{n}. ");
                        *n += 1;
                        marker
                    }
                    _ => BULLETS[depth % BULLETS.len()].to_owned(),
                };
                self.containers.push(Container::Item {
                    marker,
                    marker_shown: false,
                });
            }
            Tag::Emphasis => self.push_style(Style::default().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.push_style(Style::default().add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => {
                self.push_style(Style::default().add_modifier(Modifier::CROSSED_OUT));
            }
            Tag::Link { dest_url, .. } => {
                self.push_style(
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::UNDERLINED),
                );
                self.link = Some((dest_url.into_string(), self.inline.len()));
            }
            Tag::Image { .. } => {
                self.push_span(Span::styled("[image: ", dim()));
                self.push_style(Style::default().add_modifier(Modifier::ITALIC));
            }
            Tag::Table(_) => {
                self.start_block();
                self.table = Some(Table::default());
            }
            Tag::TableHead | Tag::TableRow | Tag::TableCell => {}
            Tag::FootnoteDefinition(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition
            | Tag::Superscript
            | Tag::Subscript
            | Tag::MetadataBlock(_) => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::HtmlBlock => self.end_block(),
            TagEnd::Heading(_) => {
                self.flush_inline();
                self.pop_style();
                self.end_block();
            }
            TagEnd::BlockQuote(_) => {
                self.flush_inline();
                self.containers.pop();
                self.pending_blank = true;
            }
            TagEnd::CodeBlock => {
                if let Some(code) = self.code.take() {
                    self.render_code(&code);
                }
                self.end_block();
            }
            TagEnd::List(_) => {
                self.flush_inline();
                self.lists.pop();
                self.pending_blank = !self.in_item();
            }
            TagEnd::Item => {
                self.flush_inline();
                if let Some(Container::Item {
                    marker_shown: false,
                    ..
                }) = self.containers.last()
                {
                    self.emit(Vec::new()); // empty item: still show its marker
                }
                self.containers.pop();
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => self.pop_style(),
            TagEnd::Link => {
                self.pop_style();
                if let Some((dest, start)) = self.link.take() {
                    let text: String = self.inline[start.min(self.inline.len())..]
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect();
                    if !dest.is_empty() && text != dest && !dest.starts_with('#') {
                        self.push_span(Span::styled(format!(" ({dest})"), dim()));
                    }
                }
            }
            TagEnd::Image => {
                self.pop_style();
                self.push_span(Span::styled("]", dim()));
            }
            TagEnd::TableCell => {
                if let Some(table) = &mut self.table {
                    let cell = std::mem::take(&mut table.cell);
                    table.row.push(cell);
                }
            }
            TagEnd::TableHead | TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    let row = std::mem::take(&mut table.row);
                    table.rows.push(row);
                    if tag == TagEnd::TableHead {
                        table.header_rows = table.rows.len();
                    }
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.render_table(table);
                }
                self.end_block();
            }
            TagEnd::FootnoteDefinition
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::Superscript
            | TagEnd::Subscript
            | TagEnd::MetadataBlock(_) => {}
        }
    }

    fn text(&mut self, text: &str) {
        if let Some(code) = &mut self.code {
            code.text.push_str(text);
            return;
        }
        let style = self.style();
        let mut parts = text.split('\n').peekable();
        while let Some(part) = parts.next() {
            if !part.is_empty() {
                self.push_span(Span::styled(part.to_owned(), style));
            }
            if parts.peek().is_some() {
                self.flush_inline();
            }
        }
    }

    fn push_span(&mut self, span: Span<'static>) {
        match &mut self.table {
            Some(table) => table.cell.push(span),
            None => self.inline.push(span),
        }
    }

    fn in_item(&self) -> bool {
        matches!(self.containers.last(), Some(Container::Item { .. }))
    }

    /// Prepares a new block: flushes pending text and inserts the separating blank line.
    fn start_block(&mut self) {
        self.flush_inline();
        if self.pending_blank && !self.lines.is_empty() {
            let prefix = self.quote_prefix();
            self.lines.push(Line::from(prefix));
        }
        self.pending_blank = false;
    }

    fn end_block(&mut self) {
        self.flush_inline();
        self.pending_blank = true;
    }

    /// Wraps and emits the current logical line, if any.
    fn flush_inline(&mut self) {
        if self.inline.is_empty() {
            return;
        }
        let spans = std::mem::take(&mut self.inline);
        let available = self.width.saturating_sub(self.prefix_width()).max(1);
        for line in wrap_spans(&spans, available) {
            self.emit(line);
        }
    }

    /// Emits one already-wrapped line with the container prefix.
    fn emit(&mut self, spans: Vec<Span<'static>>) {
        let mut line = self.next_prefix();
        line.extend(spans);
        self.lines.push(Line::from(line));
    }

    fn prefix_width(&self) -> usize {
        self.containers
            .iter()
            .map(|c| match c {
                Container::Quote => 2,
                Container::Item { marker, .. } => display_width(marker),
            })
            .sum()
    }

    /// Prefix for the next line; consumes list markers so they appear only once.
    fn next_prefix(&mut self) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        for container in &mut self.containers {
            match container {
                Container::Quote => spans.push(Span::styled("│ ", dim())),
                Container::Item {
                    marker,
                    marker_shown,
                } => {
                    if *marker_shown {
                        spans.push(Span::raw(" ".repeat(display_width(marker))));
                    } else {
                        spans.push(Span::styled(
                            marker.clone(),
                            Style::default().fg(Color::Blue),
                        ));
                        *marker_shown = true;
                    }
                }
            }
        }
        spans
    }

    /// Prefix for blank lines: keeps quote bars, never shows list markers.
    fn quote_prefix(&self) -> Vec<Span<'static>> {
        self.containers
            .iter()
            .map(|c| match c {
                Container::Quote => Span::styled("│", dim()),
                Container::Item { marker, .. } => Span::raw(" ".repeat(display_width(marker))),
            })
            .collect()
    }

    fn render_code(&mut self, code: &CodeBlock) {
        const GUTTER: &str = "▎ ";
        let gutter = || Span::styled(GUTTER, dim());
        let available = self
            .width
            .saturating_sub(self.prefix_width() + display_width(GUTTER))
            .max(1);
        let lang = code.lang.trim();
        self.code_count += 1;
        let number = self.number_code.then(|| format!("#{}", self.code_count));
        let label = match (lang.is_empty(), number) {
            (false, Some(number)) => Some(format!("{lang} · {number}")),
            (false, None) => Some(lang.to_owned()),
            (true, number) => number,
        };
        if let Some(label) = label {
            let label = Span::styled(label, dim().add_modifier(Modifier::ITALIC));
            self.emit(vec![gutter(), label]);
        }
        let highlighted = if self.code_closed {
            highlight_cached(&code.text, lang)
        } else {
            highlight(&code.text, lang)
        };
        for source_line in highlighted {
            for wrapped in wrap_plain(&source_line, available) {
                let mut spans = vec![gutter()];
                spans.extend(wrapped);
                self.emit(spans);
            }
        }
    }

    fn render_table(&mut self, table: Table) {
        const SEPARATOR: &str = " │ ";
        let columns = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        if columns == 0 {
            return;
        }
        let cell_width = |cell: &Vec<Span<'static>>| {
            cell.iter()
                .map(|s| display_width(&s.content))
                .sum::<usize>()
        };
        let mut widths = vec![1usize; columns];
        for row in &table.rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell_width(cell));
            }
        }
        // Shrink the widest column until the table fits.
        let available = self.width.saturating_sub(self.prefix_width()).max(1);
        let separators = display_width(SEPARATOR) * (columns - 1);
        while widths.iter().sum::<usize>() + separators > available {
            let Some((widest, &w)) = widths.iter().enumerate().max_by_key(|&(_, w)| *w) else {
                break;
            };
            if w <= 3 {
                break;
            }
            widths[widest] -= 1;
        }

        let bold = Style::default().add_modifier(Modifier::BOLD);
        for (r, row) in table.rows.iter().enumerate() {
            let header = r < table.header_rows;
            let wrapped: Vec<Vec<Vec<Span<'static>>>> = (0..columns)
                .map(|i| {
                    let cell = row.get(i).cloned().unwrap_or_default();
                    let cell: Vec<Span<'static>> = if header {
                        cell.into_iter().map(|s| s.patch_style(bold)).collect()
                    } else {
                        cell
                    };
                    wrap_spans(&cell, widths[i])
                })
                .collect();
            let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
            for line_index in 0..height {
                let mut spans = Vec::new();
                for (i, cell_lines) in wrapped.iter().enumerate() {
                    if i > 0 {
                        spans.push(Span::styled(SEPARATOR, dim()));
                    }
                    let content = cell_lines.get(line_index).cloned().unwrap_or_default();
                    let used: usize = content.iter().map(|s| display_width(&s.content)).sum();
                    spans.extend(content);
                    if i + 1 < columns {
                        spans.push(Span::raw(" ".repeat(widths[i].saturating_sub(used))));
                    }
                }
                self.emit(spans);
            }
            if header && r + 1 == table.header_rows {
                let rule = widths
                    .iter()
                    .map(|w| "─".repeat(*w))
                    .collect::<Vec<_>>()
                    .join("─┼─");
                self.emit(vec![Span::styled(rule, dim())]);
            }
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_inline();
        if let Some(code) = self.code.take() {
            self.render_code(&code);
        }
        while self
            .lines
            .last()
            .is_some_and(|l| l.spans.iter().all(|s| s.content.trim().is_empty()))
        {
            self.lines.pop();
        }
        self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn span<'a>(lines: &'a [Line<'static>], needle: &str) -> &'a Span<'static> {
        lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.contains(needle))
            .unwrap_or_else(|| panic!("no span containing {needle:?}"))
    }

    const SAMPLE: &str = r#"# Titre principal

Du texte avec du **gras**, de l'*italique*, du `code` et un [lien](https://ratatui.rs).

## Liste

- premier élément assez long pour être replié sur la ligne suivante
- second
  1. imbriqué un
  2. imbriqué deux
- [x] tâche faite

> Une citation
> sur deux lignes.

```rust
fn main() {
    println!("bonjour");
}
```

| Langage | Typage |
|---------|--------|
| Rust | statique |
| Python | dynamique |

---
Fin."#;

    #[test]
    fn sample_layout() {
        insta::assert_snapshot!(text(&render(SAMPLE, 40)));
    }

    #[test]
    fn no_line_exceeds_the_width() {
        for width in [12, 20, 40, 80] {
            for line in render(SAMPLE, width) {
                assert!(line.width() <= width, "{:?} wider than {width}", line);
            }
        }
    }

    #[test]
    fn inline_styles() {
        let lines = render("**gras** *italique* `code` ~~barré~~", 80);
        assert!(
            span(&lines, "gras")
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert!(
            span(&lines, "italique")
                .style
                .add_modifier
                .contains(Modifier::ITALIC)
        );
        assert_eq!(span(&lines, "code").style.fg, Some(Color::Yellow));
        assert!(
            span(&lines, "barré")
                .style
                .add_modifier
                .contains(Modifier::CROSSED_OUT)
        );
    }

    #[test]
    fn nested_styles_combine() {
        let lines = render("***les deux***", 80);
        let modifiers = span(&lines, "deux").style.add_modifier;
        assert!(modifiers.contains(Modifier::BOLD | Modifier::ITALIC));
    }

    #[test]
    fn headings_are_styled() {
        let lines = render("# Un\n\n### Trois", 80);
        let h1 = span(&lines, "Un").style;
        assert!(
            h1.add_modifier
                .contains(Modifier::BOLD | Modifier::UNDERLINED)
        );
        assert!(
            span(&lines, "Trois")
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
    }

    #[test]
    fn list_continuation_lines_are_indented() {
        let lines = render("- un deux trois quatre", 10);
        assert_eq!(text(&lines), "• un deux\n  trois\n  quatre");
    }

    #[test]
    fn ordered_list_honours_start_number() {
        assert_eq!(text(&render("3. a\n4. b", 80)), "3. a\n4. b");
    }

    #[test]
    fn code_block_is_highlighted_with_gutter() {
        let lines = render("```rust\nlet x = 1;\n```", 80);
        assert_eq!(text(&lines), "▎ rust\n▎ let x = 1;");
        assert!(matches!(span(&lines, "let").style.fg, Some(Color::Rgb(..))));
    }

    #[test]
    fn unterminated_code_block_while_streaming() {
        let lines = render("Voici :\n\n```python\nprint('a')\nfor", 80);
        assert_eq!(text(&lines), "Voici :\n\n▎ python\n▎ print('a')\n▎ for");
    }

    #[test]
    fn long_code_lines_are_hard_wrapped() {
        let lines = render("```\nabcdefghij\n```", 7);
        assert_eq!(text(&lines), "▎ abcde\n▎ fghij");
    }

    #[test]
    fn soft_breaks_become_spaces_and_hard_breaks_new_lines() {
        assert_eq!(text(&render("a\nb", 80)), "a b");
        assert_eq!(text(&render("a  \nb", 80)), "a\nb");
    }

    #[test]
    fn link_shows_destination_once() {
        assert_eq!(
            text(&render("[doc](https://x.y) <https://a.b>", 80)),
            "doc (https://x.y) https://a.b"
        );
    }

    #[test]
    fn closed_fence_detection() {
        assert!(is_closed_fence("```rust\nlet x = 1;\n```\n"));
        assert!(is_closed_fence("~~~\nx\n~~~~"));
        assert!(!is_closed_fence("```rust\nlet x = 1;\n"));
        assert!(!is_closed_fence("```"));
        assert!(!is_closed_fence("```\n``"));
        assert!(is_closed_fence("    indented code\n"));
    }

    #[test]
    fn empty_input() {
        assert!(render("", 80).is_empty());
    }

    #[test]
    fn code_blocks_are_extracted_and_numbered_when_several() {
        let markdown = "Voici :\n\n```rust\nfn a() {}\n```\n\npuis\n\n    indenté\n\n~~~\nb\n~~~\n";
        assert_eq!(
            code_blocks(markdown),
            vec!["fn a() {}\n", "indenté\n", "b\n"]
        );
        let text: Vec<String> = render(markdown, 40)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(text.contains(&"▎ rust · #1".to_owned()), "{text:?}");
        assert!(text.contains(&"▎ #2".to_owned()));
        let single: Vec<String> = render("```rust\nx\n```", 40)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(single[0], "▎ rust", "a single block is not numbered");
    }
}
