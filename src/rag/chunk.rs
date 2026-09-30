//! Splitting documents into passages.
//!
//! Text is cut into blocks (paragraphs, code blocks: runs of lines separated by blank
//! lines), then blocks are packed greedily up to about `chunk_tokens` tokens. A small
//! last block is repeated at the start of the next passage, so that an idea cut in two
//! stays findable. Each passage records where it comes from:
//!
//! - documents with headings (Markdown, .docx, .odt): `§ Heading`; a new level-1 or
//!   level-2 heading always starts a new passage;
//! - PDFs: `p. 12` (passages never span two pages);
//! - code and plain text: `L40-88`.

use super::extract::{Extracted, FileKind};

/// A passage of a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Passage {
    /// Position in the document, from 0.
    pub ordinal: usize,
    /// Where it comes from, for citations (`§ …`, `p. 3`, `L10-42`).
    pub location: String,
    pub text: String,
}

impl Passage {
    /// Text given to the embedding model: the passage with its origin, which helps
    /// retrieval ("what does the Rust course say about …").
    pub fn embedding_text(&self, file: &str) -> String {
        format!("{file} — {}\n{}", self.location, self.text)
    }
}

/// A block and its position.
#[derive(Clone, Debug)]
struct Block {
    text: String,
    first_line: usize,
    last_line: usize,
    /// Heading in effect (for documents with sections).
    heading: Option<String>,
    /// `true` when the block is a level-1 or level-2 heading.
    major_heading: bool,
}

/// Splits a document into passages of about `chunk_tokens` tokens.
pub fn split(extracted: &Extracted, chunk_tokens: usize) -> Vec<Passage> {
    let max_chars = chunk_tokens.max(50) * 4;
    let overlap_chars = max_chars / 8;
    let mut passages = Vec::new();
    for (page_index, page) in extracted.pages.iter().enumerate() {
        let blocks = blocks(page, extracted.kind.has_sections(), max_chars);
        for (text, first, last) in pack(&blocks, max_chars, overlap_chars) {
            let location = match extracted.kind {
                FileKind::Pdf => format!("p. {}", page_index + 1),
                FileKind::Markdown | FileKind::Docx | FileKind::Odt => first
                    .heading
                    .as_ref()
                    .map_or_else(|| "début".to_owned(), |h| format!("§ {h}")),
                FileKind::Code | FileKind::Text => {
                    format!("L{}-{}", first.first_line, last.last_line)
                }
            };
            passages.push(Passage {
                ordinal: passages.len(),
                location,
                text,
            });
        }
    }
    passages
}

/// Cuts text into blocks separated by blank lines; oversized blocks are cut further.
fn blocks(text: &str, sections: bool, max_chars: usize) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut heading: Option<String> = None;
    let mut current: Vec<&str> = Vec::new();
    let mut first_line = 1;

    #[allow(clippy::too_many_arguments)]
    fn flush(
        blocks: &mut Vec<Block>,
        current: &mut Vec<&str>,
        first: usize,
        last: usize,
        heading: &Option<String>,
        max_chars: usize,
    ) {
        if current.iter().all(|l| l.trim().is_empty()) {
            current.clear();
            return;
        }
        let text = current.join("\n");
        current.clear();
        for (piece, (a, b)) in cut(&text, first, max_chars) {
            blocks.push(Block {
                text: piece,
                first_line: a,
                last_line: b.min(last),
                heading: heading.clone(),
                major_heading: false,
            });
        }
    }

    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        let title = if sections {
            markdown_heading(line)
        } else {
            None
        };
        if let Some((level, title)) = title {
            flush(
                &mut blocks,
                &mut current,
                first_line,
                number.saturating_sub(1),
                &heading,
                max_chars,
            );
            heading = Some(title.to_owned());
            blocks.push(Block {
                text: line.trim().to_owned(),
                first_line: number,
                last_line: number,
                heading: heading.clone(),
                major_heading: level <= 2,
            });
            first_line = number + 1;
        } else if line.trim().is_empty() {
            flush(
                &mut blocks,
                &mut current,
                first_line,
                number.saturating_sub(1),
                &heading,
                max_chars,
            );
            first_line = number + 1;
        } else {
            if current.is_empty() {
                first_line = number;
            }
            current.push(line);
        }
    }
    let last = text.lines().count();
    flush(
        &mut blocks,
        &mut current,
        first_line,
        last,
        &heading,
        max_chars,
    );
    blocks
}

/// `# Title` → `(1, "Title")`.
fn markdown_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let title = trimmed[level..].strip_prefix(' ')?.trim();
    (!title.is_empty()).then_some((level, title))
}

/// Cuts an oversized block at line boundaries (or inside a line when a single line is too
/// long). Returns the pieces with their line ranges.
fn cut(text: &str, first_line: usize, max_chars: usize) -> Vec<(String, (usize, usize))> {
    if text.chars().count() <= max_chars {
        let lines = text.lines().count().max(1);
        return vec![(text.to_owned(), (first_line, first_line + lines - 1))];
    }
    let mut pieces = Vec::new();
    let mut piece = String::new();
    let mut piece_first = first_line;
    for (index, line) in text.lines().enumerate() {
        let number = first_line + index;
        let mut line = line.to_owned();
        while line.chars().count() > max_chars {
            let head: String = line.chars().take(max_chars).collect();
            let tail: String = line.chars().skip(max_chars).collect();
            if !piece.is_empty() {
                pieces.push((std::mem::take(&mut piece), (piece_first, number)));
            }
            pieces.push((head, (number, number)));
            piece_first = number;
            line = tail;
        }
        if !piece.is_empty() && piece.chars().count() + line.chars().count() + 1 > max_chars {
            pieces.push((std::mem::take(&mut piece), (piece_first, number - 1)));
            piece_first = number;
        }
        if piece.is_empty() {
            piece_first = number;
        } else {
            piece.push('\n');
        }
        piece.push_str(&line);
    }
    if !piece.trim().is_empty() {
        let last = first_line + text.lines().count().saturating_sub(1);
        pieces.push((piece, (piece_first, last)));
    }
    pieces
}

/// Packs blocks into passages. Returns each passage with its first and last block.
fn pack<'a>(
    blocks: &'a [Block],
    max_chars: usize,
    overlap_chars: usize,
) -> Vec<(String, &'a Block, &'a Block)> {
    let mut passages = Vec::new();
    let mut current: Vec<&Block> = Vec::new();
    let mut size = 0;
    // Number of leading blocks of `current` repeated from the previous passage.
    let mut carried = 0;

    let emit = |current: &[&'a Block], passages: &mut Vec<(String, &'a Block, &'a Block)>| {
        if let (Some(first), Some(last)) = (current.first(), current.last()) {
            let text = current
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            passages.push((text, *first, *last));
        }
    };

    for block in blocks {
        let len = block.text.chars().count();
        let starts_section = block.major_heading && current.len() > carried;
        let too_big = size + len + 2 > max_chars && current.len() > carried;
        if starts_section || too_big {
            emit(&current, &mut passages);
            // Repeat a short last block, unless we are starting a new section.
            let keep: Vec<&Block> = match current.last() {
                Some(last)
                    if !starts_section
                        && last.text.chars().count() <= overlap_chars
                        && !last.major_heading =>
                {
                    vec![*last]
                }
                _ => Vec::new(),
            };
            carried = keep.len();
            size = keep.iter().map(|b| b.text.chars().count() + 2).sum();
            current = keep;
        }
        current.push(block);
        size += len + 2;
    }
    if current.len() > carried {
        emit(&current, &mut passages);
    }
    passages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(kind: FileKind, text: &str) -> Extracted {
        Extracted {
            kind,
            pages: vec![text.to_owned()],
        }
    }

    #[test]
    fn small_document_is_one_passage() {
        let passages = split(&doc(FileKind::Markdown, "# Titre\n\nUn paragraphe.\n"), 800);
        assert_eq!(passages.len(), 1);
        assert_eq!(passages[0].location, "§ Titre");
        assert_eq!(passages[0].text, "# Titre\n\nUn paragraphe.");
    }

    #[test]
    fn major_headings_start_new_passages() {
        let text = "# Plan\n\nIntro.\n\n## Séance 1\n\nOwnership.\n\n### Détail\n\nEmprunts.\n\n## Séance 2\n\nTraits.\n";
        let passages = split(&doc(FileKind::Markdown, text), 800);
        let locations: Vec<&str> = passages.iter().map(|p| p.location.as_str()).collect();
        assert_eq!(locations, ["§ Plan", "§ Séance 1", "§ Séance 2"]);
        assert!(
            passages[1].text.contains("### Détail\n\nEmprunts."),
            "minor heading kept inside"
        );
    }

    #[test]
    fn long_text_is_packed_under_the_limit_with_overlap() {
        let paragraph = "mot ".repeat(40); // 160 characters
        let text: String = (0..30).map(|i| format!("{i} {paragraph}\n\n")).collect();
        let passages = split(&doc(FileKind::Text, &text), 250); // ~1000 characters
        assert!(passages.len() > 3);
        for passage in &passages {
            assert!(
                passage.text.chars().count() <= 1_000,
                "{}",
                passage.text.len()
            );
        }
        // Every paragraph is somewhere.
        for i in 0..30 {
            assert!(
                passages
                    .iter()
                    .any(|p| p.text.contains(&format!("{i} mot"))),
                "{i}"
            );
        }
        assert_eq!(passages[0].location, "L1-11");
        assert!(passages.iter().enumerate().all(|(i, p)| p.ordinal == i));
    }

    #[test]
    fn short_last_paragraph_is_repeated_in_the_next_passage() {
        let long = "x".repeat(900);
        let text = format!("{long}\n\ncourt\n\n{long}\n");
        let passages = split(&doc(FileKind::Text, &text), 250);
        assert_eq!(passages.len(), 2);
        assert!(passages[0].text.ends_with("court"));
        assert!(passages[1].text.starts_with("court"));
    }

    #[test]
    fn code_locations_are_line_ranges() {
        let code = "fn a() {\n    1\n}\n\nfn b() {\n    2\n}\n";
        let passages = split(&doc(FileKind::Code, code), 800);
        assert_eq!(passages.len(), 1);
        assert_eq!(passages[0].location, "L1-7");
    }

    #[test]
    fn oversized_blocks_and_lines_are_cut() {
        let line = "y".repeat(3_000);
        let passages = split(&doc(FileKind::Text, &line), 250);
        assert_eq!(passages.len(), 3);
        assert!(passages.iter().all(|p| p.text.chars().count() <= 1_000));
    }

    #[test]
    fn pdf_passages_never_span_pages() {
        let extracted = Extracted {
            kind: FileKind::Pdf,
            pages: vec!["Page un.".into(), "Page deux.".into()],
        };
        let passages = split(&extracted, 800);
        let locations: Vec<&str> = passages.iter().map(|p| p.location.as_str()).collect();
        assert_eq!(locations, ["p. 1", "p. 2"]);
    }

    #[test]
    fn embedding_text_names_the_origin() {
        let passage = Passage {
            ordinal: 0,
            location: "p. 3".into(),
            text: "contenu".into(),
        };
        assert_eq!(
            passage.embedding_text("cours.pdf"),
            "cours.pdf — p. 3\ncontenu"
        );
    }
}
