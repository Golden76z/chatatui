//! Text extraction.
//!
//! Word processor documents are converted to Markdown-like text (headings become `#`
//! lines) so that the chunker can follow their sections. PDFs are read page by page so
//! that passages can cite their page. Scanned PDFs (images only) have no text and are
//! reported as such: there is no OCR.

use std::{
    io::{Cursor, Read},
    path::Path,
};

use quick_xml::{Reader, events::Event};

/// Kind of file, from its extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    Markdown,
    Text,
    /// Source code or configuration.
    Code,
    Pdf,
    Docx,
    Odt,
}

const MARKDOWN: &[&str] = &["md", "markdown", "mdx"];
const TEXT: &[&str] = &[
    "txt", "text", "rst", "org", "adoc", "csv", "tsv", "log", "tex",
];
const CODE: &[&str] = &[
    "rs",
    "py",
    "js",
    "mjs",
    "cjs",
    "ts",
    "tsx",
    "jsx",
    "go",
    "java",
    "kt",
    "kts",
    "scala",
    "c",
    "h",
    "cc",
    "cpp",
    "cxx",
    "hpp",
    "cs",
    "rb",
    "php",
    "swift",
    "m",
    "lua",
    "r",
    "jl",
    "hs",
    "ml",
    "ex",
    "exs",
    "erl",
    "clj",
    "dart",
    "zig",
    "nim",
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "sql",
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "vue",
    "svelte",
    "xml",
    "toml",
    "yaml",
    "yml",
    "json",
    "ini",
    "cfg",
    "conf",
    "gradle",
    "cmake",
    "make",
    "mk",
    "dockerfile",
    "proto",
    "graphql",
];

impl FileKind {
    /// Kind of a file, or `None` for unsupported files.
    pub fn of(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_string_lossy().to_lowercase();
        if matches!(name.as_str(), "makefile" | "dockerfile" | "justfile") {
            return Some(Self::Code);
        }
        let extension = path.extension()?.to_string_lossy().to_lowercase();
        let extension = extension.as_str();
        if MARKDOWN.contains(&extension) {
            Some(Self::Markdown)
        } else if TEXT.contains(&extension) {
            Some(Self::Text)
        } else if CODE.contains(&extension) {
            Some(Self::Code)
        } else {
            match extension {
                "pdf" => Some(Self::Pdf),
                "docx" => Some(Self::Docx),
                "odt" => Some(Self::Odt),
                _ => None,
            }
        }
    }

    /// `true` when the text has Markdown headings to follow.
    pub fn has_sections(self) -> bool {
        matches!(self, Self::Markdown | Self::Docx | Self::Odt)
    }
}

/// Text of a document, split into pages when the format has pages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extracted {
    pub kind: FileKind,
    /// One entry per page for PDFs; a single entry otherwise.
    pub pages: Vec<String>,
}

impl Extracted {
    /// `true` when no text could be found.
    pub fn is_empty(&self) -> bool {
        self.pages.iter().all(|p| p.trim().is_empty())
    }
}

/// Error for a PDF without a text layer (usually a scan).
pub const NO_TEXT_PDF: &str = "PDF sans texte (scan ?)";

/// Extracts the text of `bytes` read from a file of `kind`. Errors are user-facing.
pub fn extract(kind: FileKind, bytes: &[u8]) -> Result<Extracted, String> {
    let pages = match kind {
        FileKind::Markdown | FileKind::Text | FileKind::Code => {
            if bytes.contains(&0) {
                return Err("fichier binaire".into());
            }
            vec![String::from_utf8_lossy(bytes).into_owned()]
        }
        FileKind::Pdf => pdf_pages(bytes)?,
        FileKind::Docx => vec![docx_text(bytes)?],
        FileKind::Odt => vec![odt_text(bytes)?],
    };
    let extracted = Extracted { kind, pages };
    if extracted.is_empty() {
        return Err(match kind {
            FileKind::Pdf => NO_TEXT_PDF.into(),
            _ => "aucun texte".into(),
        });
    }
    Ok(extracted)
}

/// PDF text, page by page. The PDF library may panic on unusual files: the panic is
/// contained and reported as an error.
fn pdf_pages(bytes: &[u8]) -> Result<Vec<String>, String> {
    let result = crate::terminal::quietly(|| pdf_extract::extract_text_from_mem_by_pages(bytes));
    match result {
        Some(Ok(pages)) => Ok(pages),
        Some(Err(error)) => Err(format!("PDF illisible ({error})")),
        None => Err("PDF illisible (format non pris en charge)".into()),
    }
}

fn read_zip_entry(bytes: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| format!("archive invalide ({e})"))?;
    let mut entry = archive
        .by_name(name)
        .map_err(|_| format!("{name} absent de l'archive"))?;
    let mut content = Vec::new();
    entry
        .read_to_end(&mut content)
        .map_err(|e| format!("archive illisible ({e})"))?;
    Ok(content)
}

/// Heading level of a Word paragraph style (`Heading2`, `Titre2`, `heading 2`…).
fn heading_level(style: &str) -> Option<usize> {
    let lower = style.to_lowercase();
    let rest = lower
        .strip_prefix("heading")
        .or_else(|| lower.strip_prefix("titre"))?;
    rest.trim()
        .parse::<usize>()
        .ok()
        .filter(|level| (1..=6).contains(level))
}

/// Text of `word/document.xml`, headings as Markdown.
fn docx_text(bytes: &[u8]) -> Result<String, String> {
    let xml = read_zip_entry(bytes, "word/document.xml")?;
    let xml = String::from_utf8_lossy(&xml);
    let mut reader = Reader::from_str(&xml);
    let mut out = String::new();
    let mut paragraph = String::new();
    let mut level: Option<usize> = None;
    let mut in_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.local_name().as_ref() {
                "t" => in_text = true,
                "p" => {
                    paragraph.clear();
                    level = None;
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                "pStyle" => {
                    level = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.local_name().as_ref() == "val")
                        .and_then(|a| heading_level(&a.value));
                }
                "tab" => paragraph.push('\t'),
                "br" | "cr" => paragraph.push('\n'),
                _ => {}
            },
            Ok(Event::Text(text)) if in_text => {
                paragraph.push_str(text.as_ref());
            }
            Ok(Event::GeneralRef(reference)) if in_text => {
                if let Ok(Some(c)) = reference.resolve_char_ref() {
                    paragraph.push(c);
                } else {
                    paragraph.push_str(match reference.as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => "",
                    });
                }
            }
            Ok(Event::End(e)) => match e.local_name().as_ref() {
                "t" => in_text = false,
                "p" => push_paragraph(&mut out, &paragraph, level),
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(error) => return Err(format!("document illisible ({error})")),
            Ok(_) => {}
        }
    }
    Ok(out)
}

/// Text of an OpenDocument `content.xml`, headings as Markdown.
fn odt_text(bytes: &[u8]) -> Result<String, String> {
    let xml = read_zip_entry(bytes, "content.xml")?;
    let xml = String::from_utf8_lossy(&xml);
    let mut reader = Reader::from_str(&xml);
    let mut out = String::new();
    let mut paragraph = String::new();
    // Heading level of the open text:h, `Some(0)` for a text:p.
    let mut open: Vec<usize> = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.local_name().as_ref() {
                "h" => {
                    let level = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.local_name().as_ref() == "outline-level")
                        .and_then(|a| a.value.parse().ok())
                        .unwrap_or(1);
                    paragraph.clear();
                    open.push(level);
                }
                "p" => {
                    if open.is_empty() {
                        paragraph.clear();
                    }
                    open.push(0);
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                "s" => paragraph.push(' '),
                "tab" => paragraph.push('\t'),
                "line-break" => paragraph.push('\n'),
                _ => {}
            },
            Ok(Event::Text(text)) if !open.is_empty() => {
                paragraph.push_str(text.as_ref());
            }
            Ok(Event::GeneralRef(reference)) if !open.is_empty() => {
                if let Ok(Some(c)) = reference.resolve_char_ref() {
                    paragraph.push(c);
                } else {
                    paragraph.push_str(match reference.as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => "",
                    });
                }
            }
            Ok(Event::End(e)) if matches!(e.local_name().as_ref(), "h" | "p") => {
                let level = open.pop().unwrap_or(0);
                if open.is_empty() {
                    push_paragraph(&mut out, &paragraph, (level > 0).then_some(level));
                    paragraph.clear();
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(format!("document illisible ({error})")),
            Ok(_) => {}
        }
    }
    Ok(out)
}

fn push_paragraph(out: &mut String, text: &str, heading: Option<usize>) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if let Some(level) = heading {
        out.push_str(&"#".repeat(level));
        out.push(' ');
    }
    out.push_str(text);
    out.push_str("\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        let path = format!("{}/tests/fixtures/rag/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(path).expect("fixture exists")
    }

    #[test]
    fn kinds_from_extensions() {
        assert_eq!(
            FileKind::of(Path::new("a/notes.MD")),
            Some(FileKind::Markdown)
        );
        assert_eq!(FileKind::of(Path::new("main.rs")), Some(FileKind::Code));
        assert_eq!(FileKind::of(Path::new("Makefile")), Some(FileKind::Code));
        assert_eq!(FileKind::of(Path::new("cours.pdf")), Some(FileKind::Pdf));
        assert_eq!(FileKind::of(Path::new("plan.odt")), Some(FileKind::Odt));
        assert_eq!(FileKind::of(Path::new("photo.jpg")), None);
        assert_eq!(FileKind::of(Path::new("README")), None);
    }

    #[test]
    fn pdf_is_read_page_by_page() {
        let extracted = extract(FileKind::Pdf, &fixture("cours.pdf")).expect("text");
        assert_eq!(extracted.pages.len(), 2);
        assert!(extracted.pages[0].contains("sécurité mémoire"));
        assert!(extracted.pages[1].contains("lifetimes"));
    }

    #[test]
    fn scanned_pdf_is_reported() {
        assert_eq!(
            extract(FileKind::Pdf, &fixture("scan.pdf")),
            Err("PDF sans texte (scan ?)".into())
        );
    }

    #[test]
    fn broken_pdf_is_an_error_not_a_crash() {
        assert!(extract(FileKind::Pdf, b"%PDF-1.4 garbage").is_err());
    }

    #[test]
    fn docx_headings_become_markdown() {
        let extracted = extract(FileKind::Docx, &fixture("plan.docx")).expect("text");
        let text = &extracted.pages[0];
        assert!(text.starts_with("# Plan du module\n\n"), "{text}");
        assert!(text.contains("## Séance 1 : ownership\n\n"));
        assert!(text.contains("Les traits décrivent un comportement partagé."));
    }

    #[test]
    fn odt_headings_become_markdown() {
        let extracted = extract(FileKind::Odt, &fixture("plan.odt")).expect("text");
        let text = &extracted.pages[0];
        assert!(text.contains("# Plan du module\n\n"), "{text}");
        assert!(text.contains("## Séance 2 : traits\n\n"));
        assert!(text.contains("Introduction au module de programmation système."));
    }

    #[test]
    fn text_files_and_binaries() {
        let extracted = extract(FileKind::Code, b"fn main() {}\n").expect("text");
        assert_eq!(extracted.pages, vec!["fn main() {}\n"]);
        assert_eq!(
            extract(FileKind::Text, b"a\0b"),
            Err("fichier binaire".into())
        );
        assert_eq!(
            extract(FileKind::Markdown, b"  \n"),
            Err("aucun texte".into())
        );
    }

    #[test]
    fn word_heading_styles() {
        assert_eq!(heading_level("Heading2"), Some(2));
        assert_eq!(heading_level("Titre1"), Some(1));
        assert_eq!(heading_level("heading 3"), Some(3));
        assert_eq!(heading_level("FirstParagraph"), None);
    }
}
