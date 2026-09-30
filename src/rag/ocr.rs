//! Text recognition for scanned PDFs, with the `pdftoppm` (Poppler) and `tesseract`
//! programs when they are installed.
//!
//! Each page is rendered to a grayscale image, then read by Tesseract in the configured
//! languages (those it has data for). Slow: only used for PDFs without a text layer.

use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

/// Pages read at most per document.
const MAX_PAGES: usize = 300;
/// Rendering resolution (dots per inch): enough for body text.
const DPI: &str = "200";

/// Available OCR, with the languages to read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ocr {
    /// Tesseract `-l` value (`fra+eng`); empty for Tesseract's default.
    pub languages: String,
}

impl Ocr {
    /// OCR if both programs are installed, reading those of `wanted` (`fra+eng`) that
    /// Tesseract knows.
    pub fn detect(wanted: &str) -> Option<Self> {
        let runs = |program: &str, arg: &str| {
            Command::new(program)
                .arg(arg)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok()
        };
        if !runs("pdftoppm", "-v") || !runs("tesseract", "--version") {
            return None;
        }
        let installed = Command::new("tesseract")
            .arg("--list-langs")
            .stderr(Stdio::null())
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        Some(Self {
            languages: usable_languages(wanted, &installed),
        })
    }

    /// Text of each page of a PDF (blocking; may take seconds per page).
    pub fn pdf(&self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let dir = ScratchDir::new().map_err(|e| format!("OCR : dossier temporaire ({e})"))?;
        let input = dir.path.join("document.pdf");
        std::fs::write(&input, bytes).map_err(|e| format!("OCR : {e}"))?;
        let rendered = Command::new("pdftoppm")
            .args(["-r", DPI, "-gray", "-png", "-l", &MAX_PAGES.to_string()])
            .arg(&input)
            .arg(dir.path.join("page"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("OCR : pdftoppm ({e})"))?;
        if !rendered.success() {
            return Err("OCR : pdftoppm n'a pas pu lire le PDF".into());
        }
        let mut images: Vec<PathBuf> = std::fs::read_dir(&dir.path)
            .map_err(|e| format!("OCR : {e}"))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        // page-1.png … page-10.png (zero-padded by pdftoppm to the same width).
        images.sort();
        images
            .iter()
            .map(|image| self.image(image))
            .collect::<Result<Vec<_>, _>>()
    }

    fn image(&self, image: &Path) -> Result<String, String> {
        let mut command = Command::new("tesseract");
        command.arg(image).arg("-");
        if !self.languages.is_empty() {
            command.args(["-l", &self.languages]);
        }
        let output = command
            .stderr(Stdio::null())
            .output()
            .map_err(|e| format!("OCR : tesseract ({e})"))?;
        if !output.status.success() {
            return Err("OCR : tesseract a échoué".into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// The languages of `wanted` found in Tesseract's `--list-langs` output, joined by `+`.
fn usable_languages(wanted: &str, installed: &str) -> String {
    let installed: Vec<&str> = installed.lines().skip(1).map(str::trim).collect();
    wanted
        .split('+')
        .map(str::trim)
        .filter(|l| installed.contains(l))
        .collect::<Vec<_>>()
        .join("+")
}

/// A temporary directory removed when dropped.
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn new() -> std::io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "chatatui-ocr-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_installed_languages_are_asked_for() {
        let installed =
            "List of available languages in \"/usr/share/tessdata/\" (3):\neng\nosd\nfra\n";
        assert_eq!(usable_languages("fra+eng", installed), "fra+eng");
        assert_eq!(usable_languages("deu+eng", installed), "eng");
        assert_eq!(usable_languages("deu", installed), "");
    }

    #[test]
    fn reads_a_scanned_pdf_when_the_tools_are_installed() {
        let Some(ocr) = Ocr::detect("fra+eng") else {
            eprintln!("tesseract or pdftoppm missing: skipped");
            return;
        };
        let bytes = std::fs::read(format!(
            "{}/tests/fixtures/rag/scan-text.pdf",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture");
        let pages = ocr.pdf(&bytes).expect("ocr");
        assert_eq!(pages.len(), 2);
        assert!(pages[0].contains("lifetime"), "{pages:?}");
        assert!(pages[1].contains("RefCell"), "{pages:?}");
    }
}
