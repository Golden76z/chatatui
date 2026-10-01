//! The `/models` popup: what is downloaded, then what the catalogue offers.
//!
//! One list mixes the two so a person who has downloaded nothing still sees something to
//! choose from. A repository already on disk is never offered a second time.

use crate::models::{catalog, store::LocalModel};

/// One line of the popup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRow {
    /// On disk, with what its header said.
    Downloaded(LocalModel),
    /// Offered by [`catalog`], not downloaded.
    Available(catalog::Entry),
}

impl ModelRow {
    /// `owner/name` on HuggingFace.
    pub fn repo(&self) -> &str {
        match self {
            Self::Downloaded(model) => &model.repo,
            Self::Available(entry) => entry.repo,
        }
    }

    /// Everything the filter matches against, lowercased.
    fn haystack(&self) -> String {
        let mut text = match self {
            Self::Downloaded(model) => {
                let mut text = format!("{} {}", model.repo, model.file);
                for extra in [&model.quantization, &model.architecture]
                    .into_iter()
                    .flatten()
                {
                    text.push(' ');
                    text.push_str(extra);
                }
                text
            }
            Self::Available(entry) => format!("{} {} {}", entry.repo, entry.name, entry.note),
        };
        text.make_ascii_lowercase();
        text
    }
}

/// State of the open popup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelsPicker {
    pub rows: Vec<ModelRow>,
    /// Unix seconds when the inventory was read, for "il y a 3 h".
    pub now: i64,
    /// Case-insensitive words typed by the user; all must match.
    pub filter: String,
    /// Index into [`ModelsPicker::visible`].
    pub selected: usize,
    /// False until the inventory has been read once, so the popup can say so.
    pub loaded: bool,
    /// Answer to the last key press, shown in the popup's footer. The status bar shows
    /// overlay hints while a popup is open, so a reply to `Entrée` has to live in the popup.
    pub message: Option<String>,
}

impl ModelsPicker {
    /// A picker showing the catalogue, plus `inventory` when it has already been read.
    pub fn new(inventory: Option<(&[LocalModel], i64)>) -> Self {
        let mut picker = Self::default();
        if let Some((models, now)) = inventory {
            picker.set_models(models.to_vec(), now);
        } else {
            picker.rows = rows(&[]);
        }
        picker
    }

    /// Replaces the downloaded models, keeping the filter and a valid selection.
    pub fn set_models(&mut self, models: Vec<LocalModel>, now: i64) {
        self.rows = rows(&models);
        self.now = now;
        self.loaded = true;
        self.clamp_selection();
    }

    /// Rows matching the filter.
    pub fn visible(&self) -> Vec<&ModelRow> {
        let words: Vec<String> = self
            .filter
            .split_whitespace()
            .map(str::to_lowercase)
            .collect();
        self.rows
            .iter()
            .filter(|row| {
                let haystack = row.haystack();
                words.iter().all(|word| haystack.contains(word))
            })
            .collect()
    }

    /// The highlighted row, if the filter matches anything.
    pub fn selected(&self) -> Option<&ModelRow> {
        self.visible().get(self.selected).copied()
    }

    /// Moves the highlight by `delta`, clamped to the visible rows.
    pub fn move_selection(&mut self, delta: isize) {
        // The footer sits under a detail line about the highlighted row; an answer meant
        // for the row we are leaving would contradict it.
        self.message = None;
        let count = self.visible().len();
        if count == 0 {
            self.selected = 0;
            return;
        }
        let last = count - 1;
        self.selected = if delta < 0 {
            self.selected.saturating_sub(delta.unsigned_abs())
        } else {
            self.selected.saturating_add(delta.unsigned_abs()).min(last)
        };
    }

    /// Types a character into the filter; the highlight goes back to the first match.
    pub fn push_filter(&mut self, c: char) {
        self.filter.push(c);
        self.selected = 0;
        self.message = None;
    }

    /// Removes the last character of the filter; the highlight goes back to the first match.
    pub fn pop_filter(&mut self) {
        self.filter.pop();
        self.selected = 0;
        self.message = None;
    }

    fn clamp_selection(&mut self) {
        let count = self.visible().len();
        self.selected = match count {
            0 => 0,
            count => self.selected.min(count - 1),
        };
    }
}

/// Downloaded models first, in the order the store gave them, then the catalogue entries
/// that are not already on disk.
fn rows(models: &[LocalModel]) -> Vec<ModelRow> {
    let mut rows: Vec<ModelRow> = models.iter().cloned().map(ModelRow::Downloaded).collect();
    let downloaded: Vec<&str> = models.iter().map(|model| model.repo.as_str()).collect();
    rows.extend(
        catalog::entries()
            .iter()
            .filter(|entry| !downloaded.contains(&entry.repo))
            .copied()
            .map(ModelRow::Available),
    );
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(repo: &str, file: &str) -> LocalModel {
        LocalModel {
            repo: repo.to_owned(),
            revision: "main".to_owned(),
            file: file.to_owned(),
            path: format!("/tmp/{file}"),
            bytes: 400_000_000,
            sha256: Some("abc".to_owned()),
            architecture: Some("qwen3".to_owned()),
            quantization: Some("Q4_K_M".to_owned()),
            context_length: Some(40_960),
            parameters: Some(596_000_000),
            downloaded_at: 1_000,
        }
    }

    #[test]
    fn with_nothing_downloaded_the_whole_catalog_is_offered() {
        let picker = ModelsPicker::new(None);
        assert_eq!(picker.rows.len(), catalog::entries().len());
        assert!(
            picker
                .rows
                .iter()
                .all(|row| matches!(row, ModelRow::Available(_)))
        );
        assert!(!picker.loaded);
    }

    #[test]
    fn a_downloaded_model_comes_first_and_is_not_offered_twice() {
        let repo = catalog::entries()[0].repo;
        let mut picker = ModelsPicker::new(None);
        picker.set_models(vec![model(repo, "m-Q4_K_M.gguf")], 2_000);

        assert!(matches!(picker.rows.first(), Some(ModelRow::Downloaded(_))));
        assert_eq!(
            picker.rows.iter().filter(|row| row.repo() == repo).count(),
            1,
            "the downloaded repository must not also be offered"
        );
        assert_eq!(picker.rows.len(), catalog::entries().len());
        assert!(picker.loaded);
        assert_eq!(picker.now, 2_000);
    }

    #[test]
    fn a_downloaded_model_outside_the_catalog_is_still_listed() {
        let mut picker = ModelsPicker::new(None);
        picker.set_models(vec![model("someone/private-GGUF", "x.gguf")], 0);

        assert_eq!(picker.rows.len(), catalog::entries().len() + 1);
        assert_eq!(picker.rows[0].repo(), "someone/private-GGUF");
    }

    #[test]
    fn the_filter_matches_a_catalog_name_and_a_downloaded_file() {
        let mut picker = ModelsPicker::new(None);
        picker.set_models(vec![model("someone/private-GGUF", "rare-name.gguf")], 0);

        picker.filter = "rare-name".to_owned();
        assert_eq!(picker.visible().len(), 1);
        assert_eq!(picker.visible()[0].repo(), "someone/private-GGUF");

        picker.filter = "qwen3".to_owned();
        assert!(
            picker.visible().len() > 1,
            "several Qwen3 entries are offered"
        );

        picker.filter = "zzz nothing".to_owned();
        assert!(picker.visible().is_empty());
        assert!(picker.selected().is_none());
    }

    #[test]
    fn the_highlight_stays_inside_the_visible_rows() {
        let mut picker = ModelsPicker::new(None);
        picker.move_selection(-1);
        assert_eq!(picker.selected, 0);

        picker.move_selection(10_000);
        assert_eq!(picker.selected, picker.rows.len() - 1);
        assert!(picker.selected().is_some());

        // Filtering down to fewer rows than the current index must not leave it dangling.
        picker.push_filter('z');
        assert_eq!(picker.selected, 0);
    }

    /// A reply shown for one row must not survive onto another: the footer sits right under
    /// a detail line describing whatever is highlighted now.
    #[test]
    fn moving_the_highlight_clears_the_previous_answer() {
        let mut picker = ModelsPicker::new(None);
        picker.message = Some("about the first row".to_owned());

        picker.move_selection(1);

        assert_eq!(picker.message, None);
    }

    #[test]
    fn replacing_the_inventory_keeps_the_filter_and_clamps_the_highlight() {
        let mut picker = ModelsPicker::new(None);
        picker.filter = "qwen3".to_owned();
        picker.selected = picker.visible().len() - 1;
        let was = picker.selected;

        picker.set_models(vec![model("someone/private-GGUF", "x.gguf")], 0);

        assert_eq!(picker.filter, "qwen3");
        assert!(picker.selected <= was + 1);
        assert!(picker.selected < picker.visible().len());
    }
}
