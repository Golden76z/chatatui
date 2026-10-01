//! The popup that picks which GGUF file of a repository to download.

use crate::models::hub::RemoteFile;

/// State of the open popup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GgufPicker {
    /// Repository the files come from.
    pub repo: String,
    pub files: Vec<RemoteFile>,
    /// Case-insensitive words typed by the user; all must match.
    pub filter: String,
    /// Index into [`GgufPicker::visible`].
    pub selected: usize,
}

impl GgufPicker {
    /// A picker over a repository's files, smallest first so the usual choice is on top.
    pub fn new(repo: String, mut files: Vec<RemoteFile>) -> Self {
        files.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| a.path.cmp(&b.path)));
        Self {
            repo,
            files,
            filter: String::new(),
            selected: 0,
        }
    }

    /// Files matching the filter.
    pub fn visible(&self) -> Vec<&RemoteFile> {
        let words: Vec<String> = self
            .filter
            .split_whitespace()
            .map(str::to_lowercase)
            .collect();
        self.files
            .iter()
            .filter(|file| {
                let haystack = file.path.to_lowercase();
                words.iter().all(|word| haystack.contains(word))
            })
            .collect()
    }

    /// The highlighted file, if the filter matches anything.
    pub fn selected(&self) -> Option<&RemoteFile> {
        self.visible().get(self.selected).copied()
    }

    /// Moves the highlight by `delta`, clamped to the visible files.
    pub fn move_selection(&mut self, delta: isize) {
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
    }

    /// Deletes the filter's last character.
    pub fn pop_filter(&mut self) {
        self.filter.pop();
        self.selected = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, bytes: u64) -> RemoteFile {
        RemoteFile {
            path: path.to_owned(),
            bytes,
            sha256: None,
        }
    }

    #[test]
    fn lists_the_smallest_file_first() {
        let picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("b-Q8_0.gguf", 7_000), file("a-Q4_K_M.gguf", 4_000)],
        );

        assert_eq!(
            picker.selected().map(|f| f.path.as_str()),
            Some("a-Q4_K_M.gguf")
        );
    }

    #[test]
    fn the_filter_keeps_only_matching_files() {
        let mut picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("a-Q4_K_M.gguf", 4_000), file("b-Q8_0.gguf", 7_000)],
        );
        picker.filter = "q8".to_owned();

        assert_eq!(picker.visible().len(), 1);
        assert_eq!(
            picker.selected().map(|f| f.path.as_str()),
            Some("b-Q8_0.gguf")
        );
    }

    #[test]
    fn a_filter_matching_nothing_selects_nothing() {
        let mut picker = GgufPicker::new("owner/name".to_owned(), vec![file("a.gguf", 4_000)]);
        picker.filter = "zzz".to_owned();

        assert_eq!(picker.selected(), None);
    }

    #[test]
    fn the_highlight_stays_inside_the_list() {
        let mut picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("a.gguf", 1), file("b.gguf", 2)],
        );

        picker.move_selection(10);
        assert_eq!(picker.selected().map(|f| f.path.as_str()), Some("b.gguf"));
        picker.move_selection(-10);
        assert_eq!(picker.selected().map(|f| f.path.as_str()), Some("a.gguf"));
    }

    #[test]
    fn typing_in_the_filter_goes_back_to_the_first_match() {
        let mut picker = GgufPicker::new(
            "owner/name".to_owned(),
            vec![file("a-Q4.gguf", 1), file("b-Q8.gguf", 2)],
        );
        picker.move_selection(1);

        picker.push_filter('q');
        assert_eq!(picker.selected, 0);
        picker.pop_filter();
        assert!(picker.filter.is_empty());
    }
}
