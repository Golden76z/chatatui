//! Searching the open conversation (Ctrl+F): the query and where it matches.
//!
//! Matches are positions in the transcript's display lines, so the view can scroll to
//! them and highlight them exactly. Case and accents are ignored, like the list search.

/// A match: display line and character range within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FindMatch {
    pub line: usize,
    /// First character (index in the line's text).
    pub start: usize,
    /// One past the last character.
    pub end: usize,
}

/// State of the find bar.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Find {
    pub query: String,
    pub matches: Vec<FindMatch>,
    /// Index of the current match in `matches`.
    pub current: usize,
    /// Transcript revision the matches were computed for.
    pub revision: Option<u64>,
}

impl Find {
    /// The current match, if any.
    pub fn current_match(&self) -> Option<FindMatch> {
        self.matches.get(self.current).copied()
    }

    /// Recomputes the matches over `lines`, keeping the current one near `keep_line`
    /// (the first match at or after it).
    pub fn search<'a>(&mut self, lines: impl Iterator<Item = &'a str>, keep_line: usize) {
        let query: Vec<char> = crate::export::folded(self.query.trim()).chars().collect();
        self.matches = if query.is_empty() {
            Vec::new()
        } else {
            lines
                .enumerate()
                .flat_map(|(line, text)| find_in_line(text, &query, line))
                .collect()
        };
        self.current = self
            .matches
            .iter()
            .position(|m| m.line >= keep_line)
            .unwrap_or(0);
    }

    /// Searches again after the lines changed (new text, other width): the current match
    /// keeps its rank when the number of matches is the same.
    pub fn refresh<'a>(&mut self, lines: impl Iterator<Item = &'a str>) {
        let (count, current) = (self.matches.len(), self.current);
        let keep = self.current_match().map_or(0, |m| m.line);
        self.search(lines, keep);
        if self.matches.len() == count {
            self.current = current;
        }
    }

    /// Moves to the next match, wrapping around.
    pub fn next(&mut self) {
        if !self.matches.is_empty() {
            self.current = (self.current + 1) % self.matches.len();
        }
    }

    /// Moves to the previous match, wrapping around.
    pub fn previous(&mut self) {
        let len = self.matches.len();
        if len > 0 {
            self.current = (self.current + len - 1) % len;
        }
    }
}

/// Occurrences of the folded `query` in `text` (non-overlapping).
fn find_in_line(text: &str, query: &[char], line: usize) -> Vec<FindMatch> {
    // Folded characters, each with the index of the character it comes from.
    let folded: Vec<(char, usize)> = text
        .chars()
        .enumerate()
        .flat_map(|(i, c)| crate::export::fold_char(c).map(move |f| (f, i)))
        .collect();
    let mut matches = Vec::new();
    let mut at = 0;
    while at + query.len() <= folded.len() {
        let window = &folded[at..at + query.len()];
        if window.iter().map(|(c, _)| *c).eq(query.iter().copied()) {
            let start = window[0].1;
            let end = window[window.len() - 1].1 + 1;
            // A match ending inside a folded ligature still covers the whole character.
            matches.push(FindMatch { line, start, end });
            at += query.len();
        } else {
            at += 1;
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(query: &str, lines: &[&str], keep: usize) -> Find {
        let mut find = Find {
            query: query.into(),
            ..Find::default()
        };
        find.search(lines.iter().copied(), keep);
        find
    }

    #[test]
    fn matches_ignore_case_and_accents() {
        let f = find("creme", &["La Crème brûlée", "rien", "crème, CREME"], 0);
        assert_eq!(
            f.matches,
            vec![
                FindMatch {
                    line: 0,
                    start: 3,
                    end: 8
                },
                FindMatch {
                    line: 2,
                    start: 0,
                    end: 5
                },
                FindMatch {
                    line: 2,
                    start: 7,
                    end: 12
                },
            ]
        );
        let f = find("œu", &["un cœur"], 0);
        assert_eq!(
            f.matches,
            vec![FindMatch {
                line: 0,
                start: 4,
                end: 6
            }]
        );
        assert!(find("  ", &["x"], 0).matches.is_empty());
    }

    #[test]
    fn navigation_starts_near_the_view_and_wraps() {
        let lines = ["a x", "b", "x", "c", "x"];
        let mut f = find("x", &lines, 3);
        assert_eq!(
            f.current_match().map(|m| m.line),
            Some(4),
            "first from line 3"
        );
        f.next();
        assert_eq!(f.current_match().map(|m| m.line), Some(0));
        f.previous();
        f.previous();
        assert_eq!(f.current_match().map(|m| m.line), Some(2));
        let mut none = find("zz", &lines, 0);
        none.next();
        assert_eq!(none.current_match(), None);
    }
}
