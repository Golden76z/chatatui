//! The conversation list panel: navigation, search, rename and delete.

use crate::storage::{ConversationId, ConversationSummary};

/// State of the open side panel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sidebar {
    /// `None` while the list is loading.
    pub items: Option<Vec<ConversationSummary>>,
    /// Excerpt of a matching message for each item, when searching.
    pub snippets: Vec<Option<String>>,
    /// Index of the highlighted item.
    pub selected: usize,
    /// When the list was produced (Unix seconds), for relative dates.
    pub listed_at: i64,
    /// Search typed in the panel (empty: every conversation).
    pub filter: String,
    /// New title being typed for the highlighted conversation (Ctrl+R).
    pub rename: Option<String>,
    /// Conversation to delete if `Suppr` is pressed again.
    pub confirm_delete: Option<ConversationId>,
}

impl Sidebar {
    /// Fills the list, highlighting `current` when it is in it.
    pub fn set_items(
        &mut self,
        items: Vec<ConversationSummary>,
        now: i64,
        current: Option<&ConversationId>,
    ) {
        let results = items.into_iter().map(|item| (item, None)).collect();
        self.set_results(results, now, current);
    }

    /// Fills the list with search results, highlighting `current` when it is in it.
    pub fn set_results(
        &mut self,
        results: Vec<(ConversationSummary, Option<String>)>,
        now: i64,
        current: Option<&ConversationId>,
    ) {
        let (items, snippets): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        self.selected = current
            .and_then(|id| items.iter().position(|item| &item.id == id))
            .unwrap_or(0);
        self.items = Some(items);
        self.snippets = snippets;
        self.listed_at = now;
        self.confirm_delete = None;
    }

    /// Excerpt of the item at `index`, if it was found by its messages.
    pub fn snippet(&self, index: usize) -> Option<&str> {
        self.snippets.get(index)?.as_deref()
    }

    /// Moves the highlight up, wrapping around.
    pub fn select_previous(&mut self) {
        self.confirm_delete = None;
        let len = self.len();
        if len > 0 {
            self.selected = (self.selected + len - 1) % len;
        }
    }

    /// Moves the highlight down, wrapping around.
    pub fn select_next(&mut self) {
        self.confirm_delete = None;
        let len = self.len();
        if len > 0 {
            self.selected = (self.selected + 1) % len;
        }
    }

    /// The highlighted conversation.
    pub fn selected_item(&self) -> Option<&ConversationSummary> {
        self.items.as_ref()?.get(self.selected)
    }

    /// Changes the title of a listed conversation.
    pub fn retitle(&mut self, id: &ConversationId, title: &str) {
        if let Some(item) = self.items.iter_mut().flatten().find(|item| &item.id == id) {
            item.title = title.to_owned();
        }
    }

    fn len(&self) -> usize {
        self.items.as_ref().map_or(0, Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(id: &str) -> ConversationSummary {
        ConversationSummary {
            id: ConversationId(id.into()),
            title: id.into(),
            provider: "ollama".into(),
            model: "m".into(),
            updated_at: 0,
        }
    }

    #[test]
    fn current_conversation_is_preselected() {
        let mut sidebar = Sidebar::default();
        let current = ConversationId("b".into());
        sidebar.set_items(vec![summary("a"), summary("b")], 0, Some(&current));
        assert_eq!(sidebar.selected, 1);
    }

    #[test]
    fn navigation_wraps() {
        let mut sidebar = Sidebar::default();
        sidebar.set_items(vec![summary("a"), summary("b"), summary("c")], 0, None);
        sidebar.select_previous();
        assert_eq!(sidebar.selected_item().map(|s| s.title.as_str()), Some("c"));
        sidebar.select_next();
        assert_eq!(sidebar.selected, 0);
    }

    #[test]
    fn empty_or_loading_list_has_no_selection() {
        let mut sidebar = Sidebar::default();
        sidebar.select_next();
        assert!(sidebar.selected_item().is_none());
        sidebar.set_items(Vec::new(), 0, None);
        sidebar.select_previous();
        assert!(sidebar.selected_item().is_none());
    }
}
