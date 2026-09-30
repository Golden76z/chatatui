//! The conversation list panel.

use crate::storage::{ConversationId, ConversationSummary};

/// State of the open side panel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sidebar {
    /// `None` while the list is loading.
    pub items: Option<Vec<ConversationSummary>>,
    /// Index of the highlighted item.
    pub selected: usize,
    /// When the list was produced (Unix seconds), for relative dates.
    pub listed_at: i64,
}

impl Sidebar {
    /// Fills the list, highlighting `current` when it is in it.
    pub fn set_items(
        &mut self,
        items: Vec<ConversationSummary>,
        now: i64,
        current: Option<&ConversationId>,
    ) {
        self.selected = current
            .and_then(|id| items.iter().position(|item| &item.id == id))
            .unwrap_or(0);
        self.items = Some(items);
        self.listed_at = now;
    }

    /// Moves the highlight up, wrapping around.
    pub fn select_previous(&mut self) {
        let len = self.len();
        if len > 0 {
            self.selected = (self.selected + len - 1) % len;
        }
    }

    /// Moves the highlight down, wrapping around.
    pub fn select_next(&mut self) {
        let len = self.len();
        if len > 0 {
            self.selected = (self.selected + 1) % len;
        }
    }

    /// The highlighted conversation.
    pub fn selected_item(&self) -> Option<&ConversationSummary> {
        self.items.as_ref()?.get(self.selected)
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
