//! The command palette (Ctrl+P).

use crate::commands::{self, CommandSpec};

/// State of the open palette.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Palette {
    /// Text typed to filter commands.
    pub filter: String,
    /// Index into [`Palette::visible`].
    pub selected: usize,
}

impl Palette {
    /// Commands matching the filter.
    pub fn visible(&self) -> Vec<&'static CommandSpec> {
        commands::search(&self.filter)
    }

    /// The highlighted command.
    pub fn selected_command(&self) -> Option<&'static CommandSpec> {
        self.visible().get(self.selected).copied()
    }

    /// Moves the highlight up, wrapping around.
    pub fn select_previous(&mut self) {
        let len = self.visible().len();
        if len > 0 {
            self.selected = (self.selected + len - 1) % len;
        }
    }

    /// Moves the highlight down, wrapping around.
    pub fn select_next(&mut self) {
        let len = self.visible().len();
        if len > 0 {
            self.selected = (self.selected + 1) % len;
        }
    }

    /// Appends to the filter and highlights the first match.
    pub fn push_filter(&mut self, c: char) {
        self.filter.push(c);
        self.selected = 0;
    }

    /// Removes the last filter character.
    pub fn pop_filter(&mut self) {
        self.filter.pop();
        self.selected = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::CommandId;

    #[test]
    fn filters_and_selects() {
        let mut palette = Palette::default();
        assert_eq!(
            palette.selected_command().map(|c| c.id),
            Some(CommandId::New)
        );
        for c in "mod".chars() {
            palette.push_filter(c);
        }
        assert_eq!(
            palette.selected_command().map(|c| c.id),
            Some(CommandId::Model)
        );
        palette.push_filter('z');
        assert!(palette.selected_command().is_none());
        palette.select_next();
        palette.pop_filter();
        assert_eq!(palette.selected, 0);
    }

    #[test]
    fn navigation_wraps() {
        let mut palette = Palette::default();
        palette.select_previous();
        assert_eq!(palette.selected, palette.visible().len() - 1);
        palette.select_next();
        assert_eq!(palette.selected, 0);
    }
}
