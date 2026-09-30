//! Screen layout, shared by the update logic (which needs sizes, e.g. for scrolling) and the
//! renderer, so both always agree.

use ratatui::layout::{Constraint, Layout, Rect};

/// Maximum number of input lines shown before the input box scrolls.
pub const MAX_INPUT_LINES: u16 = 8;

/// Maximum width of the conversation list.
pub const SIDEBAR_WIDTH: u16 = 32;

/// Areas of the main screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppLayout {
    /// Conversation list, when open.
    pub sidebar: Option<Rect>,
    pub chat: Rect,
    pub input: Rect,
    pub status: Rect,
}

/// Area where the conversation text is drawn: the chat area minus one column of padding
/// on each side.
pub fn chat_content(chat: Rect) -> Rect {
    Rect {
        x: chat.x.saturating_add(1),
        width: chat.width.saturating_sub(2),
        ..chat
    }
}

/// Splits the screen into optional sidebar, conversation, input box (grows with its
/// content) and status bar.
pub fn compute(area: Rect, input_lines: usize, sidebar_open: bool) -> AppLayout {
    let lines = u16::try_from(input_lines)
        .unwrap_or(MAX_INPUT_LINES)
        .clamp(1, MAX_INPUT_LINES);
    let [main, status] = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(area);
    let (sidebar, main) = if sidebar_open {
        let width = SIDEBAR_WIDTH.min(main.width / 3);
        let [sidebar, main] =
            Layout::horizontal([Constraint::Length(width), Constraint::Min(1)]).areas(main);
        (Some(sidebar), main)
    } else {
        (None, main)
    };
    let [chat, input] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(lines + 2), // + top and bottom borders
    ])
    .areas(main);
    AppLayout {
        sidebar,
        chat,
        input,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_grows_then_caps() {
        let area = Rect::new(0, 0, 80, 30);
        assert_eq!(compute(area, 0, false).input.height, 3);
        assert_eq!(compute(area, 3, false).input.height, 5);
        assert_eq!(compute(area, 100, false).input.height, MAX_INPUT_LINES + 2);
    }

    #[test]
    fn areas_cover_the_screen() {
        let area = Rect::new(0, 0, 80, 30);
        let layout = compute(area, 2, false);
        assert_eq!(
            layout.chat.height + layout.input.height + layout.status.height,
            30
        );
        assert_eq!(layout.status.y, 29);
        assert!(layout.sidebar.is_none());
    }

    #[test]
    fn sidebar_takes_the_left_side_above_the_status_bar() {
        let area = Rect::new(0, 0, 120, 30);
        let layout = compute(area, 1, true);
        let sidebar = layout.sidebar.expect("sidebar");
        assert_eq!(sidebar.width, SIDEBAR_WIDTH);
        assert_eq!(sidebar.height, 29);
        assert_eq!(layout.chat.x, SIDEBAR_WIDTH);
        assert_eq!(layout.input.width, 120 - SIDEBAR_WIDTH);
        assert_eq!(layout.status.width, 120);
    }

    #[test]
    fn sidebar_shrinks_on_narrow_screens() {
        let layout = compute(Rect::new(0, 0, 60, 20), 1, true);
        assert_eq!(layout.sidebar.map(|s| s.width), Some(20));
    }
}
