//! Maps raw key events to [`Action`]s.
//!
//! Terminals without the kitty keyboard protocol cannot distinguish `Shift+Enter` from
//! `Enter`, so `Alt+Enter` and `Ctrl+J` are accepted as alternatives for a new line.
//!
//! `Up` / `Down` edit the input when it has text and scroll the conversation when it is
//! empty.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::{action::Action, state::OverlayKind};

/// State the keymap depends on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyContext {
    /// The input box contains no text.
    pub input_empty: bool,
    /// The conversation list is open and has the focus.
    pub sidebar_open: bool,
    /// A popup is open and has the focus (takes precedence over everything else).
    pub overlay: Option<OverlayKind>,
    /// Slash-command suggestions are shown above the input.
    pub suggestions_open: bool,
    /// The input is `/add …`: Tab completes the path.
    pub completing_path: bool,
}

/// Translates a key press into an action. Returns `None` for keys that do nothing.
pub fn map_key(key: KeyEvent, context: KeyContext) -> Option<Action> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let plain = key.modifiers.is_empty();
    if matches!(key.code, KeyCode::Char('c' | 'q')) && ctrl {
        return Some(Action::Quit);
    }
    match context.overlay {
        Some(OverlayKind::List) => {
            return match key.code {
                KeyCode::Up => Some(Action::OverlayUp),
                KeyCode::Down => Some(Action::OverlayDown),
                KeyCode::Enter => Some(Action::OverlaySelect),
                KeyCode::Backspace => Some(Action::OverlayBackspace),
                // Esc and the shortcut that opened a popup both close it.
                KeyCode::Esc | KeyCode::F(1 | 2) => Some(Action::Cancel),
                KeyCode::Char('m' | 'p') if ctrl => Some(Action::Cancel),
                KeyCode::Char(c) if !ctrl && !alt => Some(Action::OverlayFilter(c)),
                _ => None,
            };
        }
        Some(OverlayKind::Text) => {
            return match key.code {
                KeyCode::Up => Some(Action::OverlayUp),
                KeyCode::Down => Some(Action::OverlayDown),
                KeyCode::PageUp => Some(Action::OverlayPageUp),
                KeyCode::PageDown => Some(Action::OverlayPageDown),
                KeyCode::Esc | KeyCode::Enter | KeyCode::F(1) | KeyCode::Char('q') => {
                    Some(Action::Cancel)
                }
                _ => None,
            };
        }
        None => {}
    }
    // Global shortcuts.
    match key.code {
        // Ctrl+M only reaches us with the kitty keyboard protocol (otherwise it is Enter).
        KeyCode::Char('m') if ctrl => return Some(Action::OpenModelPicker),
        KeyCode::F(2) => return Some(Action::OpenModelPicker),
        KeyCode::Char('p') if ctrl => return Some(Action::OpenPalette),
        KeyCode::F(1) => return Some(Action::OpenHelp),
        KeyCode::Char('n') if ctrl => return Some(Action::NewConversation),
        KeyCode::Char('l') if ctrl => return Some(Action::ToggleSidebar),
        KeyCode::Char('y') if ctrl => return Some(Action::CopyLastReply),
        _ => {}
    }
    if context.sidebar_open {
        return match key.code {
            KeyCode::Up => Some(Action::SidebarUp),
            KeyCode::Down => Some(Action::SidebarDown),
            KeyCode::Enter => Some(Action::SidebarOpen),
            KeyCode::Esc => Some(Action::Cancel),
            KeyCode::PageUp => Some(Action::PageUp),
            KeyCode::PageDown => Some(Action::PageDown),
            KeyCode::Backspace => Some(Action::SidebarBackspace),
            KeyCode::Delete => Some(Action::SidebarDelete),
            KeyCode::Char('r') if ctrl => Some(Action::SidebarRename),
            // Typing searches (or edits the new title): the input does not have the focus.
            KeyCode::Char(c) if !ctrl && !alt => Some(Action::SidebarType(c)),
            _ => None,
        };
    }
    if context.suggestions_open {
        match key.code {
            KeyCode::Up => return Some(Action::SuggestionUp),
            KeyCode::Down => return Some(Action::SuggestionDown),
            KeyCode::Tab => return Some(Action::CompleteSuggestion),
            KeyCode::Esc => return Some(Action::DismissSuggestions),
            _ => {}
        }
    }
    if context.completing_path && key.code == KeyCode::Tab {
        return Some(Action::CompletePath);
    }
    match key.code {
        KeyCode::PageUp => Some(Action::PageUp),
        KeyCode::PageDown => Some(Action::PageDown),
        KeyCode::Home if ctrl => Some(Action::ScrollToTop),
        KeyCode::End if ctrl => Some(Action::ScrollToBottom),
        KeyCode::Up if plain && context.input_empty => Some(Action::ScrollUp(1)),
        KeyCode::Down if plain && context.input_empty => Some(Action::ScrollDown(1)),
        // Raw-mode terminals report a bare line feed as Ctrl+J; some map Shift+Enter to it.
        KeyCode::Char('j') if ctrl => Some(Action::InsertNewline),
        KeyCode::Enter if key.modifiers.is_empty() => Some(Action::Submit),
        KeyCode::Enter => Some(Action::InsertNewline),
        KeyCode::Esc => Some(Action::Cancel),
        _ => Some(Action::Edit(key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn map_key(key: KeyEvent) -> Option<Action> {
        super::map_key(key, KeyContext::default())
    }

    #[test]
    fn arrows_scroll_only_when_input_is_empty() {
        let up = key(KeyCode::Up, KeyModifiers::NONE);
        let empty = KeyContext {
            input_empty: true,
            ..KeyContext::default()
        };
        assert_eq!(super::map_key(up, empty), Some(Action::ScrollUp(1)));
        assert_eq!(map_key(up), Some(Action::Edit(up)));
    }

    #[test]
    fn conversation_shortcuts() {
        assert_eq!(
            map_key(key(KeyCode::Char('n'), KeyModifiers::CONTROL)),
            Some(Action::NewConversation)
        );
        assert_eq!(
            map_key(key(KeyCode::Char('l'), KeyModifiers::CONTROL)),
            Some(Action::ToggleSidebar)
        );
    }

    #[test]
    fn sidebar_takes_the_focus() {
        let context = KeyContext {
            input_empty: false,
            sidebar_open: true,
            ..KeyContext::default()
        };
        let map = |code| super::map_key(key(code, KeyModifiers::NONE), context);
        assert_eq!(map(KeyCode::Up), Some(Action::SidebarUp));
        assert_eq!(map(KeyCode::Down), Some(Action::SidebarDown));
        assert_eq!(map(KeyCode::Enter), Some(Action::SidebarOpen));
        assert_eq!(map(KeyCode::Esc), Some(Action::Cancel));
        assert_eq!(
            map(KeyCode::Char('x')),
            Some(Action::SidebarType('x')),
            "typing searches"
        );
        assert_eq!(map(KeyCode::Backspace), Some(Action::SidebarBackspace));
        assert_eq!(map(KeyCode::Delete), Some(Action::SidebarDelete));
        assert_eq!(
            super::map_key(key(KeyCode::Char('r'), KeyModifiers::CONTROL), context),
            Some(Action::SidebarRename)
        );
        assert_eq!(
            super::map_key(key(KeyCode::Char('l'), KeyModifiers::CONTROL), context),
            Some(Action::ToggleSidebar)
        );
    }

    #[test]
    fn model_picker_shortcuts() {
        assert_eq!(
            map_key(key(KeyCode::F(2), KeyModifiers::NONE)),
            Some(Action::OpenModelPicker)
        );
        assert_eq!(
            map_key(key(KeyCode::Char('m'), KeyModifiers::CONTROL)),
            Some(Action::OpenModelPicker)
        );
    }

    #[test]
    fn list_popup_takes_the_focus_over_the_sidebar() {
        let context = KeyContext {
            input_empty: true,
            sidebar_open: true,
            overlay: Some(OverlayKind::List),
            ..KeyContext::default()
        };
        let map = |code, modifiers| super::map_key(key(code, modifiers), context);
        assert_eq!(
            map(KeyCode::Up, KeyModifiers::NONE),
            Some(Action::OverlayUp)
        );
        assert_eq!(
            map(KeyCode::Enter, KeyModifiers::NONE),
            Some(Action::OverlaySelect)
        );
        assert_eq!(
            map(KeyCode::Char('Q'), KeyModifiers::SHIFT),
            Some(Action::OverlayFilter('Q'))
        );
        assert_eq!(
            map(KeyCode::Backspace, KeyModifiers::NONE),
            Some(Action::OverlayBackspace)
        );
        assert_eq!(map(KeyCode::Esc, KeyModifiers::NONE), Some(Action::Cancel));
        assert_eq!(
            map(KeyCode::Char('p'), KeyModifiers::CONTROL),
            Some(Action::Cancel)
        );
        assert_eq!(map(KeyCode::Char('l'), KeyModifiers::CONTROL), None);
        assert_eq!(
            map(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(Action::Quit)
        );
    }

    #[test]
    fn text_popup_scrolls_and_closes() {
        let context = KeyContext {
            overlay: Some(OverlayKind::Text),
            ..KeyContext::default()
        };
        let map = |code| super::map_key(key(code, KeyModifiers::NONE), context);
        assert_eq!(map(KeyCode::Down), Some(Action::OverlayDown));
        assert_eq!(map(KeyCode::Char('q')), Some(Action::Cancel));
        assert_eq!(map(KeyCode::Char('x')), None);
    }

    #[test]
    fn palette_and_help_shortcuts() {
        assert_eq!(
            map_key(key(KeyCode::Char('p'), KeyModifiers::CONTROL)),
            Some(Action::OpenPalette)
        );
        assert_eq!(
            map_key(key(KeyCode::F(1), KeyModifiers::NONE)),
            Some(Action::OpenHelp)
        );
    }

    #[test]
    fn tab_completes_paths_after_add() {
        let context = KeyContext {
            completing_path: true,
            ..KeyContext::default()
        };
        assert_eq!(
            super::map_key(key(KeyCode::Tab, KeyModifiers::NONE), context),
            Some(Action::CompletePath)
        );
    }

    #[test]
    fn suggestions_capture_navigation_keys() {
        let context = KeyContext {
            suggestions_open: true,
            ..KeyContext::default()
        };
        let map = |code| super::map_key(key(code, KeyModifiers::NONE), context);
        assert_eq!(map(KeyCode::Up), Some(Action::SuggestionUp));
        assert_eq!(map(KeyCode::Tab), Some(Action::CompleteSuggestion));
        assert_eq!(map(KeyCode::Esc), Some(Action::DismissSuggestions));
        assert_eq!(map(KeyCode::Enter), Some(Action::Submit));
        let x = key(KeyCode::Char('x'), KeyModifiers::NONE);
        assert_eq!(super::map_key(x, context), Some(Action::Edit(x)));
    }

    #[test]
    fn page_keys_scroll() {
        assert_eq!(
            map_key(key(KeyCode::PageUp, KeyModifiers::NONE)),
            Some(Action::PageUp)
        );
        assert_eq!(
            map_key(key(KeyCode::End, KeyModifiers::CONTROL)),
            Some(Action::ScrollToBottom)
        );
        let end = key(KeyCode::End, KeyModifiers::NONE);
        assert_eq!(map_key(end), Some(Action::Edit(end)), "End alone edits");
    }

    #[test]
    fn enter_submits() {
        assert_eq!(
            map_key(key(KeyCode::Enter, KeyModifiers::NONE)),
            Some(Action::Submit)
        );
    }

    #[test]
    fn modified_enter_inserts_newline() {
        for modifiers in [KeyModifiers::SHIFT, KeyModifiers::ALT] {
            assert_eq!(
                map_key(key(KeyCode::Enter, modifiers)),
                Some(Action::InsertNewline)
            );
        }
        assert_eq!(
            map_key(key(KeyCode::Char('j'), KeyModifiers::CONTROL)),
            Some(Action::InsertNewline)
        );
    }

    #[test]
    fn ctrl_c_and_ctrl_q_quit() {
        for c in ['c', 'q'] {
            assert_eq!(
                map_key(key(KeyCode::Char(c), KeyModifiers::CONTROL)),
                Some(Action::Quit)
            );
        }
    }

    #[test]
    fn plain_characters_are_edits() {
        let k = key(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(map_key(k), Some(Action::Edit(k)));
    }

    #[test]
    fn escape_cancels() {
        assert_eq!(
            map_key(key(KeyCode::Esc, KeyModifiers::NONE)),
            Some(Action::Cancel)
        );
    }
}
