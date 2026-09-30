//! Slash-command suggestions shown above the input while typing `/…`.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Clear, List, ListItem, ListState},
};

use crate::{app::App, commands::CommandSpec, markdown::display_width};

const MAX_WIDTH: u16 = 72;
const MAX_ROWS: u16 = 8;

/// Draws the suggestion list just above `input`, over the bottom of `chat`.
pub fn render(app: &App, suggestions: &[&CommandSpec], frame: &mut Frame, chat: Rect, input: Rect) {
    let rows = u16::try_from(suggestions.len())
        .unwrap_or(MAX_ROWS)
        .min(MAX_ROWS);
    let height = (rows + 2).min(chat.height);
    let area = Rect {
        x: input.x,
        y: input.y.saturating_sub(height),
        width: input.width.min(MAX_WIDTH),
        height,
    };
    frame.render_widget(Clear, area);

    let usage_width = suggestions
        .iter()
        .map(|c| display_width(&c.usage()))
        .max()
        .unwrap_or(0);
    let dim = Style::default().fg(crate::theme::palette().dim);
    let items: Vec<ListItem> = suggestions
        .iter()
        .map(|spec| {
            let usage = spec.usage();
            let padding = " ".repeat(usage_width.saturating_sub(display_width(&usage)) + 2);
            let mut spans = vec![
                Span::styled(usage, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(padding),
                Span::raw(spec.description),
            ];
            if let Some(shortcut) = spec.shortcut(app.keyboard_enhanced) {
                spans.push(Span::styled(format!("  {shortcut}"), dim));
            }
            ListItem::new(Line::from(spans))
        })
        .collect();

    // Fit the widest row: borders + highlight symbol + content.
    let content_width = items.iter().map(ListItem::width).max().unwrap_or(0);
    let width = u16::try_from(content_width + 4)
        .unwrap_or(MAX_WIDTH)
        .min(MAX_WIDTH)
        .min(input.width);
    let rows = u16::try_from(suggestions.len())
        .unwrap_or(MAX_ROWS)
        .min(MAX_ROWS);
    let height = (rows + 2).min(chat.height);
    let area = Rect {
        x: input.x,
        y: input.y.saturating_sub(height),
        width,
        height,
    };
    frame.render_widget(Clear, area);

    let list = List::new(items)
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(crate::theme::palette().accent))
                .title(" Commandes "),
        )
        .highlight_symbol("▌")
        .highlight_style(Style::default().bg(crate::theme::palette().selection_bg));
    let selected = app.suggestion.min(suggestions.len().saturating_sub(1));
    // A throwaway state: the selection lives in `App`.
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}
