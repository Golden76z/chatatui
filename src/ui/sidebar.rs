//! Conversation list panel.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, List, ListItem, ListState, Paragraph},
};

use crate::{app::App, markdown::display_width, state::Sidebar};

/// Draws the panel.
pub fn render(app: &App, sidebar: &Sidebar, frame: &mut Frame, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Conversations ")
        .title_bottom(Line::from(" Ctrl+N nouvelle ").right_aligned());

    let Some(items) = &sidebar.items else {
        let loading = Paragraph::new("Chargement…").style(dim()).block(block);
        frame.render_widget(loading, area);
        return;
    };
    if items.is_empty() {
        let empty = Paragraph::new("Aucune conversation enregistrée.")
            .style(dim())
            .wrap(ratatui::widgets::Wrap { trim: true })
            .block(block);
        frame.render_widget(empty, area);
        return;
    }

    let width = usize::from(area.width.saturating_sub(4)); // borders + highlight symbol
    let list_items: Vec<ListItem> = items
        .iter()
        .map(|item| {
            let current = Some(&item.id) == app.conversation_id.as_ref();
            let marker = if current { "● " } else { "" };
            let title = truncate(&format!("{marker}{}", item.title), width);
            let details = truncate(
                &format!(
                    "{} · {}{}",
                    ago(sidebar.listed_at - item.updated_at),
                    // Only cloud conversations are flagged, to keep the line short.
                    if app
                        .providers
                        .iter()
                        .any(|p| p.id == item.provider && !p.local)
                    {
                        "☁ "
                    } else {
                        ""
                    },
                    item.model
                ),
                width,
            );
            ListItem::new(vec![
                Line::from(Span::styled(
                    title,
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(details, dim())),
            ])
        })
        .collect();

    let list = List::new(list_items)
        .block(block)
        .highlight_symbol("▌ ")
        .highlight_style(Style::default().bg(Color::DarkGray));
    // A throwaway state: the selection lives in `Sidebar`, rendering only reads it.
    let mut state = ListState::default().with_selected(Some(sidebar.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

/// Cuts `text` to `width` columns, with an ellipsis when shortened.
pub(super) fn truncate(text: &str, width: usize) -> String {
    if display_width(text) <= width {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}

/// Human-readable age of a conversation, in French.
pub fn ago(seconds: i64) -> String {
    const MINUTE: i64 = 60;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;
    match seconds {
        s if s < MINUTE => "à l'instant".to_owned(),
        s if s < HOUR => format!("il y a {} min", s / MINUTE),
        s if s < DAY => format!("il y a {} h", s / HOUR),
        s if s < 2 * DAY => "hier".to_owned(),
        s if s < 30 * DAY => format!("il y a {} j", s / DAY),
        s if s < 365 * DAY => format!("il y a {} mois", s / (30 * DAY)),
        s => format!("il y a {} an(s)", s / (365 * DAY)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_ages() {
        assert_eq!(ago(-5), "à l'instant", "clock skew is harmless");
        assert_eq!(ago(59), "à l'instant");
        assert_eq!(ago(5 * 60), "il y a 5 min");
        assert_eq!(ago(3 * 3600), "il y a 3 h");
        assert_eq!(ago(30 * 3600), "hier");
        assert_eq!(ago(10 * 86_400), "il y a 10 j");
        assert_eq!(ago(90 * 86_400), "il y a 3 mois");
        assert_eq!(ago(800 * 86_400), "il y a 2 an(s)");
    }

    #[test]
    fn truncation() {
        assert_eq!(truncate("court", 10), "court");
        assert_eq!(truncate("un titre bien trop long", 10), "un titre …");
    }
}
