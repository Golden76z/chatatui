//! Command palette popup (Ctrl+P).

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Clear, List, ListItem, ListState, Paragraph},
};

use crate::{app::App, markdown::display_width, state::Palette};

const WIDTH: u16 = 70;
const MAX_ROWS: u16 = 12;

/// Draws the palette centred in `area`.
pub fn render(app: &App, palette: &Palette, frame: &mut Frame, area: Rect) {
    let visible = palette.visible();
    let rows = u16::try_from(visible.len().max(1))
        .unwrap_or(MAX_ROWS)
        .min(MAX_ROWS);
    let popup = super::centered(area, WIDTH, rows + 4);
    frame.render_widget(Clear, popup);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(crate::theme::palette().accent))
        .title(" Commandes ")
        .title_bottom(Line::from(" Entrée lancer · Échap fermer ").right_aligned());
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let [filter_area, _, list_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);

    let dim = Style::default().fg(crate::theme::palette().dim);
    let filter = if palette.filter.is_empty() {
        Line::from(Span::styled(" Tapez pour filtrer…", dim))
    } else {
        Line::from(vec![
            Span::styled(" Filtre : ", dim),
            Span::raw(palette.filter.clone()),
            Span::styled("▏", dim),
        ])
    };
    frame.render_widget(Paragraph::new(filter), filter_area);

    if visible.is_empty() {
        frame.render_widget(
            Paragraph::new(" Aucune commande ne correspond.").style(dim),
            list_area,
        );
        return;
    }

    let width = usize::from(list_area.width.saturating_sub(1)); // highlight symbol
    let usage_width = visible
        .iter()
        .map(|c| display_width(&c.usage()) + 1)
        .max()
        .unwrap_or(0);
    let items: Vec<ListItem> = visible
        .iter()
        .map(|spec| {
            let mut usage = format!(" {}", spec.usage());
            usage.push_str(&" ".repeat(usage_width.saturating_sub(display_width(&usage))));
            let shortcut = spec
                .shortcut(app.keyboard_enhanced)
                .map(|s| format!("{s} "))
                .unwrap_or_default();
            let used = display_width(&usage) + display_width(spec.description) + 2;
            let gap = width.saturating_sub(used + display_width(&shortcut)).max(1);
            ListItem::new(Line::from(vec![
                Span::styled(usage, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(spec.description),
                Span::raw(" ".repeat(gap)),
                Span::styled(shortcut, dim),
            ]))
        })
        .collect();
    let list = List::new(items)
        .highlight_symbol("▌")
        .highlight_style(Style::default().bg(crate::theme::palette().selection_bg));
    // A throwaway state: the selection lives in `Palette`.
    let mut state = ListState::default().with_selected(Some(palette.selected));
    frame.render_stateful_widget(list, list_area, &mut state);
}
