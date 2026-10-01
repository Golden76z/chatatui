//! File selection popup for `/pull`: which GGUF of a repository to download.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, BorderType, Clear, List, ListItem, ListState, Paragraph},
};

use crate::{markdown::display_width, state::GgufPicker, tokens::format_bytes};

const WIDTH: u16 = 68;
const MAX_ROWS: u16 = 12;

/// Draws the popup centred in `area`.
pub fn render(picker: &GgufPicker, frame: &mut Frame, area: Rect) {
    let visible = picker.visible();
    let rows = u16::try_from(visible.len().max(1))
        .unwrap_or(MAX_ROWS)
        .min(MAX_ROWS);
    // Borders + repository + filter + blank line + rows.
    let popup = super::centered(area, WIDTH, rows + 5);
    frame.render_widget(Clear, popup);

    let palette = crate::theme::palette();
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette.accent))
        .title(" Télécharger un modèle ")
        .title_bottom(Line::from(" Entrée télécharger · Échap annuler ").right_aligned());
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let [repo_area, filter_area, _, list_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);

    let dim = Style::default().fg(palette.dim);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(" {}", picker.repo.clone()),
            Style::default().fg(palette.accent),
        ))),
        repo_area,
    );

    let filter = if picker.filter.is_empty() {
        Line::from(Span::styled(" Tapez pour filtrer…", dim))
    } else {
        Line::from(vec![
            Span::styled(" Filtre : ", dim),
            Span::raw(picker.filter.clone()),
            Span::styled("▏", dim),
        ])
    };
    frame.render_widget(Paragraph::new(filter), filter_area);

    if visible.is_empty() {
        frame.render_widget(
            Paragraph::new(" Aucun fichier ne correspond.").style(dim),
            list_area,
        );
        return;
    }

    // The highlight symbol eats one column; one more is left as a margin before the border.
    let room = usize::from(list_area.width).saturating_sub(2);
    let items: Vec<ListItem> = visible
        .iter()
        .map(|file| {
            let size = format_bytes(file.bytes);
            let name = super::sidebar::truncate(
                &file.path,
                room.saturating_sub(display_width(&size) + 2).max(1),
            );
            let padding = " ".repeat(
                room.saturating_sub(display_width(&name) + display_width(&size))
                    .max(1),
            );
            ListItem::new(Line::from(vec![
                Span::raw(name),
                Span::raw(padding),
                Span::styled(size, dim),
            ]))
        })
        .collect();
    let list = List::new(items)
        .highlight_symbol("▌")
        .highlight_style(Style::default().bg(palette.selection_bg));
    // A throwaway state: the selection lives in `GgufPicker`.
    let mut state = ListState::default().with_selected(Some(picker.selected));
    frame.render_stateful_widget(list, list_area, &mut state);
}
