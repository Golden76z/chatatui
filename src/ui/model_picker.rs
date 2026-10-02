//! Model selection popup, drawn over the rest of the screen.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, List, ListItem, ListState, Paragraph},
};

use crate::{
    app::App,
    markdown::display_width,
    state::{ModelList, ModelPicker},
};

const MAX_ROWS: u16 = 14;

/// Draws the popup centred in `area`.
pub fn render(app: &App, picker: &ModelPicker, frame: &mut Frame, area: Rect) {
    let visible = picker.visible();
    let errors = picker.errors();
    let rows = match &picker.models {
        ModelList::Loaded { .. } => u16::try_from(visible.len().max(1))
            .unwrap_or(MAX_ROWS)
            .min(MAX_ROWS),
        ModelList::Loading => 1,
    };
    let error_rows = u16::try_from(errors.len()).unwrap_or(u16::MAX);
    let separator = u16::from(error_rows > 0);
    // Borders + filter line + blank line + rows (+ blank + errors).
    let popup = super::centered(
        area,
        super::popup_width(area),
        rows + 4 + separator + error_rows,
    );
    super::clear_modal_rows(frame, area, popup);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(crate::theme::palette().accent))
        .title(" Modèle ")
        .title_bottom(Line::from(" Entrée choisir · Échap fermer ").right_aligned());
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let [filter_area, _, list_area, _, errors_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(separator),
        Constraint::Length(error_rows),
    ])
    .areas(inner);

    let dim = Style::default().fg(crate::theme::palette().dim);
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

    // One line per provider that failed; most messages already name the provider.
    let error_width = usize::from(errors_area.width);
    let error_lines: Vec<Line> = errors
        .iter()
        .map(|(label, error)| {
            let text = if error.starts_with(label.as_str()) {
                format!(" ✖ {error}")
            } else {
                format!(" ✖ {label} : {error}")
            };
            Line::from(Span::styled(
                super::sidebar::truncate(&text, error_width),
                Style::default().fg(crate::theme::palette().error),
            ))
        })
        .collect();
    frame.render_widget(Paragraph::new(error_lines), errors_area);

    match &picker.models {
        ModelList::Loading => {
            let text = Paragraph::new(" Chargement des modèles…").style(dim);
            frame.render_widget(text, list_area);
        }
        ModelList::Loaded { .. } if visible.is_empty() => {
            let text = Paragraph::new(" Aucun modèle disponible.").style(dim);
            frame.render_widget(text, list_area);
        }
        ModelList::Loaded { .. } => {
            let label_width = visible
                .iter()
                .map(|c| display_width(&c.label))
                .max()
                .unwrap_or(0);
            let items: Vec<ListItem> = visible
                .iter()
                .map(|choice| {
                    let current = choice.provider == app.provider && choice.model == app.model;
                    let marker = if current { "● " } else { "  " };
                    let local = app
                        .providers
                        .iter()
                        .find(|p| p.id == choice.provider)
                        .is_some_and(|p| p.local);
                    let place = if local { "  " } else { "☁ " };
                    let padding =
                        " ".repeat(label_width.saturating_sub(display_width(&choice.label)) + 2);
                    let model_style = if current {
                        Style::default().add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    let mut spans = vec![
                        Span::styled(marker, Style::default().fg(crate::theme::palette().ok)),
                        Span::styled(place, Style::default().fg(crate::theme::palette().warn)),
                        Span::styled(format!("{}{padding}", choice.label), dim),
                        Span::styled(choice.model.clone(), model_style),
                    ];
                    if let Some(window) = choice.context_window {
                        spans.push(Span::styled(format!("  {}k", window / 1000), dim));
                    }
                    ListItem::new(Line::from(spans))
                })
                .collect();
            let list = List::new(items)
                .highlight_symbol("▌")
                .highlight_style(Style::default().bg(crate::theme::palette().selection_bg));
            // A throwaway state: the selection lives in `ModelPicker`.
            let mut state = ListState::default().with_selected(Some(picker.selected));
            frame.render_stateful_widget(list, list_area, &mut state);
        }
    }
}
