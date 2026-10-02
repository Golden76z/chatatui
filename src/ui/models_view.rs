//! The `/models` popup: the models on this machine, then the ones the catalogue offers.
//!
//! One list, two kinds of row. A downloaded model shows what its own header said; an
//! offered one shows what it is for, so a person with an empty store still has something
//! to choose from.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, List, ListItem, ListState, Paragraph},
};

use super::sidebar::{ago, truncate};

use crate::{
    markdown::display_width,
    models::store::LocalModel,
    state::{ModelRow, ModelsPicker},
    tokens::{format_bytes, format_count},
};

const MAX_ROWS: u16 = 10;

/// Draws the popup centred in `area`.
pub fn render(picker: &ModelsPicker, frame: &mut Frame, area: Rect) {
    let visible = picker.visible();
    let rows = u16::try_from(visible.len().max(1))
        .unwrap_or(MAX_ROWS)
        .min(MAX_ROWS);
    // Borders + filter + blank line + rows + blank line + detail + caveat.
    let popup = super::centered(area, super::popup_width(area), rows + 7);
    super::clear_modal_rows(frame, area, popup);

    let palette = crate::theme::palette();
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette.accent))
        .title(" Modèles ")
        .title_bottom(
            Line::from(" Entrée télécharger · Suppr supprimer · Échap fermer ").right_aligned(),
        );
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let [filter_area, _, list_area, _, detail_area, footer_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(inner);

    let dim = Style::default().fg(palette.dim);
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

    // The highlight symbol eats one column; one more is left as a margin before the border.
    let room = usize::from(list_area.width).saturating_sub(2);
    if visible.is_empty() {
        let text = if picker.loaded {
            " Aucun modèle ne correspond."
        } else {
            " chargement…"
        };
        frame.render_widget(Paragraph::new(text).style(dim), list_area);
    } else {
        let items: Vec<ListItem> = visible
            .iter()
            .map(|row| ListItem::new(line(row, room, palette)))
            .collect();
        let list = List::new(items)
            .highlight_symbol("▌")
            .highlight_style(Style::default().bg(palette.selection_bg));
        // A throwaway state: the selection lives in `ModelsPicker`.
        let mut state = ListState::default().with_selected(Some(picker.selected));
        frame.render_stateful_widget(list, list_area, &mut state);
    }

    // Everything a row has no space for, for the highlighted one only.
    let detail = picker
        .selected()
        .map_or_else(String::new, |row| detail(row, picker.now));
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(" {}", truncate(&detail, room.max(1))),
            dim,
        ))),
        detail_area,
    );

    // A reply to the last key press, or the standing caveat. The status bar shows overlay
    // hints while a popup is open, so this is the only place either can be read.
    let (text, style) = match &picker.message {
        Some(message) => (message.clone(), Style::default().fg(palette.warn)),
        None => (
            "Téléchargés ou non : un modèle local ne peut pas encore répondre (J33).".to_owned(),
            dim,
        ),
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(" {}", truncate(&text, room.max(1))),
            style,
        ))),
        footer_area,
    );
}

/// The detail line under the list: what the highlighted row has no room for.
fn detail(row: &ModelRow, now: i64) -> String {
    match row {
        ModelRow::Downloaded(model) => {
            let mut parts = vec![model.file.clone()];
            if let Some(architecture) = &model.architecture {
                parts.push(architecture.clone());
            }
            if let Some(context) = model.context_length {
                parts.push(format!("{} tokens", format_count(context)));
            }
            if let Some(parameters) = model.parameters {
                parts.push(format!("{} paramètres", format_count(parameters)));
            }
            parts.push(ago(now - model.downloaded_at));
            parts.join(" · ")
        }
        ModelRow::Available(entry) => format!("{} · Entrée choisit la quantization", entry.repo),
    }
}

/// One row: a marker, the name on the left, the size or the parameter count on the right.
///
/// `●` is on disk, `○` is offered. Everything else a person might want — the context
/// window, when it landed — is detail that would squeeze the name into nothing, so it
/// stays out of the row.
fn line(row: &ModelRow, room: usize, palette: &crate::theme::Palette) -> Line<'static> {
    let dim = Style::default().fg(palette.dim);
    let (marker, name, right, style) = match row {
        ModelRow::Downloaded(model) => (
            "● ",
            model.repo.clone(),
            details(model),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        ModelRow::Available(entry) => (
            "○ ",
            format!("{} — {}", entry.name, entry.note),
            entry.parameters.to_owned(),
            Style::default(),
        ),
    };
    let right_width = display_width(&right);
    let left = room.saturating_sub(right_width + 4);
    let name = truncate(&name, left.max(1));
    let padding = " ".repeat(
        room.saturating_sub(display_width(marker) + display_width(&name) + right_width)
            .max(1),
    );
    Line::from(vec![
        Span::styled(marker, dim),
        Span::styled(name, style),
        Span::raw(padding),
        Span::styled(right, dim),
    ])
}

/// The right-hand summary of a downloaded model: quantization, size, and whether its
/// checksum could be compared at all.
fn details(model: &LocalModel) -> String {
    let mut text = match &model.quantization {
        Some(quantization) => format!("{quantization} · {}", format_bytes(model.bytes)),
        None => format_bytes(model.bytes),
    };
    if model.sha256.is_none() {
        // The Hub exposed no checksum, so nothing was compared: say so.
        text.push_str(" ⚠");
    }
    text
}
