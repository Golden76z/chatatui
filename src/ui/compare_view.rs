//! The /compare popup: the previous reply and the new one side by side.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Paragraph},
};

use crate::{
    app::{App, Comparison},
    state::{Message, Overlay},
    transcript::{Waiting, message_lines_marked},
};

/// The popup: the whole screen but a one-cell margin.
fn popup_area(area: Rect) -> Rect {
    Rect {
        x: area.x.saturating_add(1),
        y: area.y.saturating_add(1),
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    }
}

/// Inner areas: the two headers and the two columns (a separator between them).
fn columns(popup: Rect) -> [Rect; 2] {
    let inner = Block::bordered().inner(popup);
    let [left, _, right] = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(inner);
    [left, right]
}

/// Messages of each side: the previous reply (kept as a version) and the new one.
fn sides<'a>(app: &'a App, comparison: &Comparison) -> [&'a [Message]; 2] {
    let previous = app
        .tails
        .iter()
        .find(|t| t.after == comparison.after && t.number == comparison.previous)
        .map_or(&[][..], |t| t.messages.as_slice());
    [previous, app.conversation.after(comparison.after)]
}

/// Display lines of `messages` for a column of `width` cells.
///
/// `waiting` has to be threaded through: `/compare` starts a job like any other, so the
/// waiting line was being drawn into the main transcript, on the rows the popup covers,
/// while the column the user is actually looking at showed a bare cursor.
fn side_lines(messages: &[Message], width: u16, waiting: Option<&Waiting>) -> Vec<Line<'static>> {
    let width = usize::from(width).max(1);
    let mut lines: Vec<Line<'static>> = messages
        .iter()
        .flat_map(|m| message_lines_marked(m, width, None, waiting))
        .collect();
    if lines.is_empty() {
        lines.push(Line::styled(
            "(rien)",
            Style::default().fg(crate::theme::palette().dim),
        ));
    }
    lines
}

/// Model that wrote a side, from its first reply.
fn model_of(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .find(|m| m.role == crate::state::Role::Assistant)
        .and_then(|m| m.source.as_deref())
}

/// Rows of each column below its header.
fn body_height(area: Rect) -> usize {
    usize::from(
        Block::bordered()
            .inner(popup_area(area))
            .height
            .saturating_sub(2),
    )
}

/// Largest useful scroll offset (the longer side decides).
pub fn max_scroll(app: &App) -> u16 {
    let Some(comparison) = &app.compare else {
        return 0;
    };
    let area = popup_area(app.viewport);
    let [left, right] = columns(area);
    let [a, b] = sides(app, comparison);
    let waiting = app.waiting();
    let longest = side_lines(a, left.width, waiting.as_ref())
        .len()
        .max(side_lines(b, right.width, waiting.as_ref()).len());
    u16::try_from(longest.saturating_sub(body_height(app.viewport))).unwrap_or(u16::MAX)
}

/// Draws the comparison, if one is open.
pub fn render(app: &App, frame: &mut Frame, area: Rect) {
    let (Some(comparison), Some(Overlay::Compare { scroll })) = (&app.compare, &app.overlay) else {
        return;
    };
    let palette = crate::theme::palette();
    let popup = popup_area(area);
    super::clear_modal_rows(frame, area, popup);
    let generating = app.is_generating();
    let hint = if generating {
        " génération… · ↑↓ défiler · Échap fermer "
    } else {
        " ← ou 1 garder la première · 2 ou → la seconde · ↑↓ défiler · Échap "
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette.accent))
        .title(" Comparer les réponses ")
        .title_bottom(Line::from(hint).right_aligned());
    frame.render_widget(block, popup);

    let [left, right] = columns(popup);
    let separator = Rect {
        x: left.right(),
        width: 1,
        ..left
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled("│", Style::default().fg(palette.dim));
            usize::from(separator.height)
        ]),
        separator,
    );

    let [a, b] = sides(app, comparison);
    let new_model = format!(
        "{} › {}",
        app.provider_label(&comparison.provider),
        comparison.model
    );
    let labels = [
        format!("1 · {}", model_of(a).unwrap_or("réponse précédente")),
        format!(
            "2 · {}{}",
            model_of(b).unwrap_or(&new_model),
            if generating { " (en cours)" } else { "" }
        ),
    ];
    let offset = usize::from(*scroll);
    let waiting = app.waiting();
    for ((column, messages), label) in [left, right].into_iter().zip([a, b]).zip(labels) {
        let [head, _, body] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
        ])
        .areas(column);
        frame.render_widget(
            Line::from(Span::styled(
                label,
                Style::default()
                    .fg(palette.badge_fg)
                    .bg(palette.badge_bg)
                    .add_modifier(Modifier::BOLD),
            )),
            head,
        );
        let lines: Vec<Line<'static>> = side_lines(messages, body.width, waiting.as_ref())
            .into_iter()
            .skip(offset)
            .take(usize::from(body.height))
            .collect();
        frame.render_widget(Paragraph::new(lines), body);
    }
}
