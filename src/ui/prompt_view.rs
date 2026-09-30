//! Content of the /prompt popup: the exact messages of the next request.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use crate::{
    app::App,
    llm::ChatRole,
    markdown::{wrap_plain, wrap_spans},
    tokens::{self, format_count},
};

/// Lines of the /prompt popup for an inner width of `width` columns.
pub fn lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::DarkGray);
    let destination = if app.is_local() {
        "local"
    } else {
        "☁ envoyé hors de cette machine"
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!(" → {}", app.model_display()),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  ({destination})"), dim),
    ])];
    let note = [Span::styled(
        "Prochaine requête, dans l'ordre. Une source de contexte (RAG) pourra y ajouter des extraits.",
        dim,
    )];
    for wrapped in wrap_spans(&note, width.saturating_sub(1).max(1)) {
        let mut spans = vec![Span::raw(" ")];
        spans.extend(wrapped);
        lines.push(Line::from(spans));
    }
    lines.push(Line::default());

    let prompt = app.prompt();
    if prompt.is_empty() {
        lines.push(Line::styled(" (rien pour l'instant)", dim));
    }
    let content_width = width.saturating_sub(2).max(1);
    for message in prompt {
        let (name, color) = match message.role {
            ChatRole::System => ("system", Color::DarkGray),
            ChatRole::User => ("user", Color::Cyan),
            ChatRole::Assistant => ("assistant", Color::Magenta),
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!(" ── {name} "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "≈ {} tokens",
                    format_count(tokens::estimate_message(&message.content))
                ),
                dim,
            ),
        ]));
        for source_line in message.content.lines() {
            for wrapped in wrap_plain(&[Span::raw(source_line.to_owned())], content_width) {
                let mut spans = vec![Span::raw("  ")];
                spans.extend(wrapped);
                lines.push(Line::from(spans));
            }
        }
        lines.push(Line::default());
    }
    lines
}
