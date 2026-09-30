//! The question shown when the model asks to run a tool.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Clear, Paragraph, Wrap},
};

use crate::{app::App, markdown::display_width, theme::palette};

/// Draws the confirmation popup centred in `area`.
pub fn render(
    app: &App,
    description: &str,
    tool: &str,
    cloud: bool,
    frame: &mut Frame,
    area: Rect,
) {
    let p = palette();
    let dim = Style::default().fg(p.dim);
    let mut lines = vec![
        Line::from(vec![
            Span::raw(" Le modèle veut "),
            Span::styled(
                description.to_owned(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::styled(format!(" outil : {tool}"), dim),
    ];
    if tool.contains(crate::mcp::SEPARATOR) {
        lines.push(Line::styled(
            " ⚠ outil d'un serveur MCP : il peut aussi modifier ou envoyer des données",
            Style::default().fg(p.warn),
        ));
    }
    if cloud {
        lines.push(Line::styled(
            format!(
                " ☁ le résultat sera envoyé à {}",
                app.provider_label(&app.provider)
            ),
            Style::default().fg(p.warn),
        ));
    }
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::styled(
            " Entrée",
            Style::default().fg(p.ok).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" autoriser   "),
        Span::styled(
            "t",
            Style::default().fg(p.warn).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" toujours   "),
        Span::styled(
            "Échap",
            Style::default().fg(p.error).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" refuser"),
    ]));
    let widest = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| display_width(&s.content))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0);
    let width = u16::try_from(widest + 3)
        .unwrap_or(u16::MAX)
        .clamp(30, area.width.saturating_sub(2));
    // Long paths wrap: count the rows each line takes inside the borders.
    let inner = usize::from(width.saturating_sub(2)).max(1);
    let rows: usize = lines
        .iter()
        .map(|l| {
            let w: usize = l.spans.iter().map(|s| display_width(&s.content)).sum();
            // Word wrapping may need one row more than the exact division.
            if w > inner { w.div_ceil(inner) + 1 } else { 1 }
        })
        .sum();
    let height = u16::try_from(rows + 2).unwrap_or(u16::MAX).min(area.height);
    let popup = super::centered(area, width, height);
    frame.render_widget(Clear, popup);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(p.warn))
        .title(" Autoriser l'outil ? ");
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block),
        popup,
    );
}
