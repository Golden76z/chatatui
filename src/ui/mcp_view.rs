//! Content of the /mcp popup: configured MCP servers, their state and their tools.

use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::{app::App, markdown::wrap_spans, mcp::ServerState};

/// Lines of the /mcp popup for an inner width of `width` columns.
pub fn lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let palette = crate::theme::palette();
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(palette.dim);
    let mut lines = Vec::new();
    let wrapped = |lines: &mut Vec<Line<'static>>, spans: Vec<Span<'static>>, indent: usize| {
        let content = width.saturating_sub(indent).max(1);
        for row in wrap_spans(&spans, content) {
            let mut line = vec![Span::raw(" ".repeat(indent))];
            line.extend(row);
            lines.push(Line::from(line));
        }
    };

    if app.mcp_servers.is_empty() {
        lines.push(Line::styled(" Aucun serveur configuré.", dim));
        lines.push(Line::default());
        for text in [
            "Ajoutez-en dans config.toml, par exemple :",
            "[mcp.fichiers]",
            "command = \"npx\"",
            "args = [\"-y\", \"@modelcontextprotocol/server-filesystem\", \"~/Documents\"]",
        ] {
            wrapped(&mut lines, vec![Span::styled(text.to_owned(), dim)], 1);
        }
        return lines;
    }
    for (name, state) in &app.mcp_servers {
        let (label, style) = match state {
            None => ("désactivé".to_owned(), dim),
            Some(ServerState::Starting) => {
                ("démarrage…".to_owned(), Style::default().fg(palette.info))
            }
            Some(ServerState::Ready { info, tools }) => {
                let info = if info.is_empty() {
                    String::new()
                } else {
                    format!(" · {info}")
                };
                (
                    format!("✓ {} outil(s){info}", tools.len()),
                    Style::default().fg(palette.ok),
                )
            }
            Some(ServerState::Failed(_)) => {
                ("✖ échec".to_owned(), Style::default().fg(palette.error))
            }
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {name}"), bold),
            Span::styled(format!("  {label}"), style),
        ]));
        match state {
            Some(ServerState::Ready { tools, .. }) => {
                for tool in tools {
                    let mut spans = vec![Span::raw(tool.name.clone())];
                    if !tool.description.is_empty() {
                        let first = tool.description.lines().next().unwrap_or_default();
                        spans.push(Span::styled(format!("  {first}"), dim));
                    }
                    wrapped(&mut lines, spans, 3);
                }
            }
            Some(ServerState::Failed(error)) => {
                wrapped(
                    &mut lines,
                    vec![Span::styled(
                        error.clone(),
                        Style::default().fg(palette.error),
                    )],
                    3,
                );
            }
            _ => {}
        }
        lines.push(Line::default());
    }
    let note = if app.tools_enabled {
        "Les outils sont proposés au modèle ; chaque appel vous est demandé."
    } else {
        "Outils désactivés : /tools on pour les proposer au modèle (chaque appel vous est demandé)."
    };
    wrapped(&mut lines, vec![Span::styled(note.to_owned(), dim)], 1);
    lines
}
