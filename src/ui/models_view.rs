//! Content of the /models popup: the models downloaded on this machine.

use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use super::sidebar::ago;
use crate::{
    app::App,
    markdown::wrap_spans,
    models::store::LocalModel,
    tokens::{format_bytes, format_count},
};

/// Lines of the /models popup for an inner width of `width` columns.
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

    match &app.models {
        None => lines.push(Line::styled(" chargement…", dim)),
        Some((models, _)) if models.is_empty() => {
            lines.push(Line::styled(
                " Aucun modèle téléchargé. /pull <dépôt> pour en ajouter.",
                dim,
            ));
        }
        Some((models, now)) => {
            for model in models {
                let mut head = vec![
                    Span::styled(format!(" {}", model.repo), bold),
                    Span::styled(format!("  {}", ago(now - model.downloaded_at)), dim),
                ];
                if model.sha256.is_none() {
                    // The Hub exposed no checksum, so nothing was compared: say so.
                    head.push(Span::styled(
                        "  ⚠ non vérifié",
                        Style::default().fg(palette.warn),
                    ));
                }
                lines.push(Line::from(head));
                wrapped(&mut lines, vec![Span::styled(model.file.clone(), dim)], 3);
                wrapped(&mut lines, vec![Span::styled(details(model), dim)], 3);
                lines.push(Line::default());
            }
        }
    }

    wrapped(
        &mut lines,
        vec![Span::styled(
            "/pull <dépôt> [fichier] télécharge · /models cette liste · /rm <dépôt> <fichier> \
             supprime le fichier",
            dim,
        )],
        1,
    );
    wrapped(
        &mut lines,
        vec![Span::styled(
            "Les modèles locaux ne sont pas encore utilisables pour répondre (J33) : ce jalon \
             télécharge et inspecte les fichiers, il ne les exécute pas.",
            dim,
        )],
        1,
    );
    lines
}

/// The one-line summary under a model: quantization, size, architecture, window, parameters.
fn details(model: &LocalModel) -> String {
    let mut parts = Vec::new();
    if let Some(quantization) = &model.quantization {
        parts.push(quantization.clone());
    }
    parts.push(format_bytes(model.bytes));
    if let Some(architecture) = &model.architecture {
        parts.push(architecture.clone());
    }
    if let Some(context) = model.context_length {
        parts.push(format!("{} tokens", format_count(context)));
    }
    if let Some(parameters) = model.parameters {
        parts.push(format!("{} paramètres", format_count(parameters)));
    }
    parts.join(" · ")
}
