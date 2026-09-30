//! Content of the /collections popup: indexed document collections and the last run.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::sidebar::ago;
use crate::{app::App, markdown::wrap_spans, tokens::format_count};

/// Skipped files listed in the report before "… et N autres".
const SKIPPED_SHOWN: usize = 8;

/// Lines of the /collections popup for an inner width of `width` columns.
pub fn lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::DarkGray);
    let title = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let mut lines = Vec::new();
    let wrapped = |lines: &mut Vec<Line<'static>>, spans: Vec<Span<'static>>, indent: usize| {
        let content = width.saturating_sub(indent).max(1);
        for row in wrap_spans(&spans, content) {
            let mut line = vec![Span::raw(" ".repeat(indent))];
            line.extend(row);
            lines.push(Line::from(line));
        }
    };

    match &app.collections {
        None => lines.push(Line::styled(" chargement…", dim)),
        Some((collections, _)) if collections.is_empty() => {
            lines.push(Line::styled(" Aucune collection indexée.", dim));
        }
        Some((collections, now)) => {
            for collection in collections {
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}", collection.name), bold),
                    Span::styled(format!("  {}", ago(now - collection.updated_at)), dim),
                ]));
                wrapped(
                    &mut lines,
                    vec![Span::styled(collection.root.clone(), dim)],
                    3,
                );
                lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::raw(format!(
                        "{} documents · {} passages · {}",
                        format_count(collection.documents),
                        format_count(collection.chunks),
                        collection.embedding_model
                    )),
                ]));
                lines.push(Line::default());
            }
        }
    }

    if let Some(progress) = &app.indexing {
        lines.push(Line::styled(" En cours", title));
        let state = if progress.total == 0 {
            format!("« {} » : recherche des fichiers…", progress.collection)
        } else {
            format!(
                "« {} » : {}/{} fichiers",
                progress.collection, progress.done, progress.total
            )
        };
        wrapped(&mut lines, vec![Span::raw(state)], 1);
        if !progress.current.is_empty() {
            wrapped(
                &mut lines,
                vec![Span::styled(progress.current.clone(), dim)],
                3,
            );
        }
        lines.push(Line::default());
    }

    if let Some(report) = &app.last_index {
        lines.push(Line::styled(" Dernière indexation", title));
        wrapped(
            &mut lines,
            vec![Span::raw(crate::app::index_summary(report))],
            1,
        );
        for (path, reason) in report.skipped.iter().take(SKIPPED_SHOWN) {
            wrapped(
                &mut lines,
                vec![
                    Span::styled("✖ ", Style::default().fg(Color::Yellow)),
                    Span::raw(path.clone()),
                    Span::styled(format!(" — {reason}"), dim),
                ],
                3,
            );
        }
        if report.skipped.len() > SKIPPED_SHOWN {
            lines.push(Line::styled(
                format!("   … et {} autres", report.skipped.len() - SKIPPED_SHOWN),
                dim,
            ));
        }
        lines.push(Line::default());
    }

    wrapped(
        &mut lines,
        vec![
            Span::styled("Embeddings : ", dim),
            Span::raw(format!(
                "{} / {}",
                app.provider_label(&app.rag.embedding_provider),
                app.rag.embedding_model
            )),
        ],
        1,
    );
    wrapped(
        &mut lines,
        vec![Span::styled(
            "/index <dossier> [nom] pour indexer ou mettre à jour (Échap arrête)",
            dim,
        )],
        1,
    );
    lines
}
