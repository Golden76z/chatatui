//! One-line status bar: state, model and key hints.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
};

use super::context_view::gauge_color;
use crate::{
    app::App,
    state::{Overlay, Status},
    tokens,
};

/// Draws the status bar.
pub fn render(app: &App, frame: &mut Frame, area: Rect) {
    let (label, color) = if app.is_generating() {
        let speed = app
            .live_speed()
            .map(|s| format!(" {s} t/s"))
            .unwrap_or_default();
        (
            format!("◐ Génération…{speed}"),
            crate::theme::palette().warn,
        )
    } else {
        match &app.status {
            Status::Ready | Status::Generating => ("● Prêt".to_owned(), crate::theme::palette().ok),
            Status::Info(message) => (format!("● {message}"), crate::theme::palette().accent),
            Status::Error(message) => (format!("✖ {message}"), crate::theme::palette().error),
        }
    };
    let left = Line::from(vec![
        Span::raw(" "),
        Span::styled(label, Style::default().fg(color)),
        Span::styled(" │ ", Style::default().fg(crate::theme::palette().dim)),
        // Cloud providers are flagged: the conversation leaves the machine.
        Span::styled(
            if app.is_local() { "" } else { "☁ " },
            Style::default().fg(crate::theme::palette().warn),
        ),
        Span::raw(app.model_display()),
    ]);
    let left = with_pull(
        app,
        with_indexing(
            app,
            with_rag(
                app,
                with_cost(app, with_gauge(app, with_persona(app, left))),
            ),
        ),
    );

    // State and model take priority over hints: the first hint set that fits is shown.
    let style = Style::default().bg(crate::theme::palette().bar_bg);
    let available = usize::from(area.width).saturating_sub(left.width() + 1);
    let Some(hints) = hint_candidates(app)
        .into_iter()
        .map(|text| {
            Line::from(text)
                .style(Style::default().fg(crate::theme::palette().dim))
                .right_aligned()
        })
        .find(|line| line.width() <= available)
    else {
        frame.render_widget(Paragraph::new(left).style(style), area);
        return;
    };
    let hints_width = u16::try_from(hints.width()).unwrap_or(u16::MAX);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Length(hints_width)]).areas(area);
    frame.render_widget(Paragraph::new(left).style(style), left_area);
    frame.render_widget(Paragraph::new(hints).style(style), right_area);
}

/// Appends the context gauge (`ctx ≈3,2k/8,2k 39 %`) once the conversation has started.
fn with_gauge(app: &App, mut line: Line<'static>) -> Line<'static> {
    if app.conversation.is_empty() {
        return line;
    }
    let usage = app.context_usage();
    let approx = if usage.measured { "" } else { "≈" };
    let used = tokens::format_short(usage.tokens);
    let dim = Style::default().fg(crate::theme::palette().dim);
    line.push_span(Span::styled(" │ ", dim));
    match app.context_window() {
        Some((window, _)) => {
            let percent = tokens::percent(usage.tokens, window);
            let color = gauge_color(percent);
            line.push_span(Span::styled(
                format!(
                    "ctx {approx}{used}/{} {percent} %",
                    tokens::format_short(window)
                ),
                Style::default().fg(color),
            ));
            if percent >= 80 {
                line.push_span(Span::styled(" (/context)", Style::default().fg(color)));
            }
        }
        None => line.push_span(Span::styled(format!("ctx {approx}{used}"), dim)),
    }
    line
}

/// Appends what the conversation cost, for priced providers.
fn with_cost(app: &App, mut line: Line<'static>) -> Line<'static> {
    if let Some((micros, currency)) = &app.conversation_cost {
        line.push_span(Span::styled(
            format!(" · {}", crate::app::format_cost(*micros, currency)),
            Style::default().fg(crate::theme::palette().dim),
        ));
    }
    line
}

/// Appends the named system prompt (`✦ prof`) when one is chosen, and `🔧` when the
/// model may call tools.
fn with_persona(app: &App, mut line: Line<'static>) -> Line<'static> {
    if app.tools_enabled {
        line.push_span(Span::styled(
            " 🔧",
            Style::default().fg(crate::theme::palette().warn),
        ));
    }
    if let Some(persona) = &app.persona {
        line.push_span(Span::styled(
            format!(" ✦ {persona}"),
            Style::default().fg(crate::theme::palette().accent),
        ));
    }
    line
}

/// Appends the collection searched for replies (`/rag`); flagged when it goes to the cloud.
fn with_rag(app: &App, mut line: Line<'static>) -> Line<'static> {
    let names = app.rag_names();
    if names.is_empty() {
        return line;
    }
    let collection = names.join(", ");
    line.push_span(Span::styled(
        " │ ",
        Style::default().fg(crate::theme::palette().dim),
    ));
    if app.is_local() {
        line.push_span(Span::styled(
            format!("⌕ {collection}"),
            Style::default().fg(crate::theme::palette().info),
        ));
    } else {
        // The passages leave the machine with the prompt.
        line.push_span(Span::styled(
            format!("⌕ {collection} ☁"),
            Style::default().fg(crate::theme::palette().warn),
        ));
    }
    line
}

/// Appends the indexing progress (`⟳ cours 12/40`) while `/index` runs.
fn with_indexing(app: &App, mut line: Line<'static>) -> Line<'static> {
    let Some(progress) = &app.indexing else {
        return line;
    };
    line.push_span(Span::styled(
        " │ ",
        Style::default().fg(crate::theme::palette().dim),
    ));
    let count = if progress.total == 0 {
        "…".to_owned()
    } else {
        format!("{}/{}", progress.done, progress.total)
    };
    line.push_span(Span::styled(
        format!("⟳ {} {count}", progress.collection),
        Style::default().fg(crate::theme::palette().assistant),
    ));
    line
}

/// Appends the download progress (`⬇ Q4_K_M 2,1/4,4 Go · 18 Mo/s`) while `/pull` runs.
fn with_pull(app: &App, mut line: Line<'static>) -> Line<'static> {
    let Some(progress) = &app.pulling else {
        return line;
    };
    let done = tokens::format_bytes(progress.done);
    let size = match progress.total {
        Some(total) => {
            let total = tokens::format_bytes(total);
            // `2,0/4,1 Go`, not `2,0 Go/4,1 Go`, when both land on the same unit.
            match (done.rsplit_once(' '), total.rsplit_once(' ')) {
                (Some((value, unit)), Some((_, total_unit))) if unit == total_unit => {
                    format!("{value}/{total}")
                }
                _ => format!("{done}/{total}"),
            }
        }
        None => done,
    };
    let rate = if progress.rate > 0 {
        format!(" · {}/s", tokens::format_bytes(progress.rate))
    } else {
        String::new()
    };
    // The file name carries the quantization, which is what the user is waiting on.
    let name = progress.file.trim_end_matches(".gguf").to_owned();
    line.push_span(Span::styled(
        format!("  ⬇ {name} {size}{rate}"),
        Style::default().fg(crate::theme::palette().info),
    ));
    line
}

/// Hints for the current context, most complete first.
fn hint_candidates(app: &App) -> Vec<String> {
    let newline_key = if app.keyboard_enhanced {
        "Shift+Entrée"
    } else {
        "Alt+Entrée"
    };
    match &app.overlay {
        Some(Overlay::ToolConfirm { .. }) => {
            return vec![
                "Entrée autoriser · t toujours · Échap refuser ".into(),
                "Entrée · t · Échap ".into(),
            ];
        }
        Some(Overlay::Compare { .. }) => {
            return vec![
                "comparaison : la réponse non gardée reste une version (Alt+← / Alt+→) ".into(),
                "Échap fermer ".into(),
            ];
        }
        Some(Overlay::ModelPicker(_) | Overlay::Palette(_) | Overlay::GgufPicker(_)) => {
            return vec![
                "↑↓ choisir · Entrée valider · Échap fermer ".into(),
                "Échap fermer ".into(),
            ];
        }
        Some(Overlay::Models(_)) => {
            return vec![
                "↑↓ choisir · Entrée télécharger · Suppr supprimer · Échap fermer ".into(),
                "↑↓ · Entrée · Suppr · Échap ".into(),
                "Échap fermer ".into(),
            ];
        }
        Some(
            Overlay::Help { .. }
            | Overlay::Context { .. }
            | Overlay::Prompt { .. }
            | Overlay::Collections { .. }
            | Overlay::Mcp { .. },
        ) => {
            return vec![
                "↑↓ PgUp PgDn défiler · Échap fermer ".into(),
                "Échap fermer ".into(),
            ];
        }
        None => {}
    }
    if app.find.is_some() {
        return vec![
            "Entrée suivant · Maj+Entrée précédent · Échap fermer ".into(),
            "Entrée ↓ · Maj+Entrée ↑ · Échap fermer ".into(),
            "↑↓ · Échap ".into(),
        ];
    }
    if let Some(sidebar) = &app.sidebar {
        if sidebar.rename.is_some() {
            return vec![
                "Entrée enregistrer · Échap annuler ".into(),
                "Entrée · Échap ".into(),
            ];
        }
        return vec![
            "↑↓ choisir · Entrée ouvrir · tapez pour chercher · Échap fermer ".into(),
            "↑↓ choisir · Entrée ouvrir · Échap fermer ".into(),
            "Entrée ouvrir · Échap fermer ".into(),
        ];
    }
    if !app.suggestions().is_empty() {
        return vec![
            "↑↓ choisir · Tab compléter · Entrée lancer · Échap masquer ".into(),
            "Tab compléter · Entrée lancer ".into(),
        ];
    }
    if app.editing.is_some() {
        return vec![
            "✎ modification : Entrée renvoie · Échap annule ".into(),
            "✎ Entrée · Échap ".into(),
        ];
    }
    if app.is_generating() {
        return vec![
            "Échap annuler · Ctrl+C quitter ".into(),
            "Échap annuler ".into(),
        ];
    }
    if app.indexing.is_some() && app.input.is_empty() {
        return vec![
            "Échap arrêter l'indexation · /collections détails ".into(),
            "Échap arrêter ".into(),
        ];
    }
    if app.pulling.is_some() && app.input.is_empty() {
        return vec![
            "Échap arrêter le téléchargement · /models détails ".into(),
            "Échap arrêter ".into(),
        ];
    }
    vec![
        format!("Entrée envoyer · {newline_key} ligne · / commandes · Ctrl+P palette · F1 aide "),
        "Entrée envoyer · / commandes · Ctrl+P palette ".into(),
        "/ commandes · Ctrl+P ".into(),
    ]
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend};

    use super::*;
    use crate::config::Config;

    fn test_app() -> App {
        App::new(&Config::default(), false)
    }

    /// The whole bar drawn on one row of `width` columns, as plain text.
    fn line_text(app: &App, width: u16) -> String {
        let mut terminal =
            Terminal::new(TestBackend::new(width, 1)).expect("test backend never fails");
        terminal
            .draw(|frame| render(app, frame, frame.area()))
            .expect("test backend never fails");
        let buffer = terminal.backend().buffer().clone();
        (0..width)
            .map(|x| buffer[(x, 0)].symbol().to_owned())
            .collect()
    }

    #[test]
    fn shows_the_download_progress_and_its_rate() {
        let mut app = test_app();
        app.pulling = Some(crate::app::PullProgress {
            repo: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_owned(),
            file: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_owned(),
            done: 2_100_000_000,
            total: Some(4_400_000_000),
            rate: 18_000_000,
        });

        let text = line_text(&app, 120);

        assert!(text.contains('⬇'), "{text}");
        assert!(text.contains("Q4_K_M"), "{text}");
        assert!(text.contains("2,0/4,1 Go"), "{text}");
        assert!(text.contains("/s"), "{text}");
    }

    #[test]
    fn a_download_of_unknown_size_shows_what_it_has() {
        let mut app = test_app();
        app.pulling = Some(crate::app::PullProgress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 1_048_576,
            total: None,
            rate: 0,
        });

        let text = line_text(&app, 120);

        assert!(text.contains("1,0 Mo"), "{text}");
        // No total, so no percentage and no bogus denominator.
        assert!(!text.contains('%'), "{text}");
        // No rate yet: no trailing separator dangling on its own.
        assert!(!text.contains("· /s"), "{text}");
    }

    #[test]
    fn a_running_download_advertises_how_to_stop_it() {
        let mut app = test_app();
        app.pulling = Some(crate::app::PullProgress {
            repo: "owner/name".to_owned(),
            file: "m.gguf".to_owned(),
            done: 0,
            total: None,
            rate: 0,
        });

        assert!(
            hint_candidates(&app)[0].contains("Échap arrêter le téléchargement"),
            "{:?}",
            hint_candidates(&app)
        );
    }
}
