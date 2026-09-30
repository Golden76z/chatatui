//! One-line status bar: state, model and key hints.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
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
        ("◐ Génération…".to_owned(), Color::Yellow)
    } else {
        match &app.status {
            Status::Ready | Status::Generating => ("● Prêt".to_owned(), Color::Green),
            Status::Info(message) => (format!("● {message}"), Color::Cyan),
            Status::Error(message) => (format!("✖ {message}"), Color::Red),
        }
    };
    let left = Line::from(vec![
        Span::raw(" "),
        Span::styled(label, Style::default().fg(color)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        // Cloud providers are flagged: the conversation leaves the machine.
        Span::styled(
            if app.is_local() { "" } else { "☁ " },
            Style::default().fg(Color::Yellow),
        ),
        Span::raw(app.model_display()),
    ]);
    let left = with_indexing(app, with_gauge(app, left));

    // State and model take priority over hints: the first hint set that fits is shown.
    let style = Style::default().bg(Color::Black);
    let available = usize::from(area.width).saturating_sub(left.width() + 1);
    let Some(hints) = hint_candidates(app)
        .into_iter()
        .map(|text| {
            Line::from(text)
                .style(Style::default().fg(Color::DarkGray))
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
    let dim = Style::default().fg(Color::DarkGray);
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

/// Appends the indexing progress (`⟳ cours 12/40`) while `/index` runs.
fn with_indexing(app: &App, mut line: Line<'static>) -> Line<'static> {
    let Some(progress) = &app.indexing else {
        return line;
    };
    line.push_span(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));
    let count = if progress.total == 0 {
        "…".to_owned()
    } else {
        format!("{}/{}", progress.done, progress.total)
    };
    line.push_span(Span::styled(
        format!("⟳ {} {count}", progress.collection),
        Style::default().fg(Color::Magenta),
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
        Some(Overlay::ModelPicker(_) | Overlay::Palette(_)) => {
            return vec![
                "↑↓ choisir · Entrée valider · Échap fermer ".into(),
                "Échap fermer ".into(),
            ];
        }
        Some(
            Overlay::Help { .. }
            | Overlay::Context { .. }
            | Overlay::Prompt { .. }
            | Overlay::Collections { .. },
        ) => {
            return vec![
                "↑↓ PgUp PgDn défiler · Échap fermer ".into(),
                "Échap fermer ".into(),
            ];
        }
        None => {}
    }
    if app.sidebar.is_some() {
        return vec![
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
    vec![
        format!("Entrée envoyer · {newline_key} ligne · / commandes · Ctrl+P palette · F1 aide "),
        "Entrée envoyer · / commandes · Ctrl+P palette ".into(),
        "/ commandes · Ctrl+P ".into(),
    ]
}
