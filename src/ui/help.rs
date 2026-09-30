//! Help screen (F1, /help): commands and key bindings.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use crate::{app::App, commands::COMMANDS, markdown::display_width};

/// Help text: commands and key bindings.
pub fn lines(app: &App) -> Vec<Line<'static>> {
    let title = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::DarkGray);

    let mut lines = vec![Line::styled(" Commandes (tapez / ou Ctrl+P)", title)];
    let width = COMMANDS
        .iter()
        .map(|c| display_width(&c.usage()))
        .max()
        .unwrap_or(0);
    for spec in COMMANDS {
        let usage = spec.usage();
        let padding = " ".repeat(width.saturating_sub(display_width(&usage)) + 2);
        let mut spans = vec![
            Span::styled(format!("  {usage}"), bold),
            Span::raw(padding),
            Span::raw(spec.description),
        ];
        if let Some(shortcut) = spec.shortcut(app.keyboard_enhanced) {
            spans.push(Span::styled(format!("  ({shortcut})"), dim));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::styled(
        "  //texte envoie un message qui commence par /",
        dim,
    ));

    let newline = if app.keyboard_enhanced {
        "Shift+Entrée, Alt+Entrée"
    } else {
        "Alt+Entrée, Ctrl+J"
    };
    let keys: [(&str, &str); 9] = [
        ("Entrée", "envoyer"),
        (newline, "nouvelle ligne"),
        ("Échap", "fermer / annuler la génération"),
        ("PgUp PgDn, molette", "faire défiler"),
        ("↑ ↓ (saisie vide)", "défiler d'une ligne"),
        ("Ctrl+Début Ctrl+Fin", "haut / bas de la conversation"),
        ("Ctrl+P", "palette de commandes"),
        ("↑ ↓ PgUp PgDn", "faire défiler une fenêtre comme celle-ci"),
        (
            "Liste (Ctrl+L)",
            "tapez pour chercher · Ctrl+R renommer · Suppr supprimer",
        ),
    ];
    lines.push(Line::default());
    lines.push(Line::styled(" Raccourcis", title));
    let key_width = keys
        .iter()
        .map(|(k, _)| display_width(k))
        .max()
        .unwrap_or(0);
    for (key, action) in keys {
        let padding = " ".repeat(key_width.saturating_sub(display_width(key)) + 2);
        lines.push(Line::from(vec![
            Span::styled(format!("  {key}"), bold),
            Span::raw(padding),
            Span::raw(action),
        ]));
    }
    lines
}
