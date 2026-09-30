//! Content of the /context popup: window size, usage and where the tokens go.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use crate::{
    app::{App, WindowSource},
    llm::ChatRole,
    markdown::display_width,
    state::{MessageStatus, Role},
    tokens::{self, format_count},
};

const GAUGE_CELLS: usize = 30;
const LONGEST: usize = 5;

/// Lines of the /context popup for an inner width of `width` columns.
pub fn lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let bold = Style::default().add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::DarkGray);
    let title = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let label = |text: &str| Span::styled(format!(" {text:<22}"), dim);

    let place = if app.is_local() {
        "local : rien ne quitte cette machine"
    } else {
        "☁ cloud : la conversation est envoyée au fournisseur"
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!(" {}", app.model_display()), bold),
            Span::styled(format!("  ({place})"), dim),
        ]),
        Line::default(),
    ];

    let usage = app.context_usage();
    let window = app.context_window();
    lines.push(Line::from(vec![
        label("Fenêtre de contexte"),
        match window {
            Some((tokens, source)) => Span::raw(format!(
                "{} tokens ({})",
                format_count(tokens),
                match source {
                    WindowSource::Config => "configuration",
                    WindowSource::Server => "serveur",
                }
            )),
            None => Span::styled("inconnue", Style::default().fg(Color::Yellow)),
        },
    ]));
    lines.push(Line::from(vec![
        label("Occupé"),
        Span::raw(if usage.measured {
            format!(
                "{} tokens, mesuré par le serveur",
                format_count(usage.tokens)
            )
        } else {
            format!("≈ {} tokens, estimé", format_count(usage.tokens))
        }),
    ]));
    if let Some((total, _)) = window {
        let percent = tokens::percent(usage.tokens, total);
        let filled = usize::try_from(percent.min(100)).unwrap_or(100) * GAUGE_CELLS / 100;
        let color = gauge_color(percent);
        lines.push(Line::from(vec![
            label(""),
            Span::styled("█".repeat(filled), Style::default().fg(color)),
            Span::styled("░".repeat(GAUGE_CELLS - filled), dim),
            Span::styled(format!(" {percent} %"), Style::default().fg(color)),
        ]));
    }
    if let Some(measured) = app.measured {
        let output = measured
            .output_tokens
            .map_or_else(|| "?".to_owned(), format_count);
        lines.push(Line::from(vec![
            label("Dernière requête"),
            Span::raw(format!(
                "entrée {} · sortie {output} (mesuré)",
                format_count(measured.input_tokens)
            )),
        ]));
    }

    // Breakdown of the prompt, estimated per part.
    let prompt = app.prompt();
    let (system, history): (Vec<_>, Vec<_>) =
        prompt.iter().partition(|m| m.role == ChatRole::System);
    let system_tokens: u64 = system
        .iter()
        .map(|m| tokens::estimate_message(&m.content))
        .sum();
    let history_tokens: u64 = history
        .iter()
        .map(|m| tokens::estimate_message(&m.content))
        .sum();
    let total = (system_tokens + history_tokens).max(1);
    let row = |name: String, count: u64, note: &str| {
        Line::from(vec![
            Span::raw(format!("   {name:<26}")),
            Span::raw(format!("{:>8}", format_count(count))),
            Span::styled(format!("  {:>3} %", tokens::percent(count, total)), dim),
            Span::styled(note.to_owned(), dim),
        ])
    };
    lines.push(Line::default());
    lines.push(Line::styled(" Répartition (estimation)", title));
    let in_context = app.conversation.context_messages();
    let attachments: Vec<_> = in_context
        .iter()
        .filter(|m| m.role == Role::Attachment)
        .collect();
    let attached_tokens: u64 = attachments
        .iter()
        .map(|m| tokens::estimate(&m.content))
        .sum();
    let summary_tokens: u64 = in_context
        .iter()
        .filter(|m| m.role == Role::Summary && m.status == MessageStatus::Complete)
        .map(|m| tokens::estimate(&m.content))
        .sum();
    let retrieved = app
        .retrieved
        .as_ref()
        .map_or(&[][..], |r| r.chunks.as_slice());
    let retrieved_tokens: u64 = retrieved.iter().map(|c| tokens::estimate(&c.text)).sum();
    // Attached files, passages and the summary travel inside the system message: count
    // them apart.
    lines.push(row(
        "Prompt système".into(),
        system_tokens.saturating_sub(attached_tokens + summary_tokens + retrieved_tokens),
        "",
    ));
    if app.rag_collection.is_some() || !retrieved.is_empty() {
        lines.push(row(
            format!("Extraits RAG ({})", retrieved.len()),
            retrieved_tokens,
            "  dernière réponse",
        ));
    }
    lines.push(row(
        format!("Fichiers joints ({})", attachments.len()),
        attached_tokens,
        "",
    ));
    if summary_tokens > 0 {
        lines.push(row("Résumé (/compact)".into(), summary_tokens, ""));
    }
    lines.push(row(
        format!("Historique ({} messages)", history.len()),
        history_tokens,
        "",
    ));

    // Document search.
    let names = app.rag_names();
    if !names.is_empty() {
        lines.push(Line::default());
        lines.push(Line::styled(" Documents (/rag)", title));
        let place = if app.is_local() {
            Span::styled("  (local)", dim)
        } else {
            Span::styled(
                "  ☁ les extraits sont envoyés au fournisseur",
                Style::default().fg(Color::Yellow),
            )
        };
        lines.push(Line::from(vec![
            label(if names.len() > 1 {
                "Collections"
            } else {
                "Collection"
            }),
            Span::raw(names.join(", ")),
            place,
        ]));
        lines.push(Line::from(vec![
            label("Par réponse"),
            Span::raw(format!(
                "{} extraits au plus, ≈ {} tokens",
                app.rag.top_k,
                format_count(app.rag.context_tokens)
            )),
        ]));
        if let Some(retrieved) = &app.retrieved {
            for (i, chunk) in retrieved.chunks.iter().enumerate() {
                let head = format!("   [{}] ", retrieved.first_number + i);
                let tail = format!("  ≈ {}", format_count(tokens::estimate(&chunk.text)));
                let room = width
                    .saturating_sub(display_width(&head) + display_width(&tail))
                    .max(8);
                lines.push(Line::from(vec![
                    Span::raw(head),
                    Span::raw(shorten(&chunk.label(), room)),
                    Span::styled(tail, dim),
                ]));
            }
        }
    }

    // Longest messages.
    let mut messages: Vec<_> = app
        .conversation
        .messages()
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m, tokens::estimate_message(&m.content)))
        .collect();
    messages.sort_by_key(|m| std::cmp::Reverse(m.2));
    if !messages.is_empty() {
        lines.push(Line::default());
        lines.push(Line::styled(" Messages les plus longs (estimation)", title));
        let prefix_width = 26;
        for (index, message, count) in messages.into_iter().take(LONGEST) {
            let who = match message.role {
                Role::User => "Vous",
                Role::Assistant => "Assistant",
                Role::System => "Système",
                Role::Attachment => "Fichier",
                Role::Summary => "Résumé",
            };
            let head = format!("   #{:<3} {who:<10}{:>8}  ", index + 1, format_count(count));
            let room = width
                .saturating_sub(prefix_width + display_width(&head))
                .max(8);
            let excerpt: String = message
                .content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let excerpt = shorten(&excerpt, room + prefix_width);
            lines.push(Line::from(vec![
                Span::raw(head),
                Span::styled(excerpt, dim),
            ]));
        }
    }

    // Hints.
    lines.push(Line::default());
    if window.is_some_and(|(total, _)| tokens::percent(usage.tokens, total) >= 80) {
        lines.push(Line::styled(
            " Contexte presque plein : /compact résume l'historique, /clear repart de zéro.",
            Style::default().fg(Color::Yellow),
        ));
    }
    if app.conversation.context_start() > 0 {
        let hidden = app.conversation.messages().len() - in_context.len();
        lines.push(Line::styled(
            format!(" {hidden} message(s) plus anciens ne sont plus envoyés (/clear ou /compact)."),
            dim,
        ));
    }
    if window.is_none() {
        lines.push(Line::styled(
            format!(
                " Fenêtre inconnue : ajoutez context_window = … dans [providers.{}] de config.toml.",
                app.provider
            ),
            dim,
        ));
    }
    lines.push(Line::styled(
        " /prompt montre exactement ce qui est envoyé au modèle.",
        dim,
    ));
    lines
}

/// Gauge colour for a fill percentage.
pub fn gauge_color(percent: u64) -> Color {
    match percent {
        0..80 => Color::Green,
        80..95 => Color::Yellow,
        _ => Color::Red,
    }
}

fn shorten(text: &str, max: usize) -> String {
    if display_width(text) <= max {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if used + w + 1 > max {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}
