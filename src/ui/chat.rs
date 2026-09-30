//! Conversation area: shows the visible slice of the cached transcript.

use ratatui::{
    Frame,
    layout::{Constraint, Flex, Layout, Rect},
    style::{Style, Stylize},
    text::{Line, Span, Text},
    widgets::Paragraph,
};

use crate::{app::App, layout};

/// Draws the conversation, or a welcome screen when it is empty.
pub fn render(app: &App, frame: &mut Frame, area: Rect) {
    if let Some(preview) = &app.preview {
        render_preview(app, preview, frame, area);
        return;
    }
    if app.conversation.is_empty() {
        render_welcome(app, frame, area);
        return;
    }

    let content = layout::chat_content(area);
    let height = usize::from(content.height);
    let offset = app.scroll_offset();
    let lines = app.transcript.visible(offset, height);
    frame.render_widget(Paragraph::new(Text::from(lines)), content);

    // Hint shown while scrolled away from the latest content.
    let total = app.transcript.total_lines();
    if offset + height < total {
        let hint = Line::from(" ↓ Ctrl+Fin ").style(
            Style::default()
                .fg(crate::theme::palette().badge_fg)
                .bg(crate::theme::palette().badge_bg),
        );
        let width = u16::try_from(hint.width()).unwrap_or(u16::MAX);
        let [_, hint_area] = Layout::horizontal([Constraint::Fill(1), Constraint::Length(width)])
            .areas(Rect {
                y: area.bottom().saturating_sub(1),
                height: 1.min(area.height),
                ..area
            });
        frame.render_widget(hint, hint_area);
    }
}

/// The conversation highlighted in the list, under a line saying it is only a preview.
fn render_preview(app: &App, preview: &crate::app::Preview, frame: &mut Frame, area: Rect) {
    let [header, body] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
    let palette = crate::theme::palette();
    let title = super::sidebar::truncate(&preview.title, usize::from(area.width) / 2);
    let line = Line::from(vec![
        Span::raw(" Aperçu · "),
        Span::styled(title, Style::default().bold()),
        Span::raw(" · Entrée ouvre · Échap revient "),
    ])
    .style(Style::default().fg(palette.badge_fg).bg(palette.badge_bg));
    frame.render_widget(line, header);

    let content = layout::chat_content(body);
    let height = usize::from(content.height);
    let lines = preview.transcript.visible(app.preview_offset(), height);
    frame.render_widget(Paragraph::new(Text::from(lines)), content);
}

fn render_welcome(app: &App, frame: &mut Frame, area: Rect) {
    let text = Text::from(vec![
        Line::from("chatatui".bold()),
        Line::from(format!("modèle : {}", app.model_display())).dark_gray(),
        Line::default(),
        Line::from("Écrivez un message ci-dessous pour commencer.").dark_gray(),
        Line::from("Tapez / pour les commandes, Ctrl+P pour la palette, F1 pour l'aide.")
            .dark_gray(),
    ])
    .centered();
    let height = u16::try_from(text.height()).unwrap_or(u16::MAX);
    let [middle] = Layout::vertical([Constraint::Length(height)])
        .flex(Flex::Center)
        .areas(area);
    frame.render_widget(Paragraph::new(text), middle);
}
