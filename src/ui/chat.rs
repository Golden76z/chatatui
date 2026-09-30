//! Conversation area: shows the visible slice of the cached transcript.

use ratatui::{
    Frame,
    layout::{Constraint, Flex, Layout, Rect},
    style::{Style, Stylize},
    text::{Line, Text},
    widgets::Paragraph,
};

use crate::{app::App, layout};

/// Draws the conversation, or a welcome screen when it is empty.
pub fn render(app: &App, frame: &mut Frame, area: Rect) {
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
