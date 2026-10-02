//! The find bar (Ctrl+F), drawn in place of the input box.

use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Padding, Paragraph},
};

use crate::state::Find;

/// Draws the query and the match count in `area`.
pub fn render(find: &Find, frame: &mut Frame, area: Rect) {
    let palette = crate::theme::palette();
    let dim = Style::default().fg(palette.dim);
    let count = if find.query.trim().is_empty() {
        Span::styled("tapez le texte à chercher", dim)
    } else if find.matches.is_empty() {
        Span::styled("aucun résultat", Style::default().fg(palette.warn))
    } else {
        Span::styled(
            format!("{}/{}", find.current + 1, find.matches.len()),
            Style::default()
                .fg(palette.accent)
                .add_modifier(Modifier::BOLD),
        )
    };
    let line = Line::from(vec![
        Span::styled("⌕ ", Style::default().fg(palette.accent)),
        Span::raw(find.query.clone()),
        Span::styled("▍", dim),
        Span::raw("   "),
        count,
    ]);
    // The same chrome as the input it replaces — one rule, no box — so taking the slot does
    // not redraw its frame. The accent colour on the rule is what says the mode changed.
    let block = Block::new()
        .borders(Borders::TOP)
        .padding(Padding::new(1, 1, 0, 1))
        .border_style(Style::default().fg(palette.accent))
        .title("── Chercher dans la conversation ");
    frame.render_widget(Paragraph::new(line).block(block), area);
}
