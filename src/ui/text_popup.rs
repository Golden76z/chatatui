//! Scrollable read-only popups: help (F1), context details (/context), prompt (/prompt).
//!
//! Content is built as already-wrapped lines for the popup width, so the number of lines
//! (and hence the scroll limit) is known exactly — also from `App::update`, through
//! [`max_scroll`].

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::Line,
    widgets::{Block, BorderType, Clear, Paragraph},
};

use crate::{app::App, state::Overlay};

use super::{context_view, help, prompt_view};

/// A text popup's content.
struct TextPopup {
    title: &'static str,
    lines: Vec<Line<'static>>,
}

/// Preferred width of each popup.
fn preferred_width(overlay: &Overlay) -> u16 {
    match overlay {
        Overlay::Prompt { .. } => 100,
        _ => 76,
    }
}

/// Popup rectangle in `area` for `lines` lines of content.
fn rect(area: Rect, width: u16, lines: usize) -> Rect {
    let width = width.min(area.width.saturating_sub(2));
    let height = u16::try_from(lines)
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .min(area.height.saturating_sub(2));
    super::centered(area, width, height)
}

fn content(app: &App, overlay: &Overlay, inner_width: usize) -> Option<TextPopup> {
    match overlay {
        Overlay::Help { .. } => Some(TextPopup {
            title: " Aide ",
            lines: help::lines(app),
        }),
        Overlay::Context { .. } => Some(TextPopup {
            title: " Contexte ",
            lines: context_view::lines(app, inner_width),
        }),
        Overlay::Prompt { .. } => Some(TextPopup {
            title: " Prompt envoyé au modèle ",
            lines: prompt_view::lines(app, inner_width),
        }),
        Overlay::ModelPicker(_) | Overlay::Palette(_) => None,
    }
}

/// Inner width of the popup for the current terminal size.
fn inner_width(area: Rect, overlay: &Overlay) -> usize {
    usize::from(
        preferred_width(overlay)
            .min(area.width.saturating_sub(2))
            .saturating_sub(2),
    )
}

/// Largest useful scroll offset of the open text popup (0 when none is open).
pub fn max_scroll(app: &App) -> u16 {
    let Some(overlay) = &app.overlay else {
        return 0;
    };
    let area = app.viewport;
    let Some(popup) = content(app, overlay, inner_width(area, overlay)) else {
        return 0;
    };
    let visible = rect(area, preferred_width(overlay), popup.lines.len())
        .height
        .saturating_sub(2);
    u16::try_from(popup.lines.len())
        .unwrap_or(u16::MAX)
        .saturating_sub(visible)
}

/// Draws the open text popup, if any.
pub fn render(app: &App, frame: &mut Frame, area: Rect) {
    let Some(overlay) = &app.overlay else {
        return;
    };
    let scroll = match overlay {
        Overlay::Help { scroll } | Overlay::Context { scroll } | Overlay::Prompt { scroll } => {
            *scroll
        }
        Overlay::ModelPicker(_) | Overlay::Palette(_) => return,
    };
    let Some(popup) = content(app, overlay, inner_width(area, overlay)) else {
        return;
    };
    let total = popup.lines.len();
    let popup_area = rect(area, preferred_width(overlay), total);
    frame.render_widget(Clear, popup_area);

    let visible = popup_area.height.saturating_sub(2);
    let max = u16::try_from(total)
        .unwrap_or(u16::MAX)
        .saturating_sub(visible);
    let scroll = scroll.min(max);
    let position = if max > 0 {
        format!(
            " {}/{} · ",
            usize::from(scroll) + usize::from(visible).min(total),
            total
        )
    } else {
        " ".to_owned()
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
        .title(popup.title)
        .title_bottom(Line::from(format!("{position}Échap fermer ")).right_aligned());
    frame.render_widget(
        Paragraph::new(popup.lines).block(block).scroll((scroll, 0)),
        popup_area,
    );
}
