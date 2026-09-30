//! The conversation as display lines, with a per-message cache.
//!
//! Each message is rendered once for the current width (header, markdown body, status
//! marker) and cached. The app invalidates a message when it changes — during streaming
//! only the last one — and refreshes the cache from `App::update`, so rendering never
//! computes anything.

use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::{
    markdown::{self, wrap_spans},
    state::{Message, MessageId, MessageStatus, Role},
    tokens,
};

#[derive(Debug)]
struct Entry {
    id: MessageId,
    dirty: bool,
    lines: Vec<Line<'static>>,
}

/// Cached display lines of a conversation.
#[derive(Debug, Default)]
pub struct Transcript {
    width: usize,
    /// Context start the cache was rendered for (see `Conversation::context_start`).
    context_start: u64,
    entries: Vec<Entry>,
    /// Separator shown after the last message when the whole conversation was cleared.
    trailer: Vec<Line<'static>>,
    total_lines: usize,
    /// Incremented whenever the lines change; lets the runtime skip useless redraws.
    revision: u64,
    /// Messages rendered during the last refresh (for tests and diagnostics).
    last_rendered: usize,
    /// Version markers (`‹ 2/3 ›`): message, shown version, number of versions.
    marks: Vec<(MessageId, usize, usize)>,
}

impl Transcript {
    /// Sets the version markers; messages whose marker changed are rendered again.
    pub fn set_marks(&mut self, marks: Vec<(MessageId, usize, usize)>) {
        if marks == self.marks {
            return;
        }
        let changed: Vec<MessageId> = marks
            .iter()
            .filter(|m| !self.marks.contains(m))
            .chain(self.marks.iter().filter(|m| !marks.contains(m)))
            .map(|m| m.0)
            .collect();
        for id in changed {
            self.invalidate(id);
        }
        self.marks = marks;
    }

    /// Marks a message as changed; it is re-rendered on the next refresh.
    pub fn invalidate(&mut self, id: MessageId) {
        if let Some(entry) = self.entries.iter_mut().rev().find(|e| e.id == id) {
            entry.dirty = true;
        }
    }

    /// Forgets everything. Must be called when another conversation is loaded, since
    /// message ids are only unique within one conversation.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.trailer.clear();
        self.total_lines = 0;
        self.revision += 1;
    }

    /// Brings the cache in line with `messages` rendered at `width`, messages before
    /// `context_start` being shown as out of context. Returns `true` if any line changed.
    pub fn refresh(&mut self, messages: &[Message], width: usize, context_start: u64) -> bool {
        let width = width.max(1);
        if width != self.width || context_start != self.context_start {
            self.width = width;
            self.context_start = context_start;
            self.entries.iter_mut().for_each(|e| e.dirty = true);
        }
        // Entries are kept in message order; replace any that do not match.
        self.entries.truncate(messages.len());
        for (i, message) in messages.iter().enumerate() {
            match self.entries.get_mut(i) {
                Some(entry) if entry.id == message.id => {}
                Some(entry) => {
                    *entry = Entry {
                        id: message.id,
                        dirty: true,
                        lines: Vec::new(),
                    };
                }
                None => self.entries.push(Entry {
                    id: message.id,
                    dirty: true,
                    lines: Vec::new(),
                }),
            }
        }

        // The boundary is drawn before the first message in context, if older ones exist.
        let first_in_context = messages
            .iter()
            .position(|m| m.id.0 >= context_start)
            .filter(|&i| i > 0);
        let mut rendered = 0;
        for (i, (entry, message)) in self.entries.iter_mut().zip(messages).enumerate() {
            if entry.dirty {
                let in_context = message.id.0 >= context_start;
                let mut lines = Vec::new();
                if first_in_context == Some(i) {
                    lines.extend(boundary(message.role == Role::Summary, width));
                }
                let mark = self
                    .marks
                    .iter()
                    .find(|m| m.0 == message.id)
                    .map(|m| (m.1, m.2));
                let body = message_lines_marked(message, width, mark);
                if in_context {
                    lines.extend(body);
                } else {
                    lines.extend(body.into_iter().map(dimmed));
                }
                entry.lines = lines;
                entry.dirty = false;
                rendered += 1;
            }
        }
        let cleared_all = context_start > 0
            && !messages.is_empty()
            && messages.iter().all(|m| m.id.0 < context_start);
        let trailer = if cleared_all {
            boundary(false, width)
        } else {
            Vec::new()
        };
        let trailer_changed = trailer != self.trailer;
        self.trailer = trailer;
        self.last_rendered = rendered;
        let total = self.entries.iter().map(|e| e.lines.len()).sum::<usize>() + self.trailer.len();
        let changed = rendered > 0 || trailer_changed || total != self.total_lines;
        self.total_lines = total;
        if changed {
            self.revision += 1;
        }
        changed
    }

    /// Number of display lines.
    pub fn total_lines(&self) -> usize {
        self.total_lines
    }

    /// Changes whenever the lines change.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Number of messages rendered by the last refresh.
    pub fn last_rendered(&self) -> usize {
        self.last_rendered
    }

    /// Text of every display line, in order.
    pub fn line_texts(&self) -> Vec<String> {
        self.entries
            .iter()
            .flat_map(|e| e.lines.iter())
            .chain(self.trailer.iter())
            .map(ToString::to_string)
            .collect()
    }

    /// First display line of message `id`, once rendered.
    pub fn line_of(&self, id: MessageId) -> Option<usize> {
        let mut line = 0;
        for entry in &self.entries {
            if entry.id == id {
                return Some(line);
            }
            line += entry.lines.len();
        }
        None
    }

    /// The `height` lines starting at `offset`.
    pub fn visible(&self, offset: usize, height: usize) -> Vec<Line<'static>> {
        self.entries
            .iter()
            .flat_map(|e| e.lines.iter())
            .chain(self.trailer.iter())
            .skip(offset)
            .take(height)
            .cloned()
            .collect()
    }
}

/// Renders one message: header, body, status marker and a separating blank line.
pub fn message_lines(message: &Message, width: usize) -> Vec<Line<'static>> {
    message_lines_marked(message, width, None)
}

/// [`message_lines`], with `‹ shown/total ›` in the header when there are versions.
pub fn message_lines_marked(
    message: &Message,
    width: usize,
    mark: Option<(usize, usize)>,
) -> Vec<Line<'static>> {
    let mut header = header(message.role);
    if let Some((shown, total)) = mark {
        // With several versions, each reply says which model wrote it.
        if message.role == Role::Assistant
            && let Some(model) = &message.source
        {
            header.push_span(Span::styled(
                format!(" · {model}"),
                Style::default().fg(crate::theme::palette().dim),
            ));
        }
        header.push_span(Span::styled(
            format!("  ‹ {shown}/{total} ›  Alt+← Alt+→"),
            Style::default().fg(crate::theme::palette().dim),
        ));
    }
    let mut lines = vec![header];
    let mut body = match message.role {
        Role::Assistant | Role::Summary => markdown::render(&message.content, width),
        Role::User | Role::System => plain(&message.content, width),
        Role::Attachment => attachment_card(message, width),
        Role::Tool => tool_card(message, width),
    };

    let dim = Style::default().fg(crate::theme::palette().dim);
    match &message.status {
        MessageStatus::Complete => {}
        MessageStatus::Streaming => match body.last_mut() {
            Some(last) if last.width() < width => last.push_span(Span::styled("▍", dim)),
            _ => body.push(Line::styled("▍", dim)),
        },
        MessageStatus::Cancelled => body.push(Line::styled("[interrompu]", dim.italic())),
        MessageStatus::Failed(error) => {
            let red = Style::default().fg(crate::theme::palette().error);
            let spans = [Span::styled(format!("✖ {error}"), red)];
            body.extend(wrap_spans(&spans, width).into_iter().map(Line::from));
        }
    }
    lines.extend(body);
    if !message.citations.is_empty() {
        lines.extend(citation_lines(message, width));
    }
    lines.push(Line::default());
    lines
}

/// `Sources : [1] cours.pdf p. 3 · [2] plan.docx § Séance 1`, wrapped.
fn citation_lines(message: &Message, width: usize) -> Vec<Line<'static>> {
    let dim = Style::default().fg(crate::theme::palette().dim);
    let mut spans = vec![Span::styled("Sources : ", dim)];
    for (i, citation) in message.citations.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        spans.push(Span::styled(
            format!("[{}] ", citation.number),
            Style::default().fg(crate::theme::palette().info),
        ));
        spans.push(Span::styled(citation.label(), dim));
    }
    wrap_spans(&spans, width)
        .into_iter()
        .map(Line::from)
        .collect()
}

fn header(role: Role) -> Line<'static> {
    let (label, color) = match role {
        Role::System => ("Système", crate::theme::palette().dim),
        Role::User => ("Vous", crate::theme::palette().accent),
        Role::Assistant => ("Assistant", crate::theme::palette().assistant),
        Role::Attachment => ("Fichier joint", crate::theme::palette().warn),
        Role::Summary => ("Résumé de la conversation", crate::theme::palette().info),
        Role::Tool => ("Outil", crate::theme::palette().warn),
    };
    Line::styled(
        format!("▌ {label}"),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

/// A tool call is shown as a one-line card: what it did and how much it returned.
fn tool_card(message: &Message, width: usize) -> Vec<Line<'static>> {
    let palette = crate::theme::palette();
    let what = message.source.as_deref().unwrap_or("outil");
    let (text, color) = match &message.status {
        MessageStatus::Streaming => (
            format!("🔧 {what} · en attente de votre accord…"),
            palette.warn,
        ),
        MessageStatus::Complete => (
            format!(
                "🔧 {what} · ≈ {} tokens transmis",
                tokens::format_count(tokens::estimate(&message.content))
            ),
            palette.warn,
        ),
        MessageStatus::Failed(reason) => (format!("🔧 {what} · {reason}"), palette.dim),
        MessageStatus::Cancelled => (format!("🔧 {what} · annulé"), palette.dim),
    };
    wrap_spans(&[Span::styled(text, Style::default().fg(color))], width)
        .into_iter()
        .map(Line::from)
        .collect()
}

/// An attachment is shown as a one-line card, not its whole text.
fn attachment_card(message: &Message, width: usize) -> Vec<Line<'static>> {
    let source = message.source.as_deref().unwrap_or("fichier");
    let text = match &message.image {
        Some(image) => format!(
            "🖼 {source} · {} · image ({})",
            format_size(image.bytes()),
            image.media_type.trim_start_matches("image/")
        ),
        None => format!(
            "📎 {source} · {} · ≈ {} tokens",
            format_size(message.content.len()),
            tokens::format_count(tokens::estimate(&message.content))
        ),
    };
    wrap_spans(
        &[Span::styled(
            text,
            Style::default().fg(crate::theme::palette().warn),
        )],
        width,
    )
    .into_iter()
    .map(Line::from)
    .collect()
}

/// `812 o`, `12,3 Ko`, `1,2 Mo`.
pub fn format_size(bytes: usize) -> String {
    match bytes {
        0..1_024 => format!("{bytes} o"),
        1_024..1_048_576 => format!("{:.1} Ko", bytes as f64 / 1_024.0).replace('.', ","),
        _ => format!("{:.1} Mo", bytes as f64 / 1_048_576.0).replace('.', ","),
    }
}

/// Separator between the messages out of context and those still sent.
fn boundary(summary: bool, width: usize) -> Vec<Line<'static>> {
    let text = if summary {
        "── historique résumé : les messages au-dessus ne sont plus envoyés au modèle ──"
    } else {
        "── contexte vidé : les messages au-dessus ne sont plus envoyés au modèle ──"
    };
    let style = Style::default().fg(crate::theme::palette().info);
    let mut lines: Vec<Line<'static>> = wrap_spans(&[Span::styled(text, style)], width)
        .into_iter()
        .map(Line::from)
        .collect();
    lines.push(Line::default());
    lines
}

/// Greys out a line of a message that is no longer in the context.
fn dimmed(line: Line<'static>) -> Line<'static> {
    let spans = line
        .spans
        .into_iter()
        .map(|span| span.style(Style::default().fg(crate::theme::palette().dim)))
        .collect::<Vec<_>>();
    Line::from(spans)
}

/// User text is shown as typed: line breaks kept, words wrapped.
fn plain(text: &str, width: usize) -> Vec<Line<'static>> {
    text.lines()
        .flat_map(|line| wrap_spans(&[Span::raw(line.to_owned())], width))
        .map(Line::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Conversation;

    fn conversation() -> Conversation {
        let mut c = Conversation::new();
        c.push(Role::User, "Bonjour", MessageStatus::Complete);
        c.push(Role::Assistant, "**Salut** !", MessageStatus::Complete);
        c.push(Role::User, "Encore", MessageStatus::Complete);
        c.push(Role::Assistant, "Il", MessageStatus::Streaming);
        c
    }

    fn text(lines: &[Line<'static>]) -> Vec<String> {
        lines.iter().map(|l| l.to_string()).collect()
    }

    #[test]
    fn first_refresh_renders_everything() {
        let c = conversation();
        let mut t = Transcript::default();
        assert!(t.refresh(c.messages(), 40, 0));
        assert_eq!(t.last_rendered(), 4);
        // 4 messages × (header + 1 body line + blank)
        assert_eq!(t.total_lines(), 12);
    }

    #[test]
    fn only_invalidated_messages_are_rendered_again() {
        let mut c = conversation();
        let mut t = Transcript::default();
        t.refresh(c.messages(), 40, 0);

        let last = c.messages()[3].id;
        if let Some(m) = c.get_mut(last) {
            m.content.push_str(" était");
        }
        t.invalidate(last);
        assert!(t.refresh(c.messages(), 40, 0));
        assert_eq!(t.last_rendered(), 1);
        assert_eq!(text(&t.visible(9, 2)), vec!["▌ Assistant", "Il était▍"]);

        assert!(!t.refresh(c.messages(), 40, 0), "nothing changed");
        assert_eq!(t.last_rendered(), 0);
    }

    #[test]
    fn version_marks_are_shown_in_the_header() {
        let c = conversation();
        let mut t = Transcript::default();
        t.refresh(c.messages(), 60, 0);
        let reply = c.messages()[1].id;
        t.set_marks(vec![(reply, 2, 3)]);
        assert!(t.refresh(c.messages(), 60, 0));
        assert_eq!(t.last_rendered(), 1, "only the marked message");
        assert_eq!(
            text(&t.visible(3, 1)),
            vec!["▌ Assistant  ‹ 2/3 ›  Alt+← Alt+→"]
        );
        t.set_marks(Vec::new());
        t.refresh(c.messages(), 60, 0);
        assert_eq!(text(&t.visible(3, 1)), vec!["▌ Assistant"]);
    }

    #[test]
    fn width_change_renders_everything_again() {
        let c = conversation();
        let mut t = Transcript::default();
        t.refresh(c.messages(), 40, 0);
        t.refresh(c.messages(), 30, 0);
        assert_eq!(t.last_rendered(), 4);
    }

    #[test]
    fn cleared_cache_shows_the_new_conversation() {
        // Message ids are only unique within a conversation, so switching conversations
        // must clear the cache.
        let mut t = Transcript::default();
        t.refresh(conversation().messages(), 40, 0);
        let mut other = Conversation::new();
        other.push(Role::User, "Autre", MessageStatus::Complete);
        t.clear();
        t.refresh(other.messages(), 40, 0);
        assert_eq!(t.total_lines(), 3);
        assert_eq!(text(&t.visible(0, 3))[1], "Autre");
    }

    #[test]
    fn status_markers() {
        let mut c = Conversation::new();
        c.push(Role::Assistant, "", MessageStatus::Streaming);
        c.push(Role::Assistant, "partiel", MessageStatus::Cancelled);
        c.push(Role::Assistant, "", MessageStatus::Failed("boom".into()));
        let lines: Vec<String> = c
            .messages()
            .iter()
            .flat_map(|m| text(&message_lines(m, 40)))
            .collect();
        assert_eq!(
            lines,
            vec![
                "▌ Assistant",
                "▍",
                "",
                "▌ Assistant",
                "partiel",
                "[interrompu]",
                "",
                "▌ Assistant",
                "✖ boom",
                "",
            ]
        );
    }

    #[test]
    fn user_text_keeps_line_breaks_and_markdown_symbols() {
        let mut c = Conversation::new();
        c.push(
            Role::User,
            "**pas du gras**\nligne 2",
            MessageStatus::Complete,
        );
        let lines = text(&message_lines(&c.messages()[0], 40));
        assert_eq!(lines[1..3], ["**pas du gras**", "ligne 2"]);
    }

    #[test]
    fn cleared_messages_are_dimmed_below_a_separator() {
        let mut c = Conversation::new();
        c.push(Role::User, "ancien", MessageStatus::Complete);
        c.push(Role::Assistant, "réponse", MessageStatus::Complete);
        let start = c.next_id();
        let mut t = Transcript::default();
        t.refresh(c.messages(), 90, start);
        let lines = t.visible(0, 20);
        let last = text(&lines).into_iter().rev().find(|l| !l.is_empty());
        assert_eq!(
            last.as_deref(),
            Some("── contexte vidé : les messages au-dessus ne sont plus envoyés au modèle ──")
        );
        assert!(
            lines[1]
                .spans
                .iter()
                .all(|s| s.style.fg == Some(crate::theme::palette().dim)),
            "old messages are greyed out"
        );

        c.push(Role::User, "nouveau", MessageStatus::Complete);
        t.refresh(c.messages(), 90, start);
        let all = text(&t.visible(0, 20));
        let separator = all
            .iter()
            .position(|l| l.starts_with("── contexte vidé"))
            .expect("shown once");
        assert!(all[separator + 2..].contains(&"nouveau".to_owned()));
        assert_eq!(
            all.iter().filter(|l| l.starts_with("── contexte")).count(),
            1
        );
    }

    #[test]
    fn attachments_are_shown_as_cards() {
        let mut c = Conversation::new();
        c.push_attachment("docs/plan.md", "x".repeat(2_048));
        let lines = text(&message_lines(&c.messages()[0], 80));
        assert_eq!(lines[0], "▌ Fichier joint");
        assert_eq!(lines[1], "📎 docs/plan.md · 2,0 Ko · ≈ 512 tokens");
    }

    #[test]
    fn summary_boundary_has_its_own_wording() {
        let mut c = Conversation::new();
        c.push(Role::User, "ancien", MessageStatus::Complete);
        let id = c.push(Role::Summary, "- résumé", MessageStatus::Complete);
        let mut t = Transcript::default();
        t.refresh(c.messages(), 90, id.0);
        assert!(
            text(&t.visible(0, 20))
                .iter()
                .any(|l| l.starts_with("── historique résumé"))
        );
    }
}
