//! The conversation as display lines, with a per-message cache.
//!
//! Each message is rendered once for the current width (indent, markdown body, status
//! marker) and cached. The app invalidates a message when it changes — during streaming
//! only the last one — and refreshes the cache from `App::update`, so rendering never
//! computes anything.

use ratatui::{
    style::Style,
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

/// Spinner frames. Braille is one cell wide in every font that has the block, and the text
/// beside it carries the meaning for the fonts that do not.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// The assistant's mark, drawn in the two-column gutter that `indent` reserves for a reply.
///
/// This is the one role marker the editorial direction keeps, and it is affordable only
/// because it costs nothing: the margin is already there, so the glyph adds no line and no
/// column. It opens a *turn*, not a message — see [`opens_a_turn`].
///
/// U+2733 is East Asian Width *Neutral*, so it is one cell everywhere. A glyph of *ambiguous*
/// width would be one cell in some terminals and two in others, shunting every reply's text
/// out of alignment; `the_avatar_is_one_cell_wide` is the guard on that.
pub const AVATAR: &str = "✳";

/// What the application is doing while no token has arrived yet.
///
/// `frame` and `elapsed_s` are already reduced from `App`'s tick counter, which is
/// deliberately not stored here: the comparison in [`Transcript::set_waiting`] must see a
/// change about ten times a second, not thirty, or the markdown cap goes with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waiting {
    pub phase: crate::llm::Phase,
    pub frame: usize,
    /// Whole seconds waited; `None` under one second, so a fast reply shows no number.
    pub elapsed_s: Option<u32>,
}

impl Waiting {
    /// The line as the user reads it, without the margin.
    fn line(&self) -> Line<'static> {
        use crate::llm::Phase;
        let dim = Style::default().fg(crate::theme::palette().dim);
        let what = match &self.phase {
            Phase::Connecting => "connexion…".to_owned(),
            Phase::Retrieving { collections: 0 } => "recherche dans les documents…".to_owned(),
            Phase::Retrieving { collections: 1 } => "recherche dans 1 collection…".to_owned(),
            Phase::Retrieving { collections } => {
                format!("recherche dans {collections} collections…")
            }
            Phase::Waiting { model } => format!("{model} réfléchit…"),
            Phase::RunningTool { name } => format!("exécution de {name}…"),
        };
        let glyph = SPINNER[self.frame % SPINNER.len()];
        let mut spans = vec![Span::styled(format!("{glyph} {what}"), dim)];
        if let Some(seconds) = self.elapsed_s {
            spans.push(Span::styled(format!("  {seconds} s"), dim));
        }
        Line::from(spans)
    }
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
    /// What the application is doing while no token has arrived yet; shown in place of the
    /// streaming message's empty body.
    waiting: Option<Waiting>,
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

    /// Sets the waiting line. The streaming message is re-rendered when it changes, which
    /// is what makes the spinner turn: `refresh` bumps `revision`, and `runtime.rs` redraws
    /// on a tick only when `revision` moved.
    pub fn set_waiting(&mut self, waiting: Option<Waiting>) {
        if waiting == self.waiting {
            return;
        }
        self.waiting = waiting;
        // Only the message being streamed shows it.
        if let Some(entry) = self.entries.last() {
            let id = entry.id;
            self.invalidate(id);
        }
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
                let mut body = message_lines_marked(message, width, mark, self.waiting.as_ref());
                if opens_a_turn(message.role, i.checked_sub(1).map(|p| messages[p].role)) {
                    body = with_avatar(body);
                }
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

/// Renders one message: optional label, body, status marker and a separating blank line.
pub fn message_lines(message: &Message, width: usize) -> Vec<Line<'static>> {
    message_lines_marked(message, width, None, None)
}

/// [`message_lines`], with `‹ shown/total ›` as metadata under the body when there are
/// versions, and the waiting line shown in place of an empty streaming body.
pub fn message_lines_marked(
    message: &Message,
    width: usize,
    mark: Option<(usize, usize)>,
    waiting: Option<&Waiting>,
) -> Vec<Line<'static>> {
    let palette = crate::theme::palette();
    let dim = Style::default().fg(palette.dim);
    let columns = indent(message.role);
    // The body is wrapped to what is left once the margin is taken, then shifted into it.
    let inner = width.saturating_sub(columns).max(1);

    let mut lines = Vec::new();
    if let Some(label) = label(message.role) {
        lines.push(Line::styled(label.to_owned(), dim));
    }
    let mut body = match message.role {
        Role::Assistant | Role::Summary => markdown::render(&message.content, inner),
        Role::User => plain(&message.content, inner, dim),
        Role::System => plain(&message.content, inner, dim),
        Role::Attachment => attachment_card(message, inner),
        Role::Tool => tool_card(message, inner, waiting),
    };

    match &message.status {
        MessageStatus::Complete => {}
        // The card owns its own status rendering end to end (that is why `RunningTool` is
        // read on the card instead of as a generic waiting line): nothing is appended under
        // it, neither a second waiting line nor the streaming cursor.
        MessageStatus::Streaming if message.role == Role::Tool => {}
        MessageStatus::Streaming => match waiting {
            // Nothing has arrived yet: say what is being waited for, where the text will go.
            Some(waiting) if message.content.is_empty() => body.push(waiting.line()),
            _ => match body.last_mut() {
                Some(last) if last.width() < inner => last.push_span(Span::styled("▍", dim)),
                _ => body.push(Line::styled("▍", dim)),
            },
        },
        MessageStatus::Cancelled => body.push(Line::styled("[interrompu]", dim.italic())),
        MessageStatus::Failed(error) => {
            let red = Style::default().fg(palette.error);
            let spans = [Span::styled(format!("✖ {error}"), red)];
            body.extend(wrap_spans(&spans, inner).into_iter().map(Line::from));
        }
    }
    lines.extend(body);
    if !message.citations.is_empty() {
        lines.extend(citation_lines(message, inner));
    }
    // Metadata only on demand: the model is named only when there is another version to
    // compare it against. This is where the deleted header's marker went.
    if let Some((shown, total)) = mark {
        let mut spans = Vec::new();
        if message.role == Role::Assistant
            && let Some(model) = &message.source
        {
            spans.push(Span::styled(format!("{model} · "), dim));
        }
        spans.push(Span::styled(
            format!("‹ {shown}/{total} ›  Alt+← Alt+→"),
            dim,
        ));
        lines.push(Line::from(spans));
    }
    let mut lines = shift(lines, columns);
    // The rhythm replaces the frames: one blank line after every message, and one more
    // before a question, so a turn is separated from the next by two and its own halves by
    // one. The conversation therefore opens on a blank line, which is wanted.
    if message.role == Role::User {
        lines.insert(0, Line::default());
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

/// Left margin of each role, in columns.
///
/// This is the layout: role is read from position, not from a label. The reply sits at the
/// left margin, the question is pushed right and dimmed, and the rare roles — which cannot
/// be told apart by position — share the reply's margin and carry a label instead.
fn indent(role: Role) -> usize {
    match role {
        Role::Assistant | Role::Summary | Role::System | Role::Attachment | Role::Tool => 2,
        Role::User => 8,
    }
}

/// The dim label of a role too rare to be read from its position, or `None` when position
/// says it already.
fn label(role: Role) -> Option<&'static str> {
    match role {
        Role::User | Role::Assistant => None,
        Role::System => Some("système"),
        Role::Summary => Some("résumé de la conversation"),
        Role::Attachment => Some("fichier joint"),
        Role::Tool => Some("outil"),
    }
}

/// Shifts every line right by `columns`, leaving blank lines blank so no trailing spaces
/// land in the snapshots.
/// Whether this message opens a turn, and so carries the [`AVATAR`].
///
/// A reply resumed after a tool round — or split across two messages — is still the same turn,
/// so only the first is marked. The avatar says "the assistant is speaking", not "a message
/// begins". This reads the *previous* message's role, which never changes once an entry exists,
/// so it is safe against the per-entry render cache.
fn opens_a_turn(role: Role, previous: Option<Role>) -> bool {
    role == Role::Assistant && !previous.is_some_and(|p| matches!(p, Role::Assistant | Role::Tool))
}

/// Puts the [`AVATAR`] in the left margin of the first drawn line, keeping the text where it
/// was. The margin is pure whitespace produced by [`shift`]; anything else is left alone rather
/// than overwritten.
fn with_avatar(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    let accent = Style::default().fg(crate::theme::palette().accent);
    let Some(line) = lines.iter_mut().find(|l| !l.spans.is_empty()) else {
        return lines;
    };
    let pad = line.spans[0].content.chars().count();
    if !line.spans[0].content.trim().is_empty() || pad < 2 {
        return lines;
    }
    // The avatar plus its separating space occupy the first two columns; a wider margin keeps
    // the remainder, so the text does not move.
    line.spans[0] = Span::raw(" ".repeat(pad - 2));
    line.spans
        .splice(0..0, [Span::styled(format!("{AVATAR} "), accent)]);
    lines
}

fn shift(lines: Vec<Line<'static>>, columns: usize) -> Vec<Line<'static>> {
    let pad = " ".repeat(columns);
    lines
        .into_iter()
        .map(|line| {
            if line.spans.is_empty() {
                return line;
            }
            let style = line.style;
            let mut spans = vec![Span::raw(pad.clone())];
            spans.extend(line.spans);
            Line::from(spans).style(style)
        })
        .collect()
}

/// A tool call is shown as a one-line card: what it is doing, or what it returned.
fn tool_card(message: &Message, width: usize, waiting: Option<&Waiting>) -> Vec<Line<'static>> {
    let palette = crate::theme::palette();
    let what = message.source.as_deref().unwrap_or("outil");
    let running = matches!(
        waiting.map(|w| &w.phase),
        Some(crate::llm::Phase::RunningTool { .. })
    );
    let (text, color) = match &message.status {
        // `Streaming` covers both halves of a tool call: waiting for the user's decision,
        // and running once it is given. Only the phase can tell them apart.
        MessageStatus::Streaming if running => {
            let glyph = SPINNER[waiting.map_or(0, |w| w.frame) % SPINNER.len()];
            (format!("🔧 {what} · {glyph} exécution…"), palette.warn)
        }
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
    // `dim`, like every other piece of structure: it is a separator, not an accent. And on
    // the reply's margin, since it was the only thing on screen touching column 0.
    let style = Style::default().fg(crate::theme::palette().dim);
    let columns = indent(Role::Assistant);
    let inner = width.saturating_sub(columns).max(1);
    let wrapped: Vec<Line<'static>> = wrap_spans(&[Span::styled(text, style)], inner)
        .into_iter()
        .map(Line::from)
        .collect();
    let mut lines = shift(wrapped, columns);
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

/// User text is shown as typed: line breaks kept, words wrapped, in `style`.
fn plain(text: &str, width: usize, style: Style) -> Vec<Line<'static>> {
    text.lines()
        .flat_map(|line| wrap_spans(&[Span::styled(line.to_owned(), style)], width))
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
        // 2 user turns × (leading blank + 1 body line + trailing blank) + 2 replies ×
        // (1 body line + trailing blank)
        assert_eq!(t.total_lines(), 10);
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
        assert_eq!(text(&t.visible(8, 2)), vec!["✳ Il était▍", ""]);

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
        // The marker is its own dim line, under the body (no header left to carry it).
        assert_eq!(text(&t.visible(4, 1)), vec!["  ‹ 2/3 ›  Alt+← Alt+→"]);
        t.set_marks(Vec::new());
        t.refresh(c.messages(), 60, 0);
        assert_eq!(text(&t.visible(3, 1)), vec!["✳ Salut !"]);
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
        assert_eq!(text(&t.visible(0, 3))[1], "        Autre");
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
            vec!["  ▍", "", "  partiel", "  [interrompu]", "", "  ✖ boom", "",]
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
        assert_eq!(lines[1..3], ["        **pas du gras**", "        ligne 2"]);
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
            Some("  ── contexte vidé : les messages au-dessus ne sont plus envoyés au modèle ──"),
            "at the reply's margin, not against the left edge"
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
            .position(|l| l.starts_with("  ── contexte vidé"))
            .expect("shown once");
        assert!(all[separator + 2..].contains(&"        nouveau".to_owned()));
        assert_eq!(
            all.iter()
                .filter(|l| l.starts_with("  ── contexte"))
                .count(),
            1
        );
    }

    #[test]
    fn attachments_are_shown_as_cards() {
        let mut c = Conversation::new();
        c.push_attachment("docs/plan.md", "x".repeat(2_048));
        let lines = text(&message_lines(&c.messages()[0], 80));
        assert_eq!(lines[0], "  fichier joint");
        assert_eq!(lines[1], "  📎 docs/plan.md · 2,0 Ko · ≈ 512 tokens");
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
                .any(|l| l.starts_with("  ── historique résumé"))
        );
    }

    fn user(text: &str) -> Message {
        Message::new(MessageId(1), Role::User, text)
    }

    fn assistant(text: &str) -> Message {
        Message::new(MessageId(2), Role::Assistant, text)
    }

    /// Role is read from position: the question is pushed right and dimmed, the reply sits
    /// at the left margin in the terminal's own foreground. No label on either.
    #[test]
    fn the_role_is_read_from_the_indentation() {
        let question = message_lines(&user("Comment trier un Vec ?"), 60);
        let text: Vec<String> = question.iter().map(ToString::to_string).collect();
        assert!(
            text.iter().all(|l| !l.contains("Vous")),
            "the role label is gone: {text:?}"
        );
        // text[0] is the blank line that opens a question's turn (see
        // `a_turn_is_separated_by_two_blank_lines_and_its_halves_by_one`); the content follows.
        assert!(
            text[1].starts_with("        Comment"),
            "the question sits at column 8: {:?}",
            text[1]
        );

        let reply = message_lines(&assistant("Utilise sort_unstable."), 60);
        let text: Vec<String> = reply.iter().map(ToString::to_string).collect();
        assert!(
            text.iter().all(|l| !l.contains("Assistant")),
            "the role label is gone: {text:?}"
        );
        assert!(
            text[0].starts_with("  Utilise"),
            "the reply sits at column 2: {:?}",
            text[0]
        );
    }

    /// Four roles are too rare to be expressed by position and keep a dim label.
    #[test]
    fn the_rare_roles_keep_a_label() {
        for (role, expected) in [
            (Role::System, "système"),
            (Role::Summary, "résumé de la conversation"),
        ] {
            let message = Message::new(MessageId(3), role, "texte");
            let text = message_lines(&message, 60)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains(expected), "{role:?}: {text}");
        }
    }

    /// The version marker lost its home when the header went away; it lands on its own dim
    /// line under the body, with the model that wrote that version.
    #[test]
    fn the_version_marker_sits_under_the_body() {
        let mut message = assistant("Première réponse.");
        message.source = Some("llama3.2".to_owned());
        let lines = message_lines_marked(&message, 60, Some((2, 3)), None);
        let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
        let marker = text
            .iter()
            .position(|l| l.contains("‹ 2/3 ›"))
            .expect("the marker is shown");
        let body = text
            .iter()
            .position(|l| l.contains("Première"))
            .expect("the body is shown");
        assert!(marker > body, "the marker comes after the body: {text:?}");
        assert!(text[marker].contains("llama3.2"), "{:?}", text[marker]);
    }

    /// Two blank lines between turns, one inside a turn — the rhythm that replaces the
    /// frames. A question carries the extra one, so the rule holds wherever a turn starts.
    #[test]
    fn a_turn_is_separated_by_two_blank_lines_and_its_halves_by_one() {
        let blanks = |lines: &[Line<'static>]| -> (usize, usize) {
            let leading = lines.iter().take_while(|l| l.spans.is_empty()).count();
            let trailing = lines
                .iter()
                .rev()
                .take_while(|l| l.spans.is_empty())
                .count();
            (leading, trailing)
        };
        assert_eq!(blanks(&message_lines(&user("Question ?"), 60)), (1, 1));
        assert_eq!(blanks(&message_lines(&assistant("Réponse."), 60)), (0, 1));
    }

    /// Without versions there is no metadata at all: nothing by default.
    #[test]
    fn without_versions_there_is_no_metadata_line() {
        let mut message = assistant("Réponse.");
        message.source = Some("llama3.2".to_owned());
        let text = message_lines_marked(&message, 60, None, None)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("llama3.2"), "{text}");
        assert!(!text.contains('‹'), "{text}");
    }

    /// Review Focus 1: a pane narrower than the indentation leaves no room for text. It
    /// must not panic, and it must still produce lines.
    #[test]
    fn a_pane_narrower_than_the_indentation_does_not_panic() {
        for width in [0, 1, 2, 8, 9, 20] {
            let lines = message_lines(&user("Comment trier un Vec ?"), width);
            assert!(!lines.is_empty(), "width {width} produced nothing");
            let lines = message_lines(&assistant("# Titre\n\nTexte."), width);
            assert!(!lines.is_empty(), "width {width} produced nothing");
        }
    }

    /// Review Focus 3: a pasted URL has no break point. Wrapped at an indentation, it must
    /// stay inside the pane.
    #[test]
    fn a_long_unbroken_token_stays_inside_the_pane() {
        let url = "https://example.com/".to_owned() + &"a".repeat(200);
        let width = 40;
        for line in message_lines(&user(&url), width) {
            assert!(
                line.width() <= width,
                "line of {} columns in a pane of {width}: {line:?}",
                line.width()
            );
        }
    }

    /// Review Focus 2: the user's messages are dim now, and `dimmed()` repaints every span
    /// dim — so an out-of-context question looks exactly like an in-context one. The
    /// separator is the only remaining distinction and must be drawn.
    #[test]
    fn the_out_of_context_separator_is_the_only_distinction_left() {
        let mut transcript = Transcript::default();
        let messages = vec![user("ancienne question"), assistant("ancienne réponse")];
        // 80 columns: wide enough that the separator sentence does not itself wrap, so the
        // substring check below is not an artifact of where the wrap happens to fall.
        transcript.refresh(&messages, 80, 3);
        let text = transcript.line_texts().join("\n");
        assert!(
            text.contains("ne sont plus envoyés au modèle"),
            "the separator must be drawn: {text}"
        );
    }

    /// The separator is structure, so it is `dim` and sits on the reply's margin. Snapshots
    /// record characters only, so its colour can be checked here and nowhere else.
    #[test]
    fn the_separator_is_dim_and_on_the_reply_margin() {
        let dim = Style::default().fg(crate::theme::palette().dim);
        for summary in [false, true] {
            let lines = boundary(summary, 80);
            let first = lines.first().expect("the separator line");
            assert_eq!(
                first.spans.first().map(|s| s.content.as_ref()),
                Some("  "),
                "the separator starts at column 2 like every reply: {first:?}"
            );
            assert!(
                first.spans[1..].iter().all(|s| s.style == dim),
                "the separator is dim, not the accent: {first:?}"
            );
        }
    }

    use crate::llm::Phase;

    fn streaming(text: &str) -> Message {
        let mut message = Message::new(MessageId(2), Role::Assistant, text.to_owned());
        message.status = MessageStatus::Streaming;
        message
    }

    /// The waiting line stands where the reply will appear, so the text replaces it in
    /// place when the first token lands.
    #[test]
    fn the_waiting_line_stands_where_the_reply_will_appear() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Waiting {
                model: "llama3.2".to_owned(),
            },
            frame: 0,
            elapsed_s: Some(3),
        }));
        transcript.refresh(&[user("Raconte"), streaming("")], 60, 0);
        let text = transcript.line_texts().join("\n");

        assert!(text.contains("llama3.2 réfléchit…"), "{text}");
        assert!(text.contains("3 s"), "{text}");
        let line = transcript
            .line_texts()
            .into_iter()
            .find(|l| l.contains("réfléchit"))
            .expect("the line is there");
        // The gutter now carries the avatar, so pin what this test was always about: the
        // content begins at column 2, exactly where the reply's text will land.
        assert_eq!(
            line.chars().nth(2),
            Some('⠋'),
            "the content starts at column 2: {line:?}"
        );
    }

    /// Once a token has arrived the message is no longer empty, and the waiting line has no
    /// business being there even if the phase was not cleared yet.
    #[test]
    fn a_message_with_content_shows_no_waiting_line() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Waiting {
                model: "llama3.2".to_owned(),
            },
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[user("Raconte"), streaming("Il était")], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(!text.contains("réfléchit"), "{text}");
        assert!(text.contains("Il était"), "{text}");
    }

    /// Under a second, no number: a fast reply must not flash one.
    #[test]
    fn under_a_second_no_duration_is_shown() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Connecting,
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[user("Salut"), streaming("")], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(text.contains("connexion…"), "{text}");
        assert!(!text.contains(" s"), "{text}");
    }

    /// One collection reads differently from several, and zero reads as the documents.
    #[test]
    fn the_retrieval_line_counts_its_collections() {
        for (collections, expected) in [
            (0, "recherche dans les documents…"),
            (1, "recherche dans 1 collection…"),
            (2, "recherche dans 2 collections…"),
        ] {
            let mut transcript = Transcript::default();
            transcript.set_waiting(Some(Waiting {
                phase: Phase::Retrieving { collections },
                frame: 0,
                elapsed_s: None,
            }));
            transcript.refresh(&[user("q"), streaming("")], 60, 0);
            let text = transcript.line_texts().join("\n");
            assert!(text.contains(expected), "{collections}: {text}");
        }
    }

    /// The redraw gate is `revision`, so a frame change must move it — and an unchanged
    /// value must not, or the 30 fps markdown cap is gone.
    #[test]
    fn only_a_real_change_bumps_the_revision() {
        let mut transcript = Transcript::default();
        let messages = [user("q"), streaming("")];
        let waiting = |frame| {
            Some(Waiting {
                phase: Phase::Connecting,
                frame,
                elapsed_s: None,
            })
        };

        transcript.set_waiting(waiting(0));
        transcript.refresh(&messages, 60, 0);
        let first = transcript.revision();

        transcript.set_waiting(waiting(0));
        transcript.refresh(&messages, 60, 0);
        assert_eq!(transcript.revision(), first, "same frame, no redraw");

        transcript.set_waiting(waiting(1));
        transcript.refresh(&messages, 60, 0);
        assert!(transcript.revision() > first, "new frame, redraw");
    }

    /// Review Focus 5: the braille glyph carries no information the text does not. A
    /// terminal that cannot draw it must still show a readable line.
    #[test]
    fn the_waiting_line_reads_without_its_glyph() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::RunningTool {
                name: "read_file".to_owned(),
            },
            frame: 4,
            elapsed_s: Some(2),
        }));
        transcript.refresh(&[user("q"), streaming("")], 60, 0);
        let line = transcript
            .line_texts()
            .into_iter()
            .find(|l| l.contains("read_file"))
            .expect("the line is there");
        let without_glyph: String = line
            .chars()
            .filter(|c| !SPINNER.contains(&&*c.to_string()))
            .collect();
        assert!(
            without_glyph.contains("exécution de read_file"),
            "{without_glyph:?}"
        );
    }

    /// A tool the user already approved is executing, and must stop asking for approval.
    #[test]
    fn the_tool_card_tells_waiting_from_running() {
        let mut message = Message::new(MessageId(4), Role::Tool, String::new());
        message.source = Some("read_file".to_owned());
        message.status = MessageStatus::Streaming;

        let mut transcript = Transcript::default();
        transcript.set_waiting(None);
        transcript.refresh(&[message.clone()], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(text.contains("en attente de votre accord"), "{text}");

        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: crate::llm::Phase::RunningTool {
                name: "read_file".to_owned(),
            },
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[message], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(text.contains("exécution"), "{text}");
        assert!(!text.contains("en attente de votre accord"), "{text}");
    }

    /// Coordinator finding 1: the card owns its status text end to end. The generic
    /// `Streaming` handling (waiting line or cursor) must not add anything under it.
    #[test]
    fn a_running_tool_card_is_exactly_one_line() {
        let mut message = Message::new(MessageId(4), Role::Tool, String::new());
        message.source = Some("read_file".to_owned());
        message.status = MessageStatus::Streaming;
        let waiting = Waiting {
            phase: crate::llm::Phase::RunningTool {
                name: "read_file".to_owned(),
            },
            frame: 0,
            elapsed_s: None,
        };

        let lines = message_lines_marked(&message, 60, None, Some(&waiting));
        let card_lines: Vec<String> = text(&lines)
            .into_iter()
            .filter(|l| l.contains("🔧"))
            .collect();
        assert_eq!(card_lines.len(), 1, "{card_lines:?}");
        assert!(card_lines[0].contains("exécution"), "{card_lines:?}");
    }

    /// An unrelated phase (e.g. retrieval for the next message) must not leak a second,
    /// contradicting line under a card still waiting for approval.
    #[test]
    fn a_tool_card_awaiting_approval_ignores_an_unrelated_phase() {
        let mut message = Message::new(MessageId(4), Role::Tool, String::new());
        message.source = Some("read_file".to_owned());
        message.status = MessageStatus::Streaming;
        let waiting = Waiting {
            phase: crate::llm::Phase::Retrieving { collections: 2 },
            frame: 0,
            elapsed_s: None,
        };

        let lines = message_lines_marked(&message, 60, None, Some(&waiting));
        let all = text(&lines);
        let card_lines: Vec<&String> = all.iter().filter(|l| l.contains("🔧")).collect();
        assert_eq!(card_lines.len(), 1, "{all:?}");
        assert!(
            card_lines[0].contains("en attente de votre accord"),
            "{all:?}"
        );
        assert!(!all.join("\n").contains("recherche dans"), "{all:?}");
    }

    /// The avatar lives in the two-column gutter `indent` already reserves for a reply, so it
    /// costs neither a line nor a column of its own.
    #[test]
    fn the_avatar_opens_a_reply() {
        let mut transcript = Transcript::default();
        transcript.refresh(
            &[user("Comment trier un Vec ?"), assistant("Utilise sort.")],
            60,
            0,
        );
        let line = transcript
            .line_texts()
            .into_iter()
            .find(|l| l.contains("Utilise sort."))
            .expect("the reply is there");
        assert_eq!(
            line.trim_end(),
            format!("{AVATAR} Utilise sort."),
            "avatar at column 0, text still at column 2: {line:?}"
        );
    }

    /// A reply resumed after a tool round is the same turn, so it is not marked again — the
    /// avatar says "the assistant is speaking", not "a message begins".
    #[test]
    fn a_reply_resumed_after_a_tool_has_no_second_avatar() {
        let mut tool = Message::new(MessageId(3), Role::Tool, "lu".to_owned());
        tool.source = Some("read_file".to_owned());
        let messages = [
            Message::new(MessageId(1), Role::User, "Lis x".to_owned()),
            Message::new(MessageId(2), Role::Assistant, "Je regarde.".to_owned()),
            tool,
            Message::new(MessageId(4), Role::Assistant, "Voilà.".to_owned()),
        ];
        let mut transcript = Transcript::default();
        transcript.refresh(&messages, 60, 0);

        let all = transcript.line_texts().join("\n");
        assert_eq!(
            all.matches(AVATAR).count(),
            1,
            "one avatar for the whole turn: {all}"
        );
    }

    #[test]
    fn a_question_never_carries_the_avatar() {
        let mut transcript = Transcript::default();
        transcript.refresh(&[user("Et ça ?")], 60, 0);

        let all = transcript.line_texts().join("\n");
        assert!(!all.contains(AVATAR), "{all}");
    }

    /// The turn has begun as soon as the waiting line appears, so the avatar opens that too.
    #[test]
    fn the_avatar_opens_the_waiting_line_too() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Waiting {
                model: "llama3.2".to_owned(),
            },
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[user("Raconte"), streaming("")], 60, 0);

        let line = transcript
            .line_texts()
            .into_iter()
            .find(|l| l.contains("réfléchit"))
            .expect("the waiting line is there");
        assert!(line.starts_with(&format!("{AVATAR} ")), "{line:?}");
    }

    /// Snapshots capture characters, never colour, so the avatar's tone needs its own test.
    #[test]
    fn the_avatar_is_drawn_in_the_accent_colour() {
        let mut transcript = Transcript::default();
        transcript.refresh(&[assistant("Bonjour.")], 60, 0);

        let line = transcript
            .visible(0, 10)
            .into_iter()
            .find(|l| {
                l.spans
                    .first()
                    .is_some_and(|s| s.content.starts_with(AVATAR))
            })
            .expect("the marked line is there");
        assert_eq!(
            line.spans[0].style.fg,
            Some(crate::theme::palette().accent),
            "{:?}",
            line.spans[0]
        );
    }

    /// The gutter is exactly two columns wide. A glyph of *ambiguous* East Asian width renders
    /// as one cell in some terminals and two in others, which would shunt every reply's text
    /// out of alignment — and snapshots count characters, not cells, so they cannot see it.
    /// This test is the guard on any future change of glyph.
    #[test]
    fn the_avatar_is_one_cell_wide() {
        use unicode_width::UnicodeWidthStr;
        assert_eq!(
            AVATAR.width(),
            1,
            "{AVATAR:?} must occupy exactly one terminal cell"
        );
    }
}
