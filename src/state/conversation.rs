//! Conversation and message model.

/// Stable identifier of a message inside a conversation (used as a render-cache key).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct MessageId(pub u64);

/// Kind of a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    /// A file attached with `/add`: its text is sent as context, not as a turn.
    Attachment,
    /// A summary written by `/compact`: it replaces the messages before it.
    Summary,
    /// A tool the model called (`source`: what it did, `content`: what it returned).
    Tool,
}

/// Lifecycle of a message.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MessageStatus {
    /// Fully written.
    Complete,
    /// Tokens are still arriving.
    Streaming,
    /// Generation was stopped by the user; the content is partial.
    Cancelled,
    /// Generation failed with this user-facing error; the content is partial.
    Failed(String),
}

/// An image attached with `/add`, sent to vision models.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Image {
    /// `image/png`, `image/jpeg`, `image/gif` or `image/webp`.
    pub media_type: String,
    /// The file, base64-encoded.
    pub base64: String,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("media_type", &self.media_type)
            .field("base64_len", &self.base64.len())
            .finish()
    }
}

impl Image {
    /// Size of the file, in bytes.
    pub fn bytes(&self) -> usize {
        self.base64.len() / 4 * 3
    }
}

/// A passage the model was given to write a reply, as listed under it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    /// Number of the source in the prompt (`[3]`), which the model uses to cite it.
    pub number: usize,
    /// File path, relative to the collection root.
    pub path: String,
    /// Where in the file (`p. 3`, `§ Séance 1`, `L12-40`); may be empty.
    pub location: String,
}

impl Citation {
    /// `plan.docx § Séance 1`.
    pub fn label(&self) -> String {
        if self.location.is_empty() {
            self.path.clone()
        } else {
            format!("{} {}", self.path, self.location)
        }
    }
}

/// A single chat message.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub role: Role,
    pub content: String,
    pub status: MessageStatus,
    /// Origin of an attachment (the path as typed).
    pub source: Option<String>,
    /// Passages retrieved for an assistant reply (RAG).
    pub citations: Vec<Citation>,
    /// For an attached image: the image itself (`content` is then empty).
    pub image: Option<Image>,
}

impl Message {
    /// A complete message with no source, citations or image — the shape most tests need,
    /// overriding `status` (and other fields) afterwards when they need something else.
    pub fn new(id: MessageId, role: Role, content: impl Into<String>) -> Self {
        Self {
            id,
            role,
            content: content.into(),
            status: MessageStatus::Complete,
            source: None,
            citations: Vec::new(),
            image: None,
        }
    }
}

/// An ordered list of messages, and where the model's context starts.
///
/// Messages before `context_start` (after `/clear` or `/compact`) stay visible but are no
/// longer sent to the model.
#[derive(Clone, Debug, Default)]
pub struct Conversation {
    messages: Vec<Message>,
    next_id: u64,
    context_start: u64,
}

impl Conversation {
    /// Creates an empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuilds a conversation from stored messages (ids must be increasing).
    pub fn from_messages(messages: Vec<Message>, context_start: u64) -> Self {
        let next_id = messages.last().map_or(0, |m| m.id.0 + 1).max(context_start);
        Self {
            messages,
            next_id,
            context_start,
        }
    }

    /// Appends a message and returns its id.
    pub fn push(
        &mut self,
        role: Role,
        content: impl Into<String>,
        status: MessageStatus,
    ) -> MessageId {
        self.push_message(role, content.into(), status, None)
    }

    /// Appends a file's text as an attachment.
    pub fn push_attachment(
        &mut self,
        source: impl Into<String>,
        content: impl Into<String>,
    ) -> MessageId {
        self.push_message(
            Role::Attachment,
            content.into(),
            MessageStatus::Complete,
            Some(source.into()),
        )
    }

    fn push_message(
        &mut self,
        role: Role,
        content: String,
        status: MessageStatus,
        source: Option<String>,
    ) -> MessageId {
        let id = MessageId(self.next_id);
        self.next_id += 1;
        self.messages.push(Message {
            id,
            role,
            content,
            status,
            source,
            citations: Vec::new(),
            image: None,
        });
        id
    }

    /// Appends an image as an attachment.
    pub fn push_image(&mut self, source: impl Into<String>, image: Image) -> MessageId {
        let id = self.push_message(
            Role::Attachment,
            String::new(),
            MessageStatus::Complete,
            Some(source.into()),
        );
        if let Some(message) = self.messages.last_mut() {
            message.image = Some(image);
        }
        id
    }

    /// All messages, oldest first.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Messages still sent to the model (from the context start on).
    pub fn context_messages(&self) -> &[Message] {
        let first = self
            .messages
            .partition_point(|m| m.id.0 < self.context_start);
        &self.messages[first..]
    }

    /// Id of the first message in the context.
    pub fn context_start(&self) -> u64 {
        self.context_start
    }

    /// Excludes every message before `id` from the context.
    pub fn set_context_start(&mut self, id: u64) {
        self.context_start = id;
    }

    /// Id the next message will get.
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Appends a tool call waiting for its result (`description`: what it does).
    pub fn push_tool(&mut self, description: impl Into<String>) -> MessageId {
        self.push_message(
            Role::Tool,
            String::new(),
            MessageStatus::Streaming,
            Some(description.into()),
        )
    }

    /// Appends messages that keep their ids (a restored version); ids must be greater
    /// than those of the messages already there.
    pub fn append(&mut self, messages: Vec<Message>) {
        if let Some(last) = messages.last() {
            self.next_id = self.next_id.max(last.id.0 + 1);
        }
        self.messages.extend(messages);
    }

    /// The messages after `after` (all of them for `None`).
    pub fn after(&self, after: Option<MessageId>) -> &[Message] {
        let first = match after {
            Some(id) => self.messages.partition_point(|m| m.id <= id),
            None => 0,
        };
        &self.messages[first..]
    }

    /// Id of the message just before `id`.
    pub fn before(&self, id: MessageId) -> Option<MessageId> {
        self.messages.iter().rev().find(|m| m.id < id).map(|m| m.id)
    }

    /// Removes the message `from` and every message after it.
    pub fn truncate(&mut self, from: MessageId) {
        self.messages.retain(|m| m.id < from);
    }

    /// Read-only access to a message by id.
    pub fn get(&self, id: MessageId) -> Option<&Message> {
        // Ids are increasing, so the message is found by binary search.
        let index = self.messages.binary_search_by_key(&id, |m| m.id).ok()?;
        self.messages.get(index)
    }

    /// Mutable access to a message by id.
    pub fn get_mut(&mut self, id: MessageId) -> Option<&mut Message> {
        // Ids are increasing, so the message is found by binary search.
        let index = self.messages.binary_search_by_key(&id, |m| m.id).ok()?;
        self.messages.get_mut(index)
    }

    /// `true` when the conversation has no message.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_assigns_increasing_ids() {
        let mut conversation = Conversation::new();
        let a = conversation.push(Role::User, "hi", MessageStatus::Complete);
        let b = conversation.push(Role::Assistant, "hello", MessageStatus::Complete);
        assert!(a < b);
        assert_eq!(conversation.messages().len(), 2);
        assert_eq!(conversation.messages()[1].content, "hello");
    }

    #[test]
    fn restored_conversation_continues_numbering() {
        let mut original = Conversation::new();
        original.push(Role::User, "a", MessageStatus::Complete);
        original.push(Role::Assistant, "b", MessageStatus::Complete);
        let mut restored = Conversation::from_messages(original.messages().to_vec(), 0);
        let id = restored.push(Role::User, "c", MessageStatus::Complete);
        assert_eq!(id, MessageId(2));
    }

    #[test]
    fn get_mut_finds_message_by_id() {
        let mut conversation = Conversation::new();
        conversation.push(Role::User, "a", MessageStatus::Complete);
        let id = conversation.push(Role::Assistant, "", MessageStatus::Streaming);
        if let Some(message) = conversation.get_mut(id) {
            message.content.push_str("hi");
        }
        assert_eq!(conversation.messages()[1].content, "hi");
        assert!(conversation.get_mut(MessageId(99)).is_none());
        assert_eq!(conversation.get(id).map(|m| m.content.as_str()), Some("hi"));
        assert!(conversation.get(MessageId(99)).is_none());
    }

    #[test]
    fn context_start_hides_older_messages() {
        let mut conversation = Conversation::new();
        conversation.push(Role::User, "old", MessageStatus::Complete);
        conversation.push(Role::Assistant, "old reply", MessageStatus::Complete);
        assert_eq!(conversation.context_messages().len(), 2);

        conversation.set_context_start(conversation.next_id());
        assert!(conversation.context_messages().is_empty());
        let id = conversation.push_attachment("notes.md", "contenu");
        assert_eq!(conversation.context_messages().len(), 1);
        assert_eq!(conversation.context_messages()[0].id, id);
        assert_eq!(
            conversation.context_messages()[0].source.as_deref(),
            Some("notes.md")
        );
        assert_eq!(conversation.messages().len(), 3, "still displayed");
    }

    #[test]
    fn restoring_after_a_clear_keeps_numbering_past_the_boundary() {
        let mut restored = Conversation::from_messages(Vec::new(), 5);
        assert_eq!(
            restored.push(Role::User, "x", MessageStatus::Complete),
            MessageId(5)
        );
    }
}
