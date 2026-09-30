//! Builds the message list sent to the model.
//!
//! The system prompt, the conversation summary (`/compact`), attached files (`/add`) and
//! the context provider's chunks (RAG) are merged into a single leading system message:
//! several chat templates (Mistral, Gemma, …) reject more than one system message or a
//! system message that is not first. Attached files and retrieved chunks take the same
//! path, as numbered sources.

use crate::{
    context::{Context, ContextChunk},
    llm::{ChatMessage, ChatRole},
    state::{Message, MessageStatus, Role},
};

/// Assembles the request from the messages still in the context (see
/// `Conversation::context_messages`).
///
/// Failed, streaming and empty messages are skipped; cancelled replies are kept because
/// the user saw them.
pub fn build_messages(
    system_prompt: &str,
    context: &Context,
    history: &[Message],
) -> Vec<ChatMessage> {
    let usable = |m: &&Message| {
        !matches!(
            m.status,
            MessageStatus::Failed(_) | MessageStatus::Streaming
        ) && !m.content.trim().is_empty()
    };
    let summary = history
        .iter()
        .rev()
        .filter(|m| m.role == Role::Summary && m.status == MessageStatus::Complete)
        .find(usable);
    let mut chunks: Vec<ContextChunk> = history
        .iter()
        .filter(|m| is_sent_attachment(m))
        .map(|m| ContextChunk {
            source: m.source.clone().unwrap_or_else(|| "fichier joint".into()),
            location: String::new(),
            text: m.content.clone(),
        })
        .collect();
    chunks.extend(context.chunks.iter().cloned());

    let mut messages = Vec::with_capacity(history.len() + 1);
    let system = system_message(system_prompt, summary.map(|m| m.content.as_str()), &chunks);
    if !system.is_empty() {
        messages.push(ChatMessage::new(ChatRole::System, system));
    }
    messages.extend(history.iter().filter(usable).filter_map(|m| {
        let role = match m.role {
            Role::System => ChatRole::System,
            Role::User => ChatRole::User,
            Role::Assistant => ChatRole::Assistant,
            Role::Attachment | Role::Summary => return None,
        };
        Some(ChatMessage::new(role, m.content.clone()))
    }));
    messages
}

/// An attachment whose text goes into the prompt as a numbered source.
fn is_sent_attachment(message: &Message) -> bool {
    message.role == Role::Attachment
        && !matches!(
            message.status,
            MessageStatus::Failed(_) | MessageStatus::Streaming
        )
        && !message.content.trim().is_empty()
}

/// Number given in the prompt to the first retrieved chunk: attached files come first.
pub fn first_context_number(history: &[Message]) -> usize {
    history.iter().filter(|m| is_sent_attachment(m)).count() + 1
}

/// Request asking the model to summarize `history` (for `/compact`).
pub fn build_summary_request(history: &[Message]) -> Vec<ChatMessage> {
    let mut transcript = String::new();
    for message in history.iter().filter(|m| {
        !matches!(
            m.status,
            MessageStatus::Failed(_) | MessageStatus::Streaming
        ) && !m.content.trim().is_empty()
    }) {
        let who = match message.role {
            Role::User => "User".to_owned(),
            Role::Assistant => "Assistant".to_owned(),
            Role::System => "System".to_owned(),
            Role::Summary => "Summary of earlier messages".to_owned(),
            Role::Attachment => format!(
                "Attached file {}",
                message.source.as_deref().unwrap_or_default()
            ),
        };
        transcript.push_str(&format!("### {who}\n{}\n\n", message.content.trim()));
    }
    vec![
        ChatMessage::new(
            ChatRole::System,
            "You summarize conversations so that they can continue without the original \
             messages. Keep every fact, decision, constraint, name, number, file name and \
             important code, plus the open questions. Drop pleasantries and repetition. \
             Write in the language of the conversation, as concise bullet points.",
        ),
        ChatMessage::new(
            ChatRole::User,
            format!("Summarize this conversation:\n\n{}", transcript.trim_end()),
        ),
    ]
}

fn system_message(system_prompt: &str, summary: Option<&str>, chunks: &[ContextChunk]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let prompt = system_prompt.trim();
    if !prompt.is_empty() {
        parts.push(prompt.to_owned());
    }
    if let Some(summary) = summary {
        parts.push(format!(
            "Summary of the conversation so far (earlier messages are not included):\n{}",
            summary.trim()
        ));
    }
    if !chunks.is_empty() {
        let mut block = String::from(
            "Use the following context to answer when it is relevant. \
             Cite sources by their number.\n",
        );
        for (i, chunk) in chunks.iter().enumerate() {
            block.push_str(&format!(
                "\n[{}] {}\n{}\n",
                i + 1,
                chunk.label(),
                chunk.text.trim()
            ));
        }
        parts.push(block.trim_end().to_owned());
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Conversation;

    fn history() -> Vec<Message> {
        let mut c = Conversation::new();
        c.push(Role::User, "Q1", MessageStatus::Complete);
        c.push(Role::Assistant, "partial", MessageStatus::Cancelled);
        c.push(Role::User, "Q2", MessageStatus::Complete);
        c.push(Role::Assistant, "", MessageStatus::Failed("boom".into()));
        c.push(Role::User, "Q3", MessageStatus::Complete);
        c.messages().to_vec()
    }

    fn summary(messages: &[ChatMessage]) -> Vec<(ChatRole, &str)> {
        messages
            .iter()
            .map(|m| (m.role, m.content.as_str()))
            .collect()
    }

    #[test]
    fn system_prompt_then_history_without_failed_messages() {
        let messages = build_messages("Be brief.", &Context::default(), &history());
        assert_eq!(
            summary(&messages),
            vec![
                (ChatRole::System, "Be brief."),
                (ChatRole::User, "Q1"),
                (ChatRole::Assistant, "partial"),
                (ChatRole::User, "Q2"),
                (ChatRole::User, "Q3"),
            ]
        );
    }

    #[test]
    fn empty_system_prompt_is_omitted() {
        let messages = build_messages("  ", &Context::default(), &history());
        assert_eq!(messages[0].role, ChatRole::User);
    }

    #[test]
    fn context_is_merged_into_the_single_system_message() {
        let context = Context {
            chunks: vec![ContextChunk {
                source: "notes.md".into(),
                location: String::new(),
                text: "The sky is green.".into(),
            }],
        };
        let messages = build_messages("Be brief.", &context, &history());
        let systems: Vec<&ChatMessage> = messages
            .iter()
            .filter(|m| m.role == ChatRole::System)
            .collect();
        assert_eq!(systems.len(), 1);
        assert!(systems[0].content.starts_with("Be brief."));
        assert!(
            systems[0]
                .content
                .contains("[1] notes.md\nThe sky is green.")
        );
    }

    #[test]
    fn attachments_become_numbered_sources_before_retrieved_chunks() {
        let mut c = Conversation::new();
        c.push_attachment("./plan.md", "Plan du cours");
        c.push(Role::User, "Résume le plan", MessageStatus::Complete);
        let context = Context {
            chunks: vec![ContextChunk {
                source: "rag.md".into(),
                location: "§ Intro".into(),
                text: "extrait".into(),
            }],
        };
        assert_eq!(first_context_number(c.messages()), 2);
        let messages = build_messages("", &context, c.messages());
        assert_eq!(messages.len(), 2, "one system message + the question");
        let system = &messages[0].content;
        assert!(system.contains("[1] ./plan.md\nPlan du cours"));
        assert!(system.contains("[2] rag.md § Intro\nextrait"));
        assert_eq!(messages[1].content, "Résume le plan");
    }

    #[test]
    fn a_complete_summary_is_sent_in_the_system_message() {
        let mut c = Conversation::new();
        c.push(Role::Summary, "- on parle de Rust", MessageStatus::Complete);
        c.push(Role::User, "Et ensuite ?", MessageStatus::Complete);
        let messages = build_messages("Be brief.", &Context::default(), c.messages());
        assert!(
            messages[0]
                .content
                .contains("Summary of the conversation so far")
        );
        assert!(messages[0].content.contains("- on parle de Rust"));
        assert_eq!(messages.len(), 2);

        let mut c = Conversation::new();
        c.push(Role::Summary, "partiel", MessageStatus::Cancelled);
        let messages = build_messages("", &Context::default(), c.messages());
        assert!(messages.is_empty(), "an interrupted summary is not used");
    }

    #[test]
    fn summary_request_contains_the_transcript() {
        let mut c = Conversation::new();
        c.push_attachment("a.txt", "données");
        c.push(Role::User, "Q", MessageStatus::Complete);
        c.push(Role::Assistant, "R", MessageStatus::Complete);
        let request = build_summary_request(c.messages());
        assert_eq!(request[0].role, ChatRole::System);
        let body = &request[1].content;
        assert!(body.contains("### Attached file a.txt\ndonnées"));
        assert!(body.contains("### User\nQ"));
        assert!(body.contains("### Assistant\nR"));
    }
}
