//! A conversation as a Markdown document (`/export`).
//!
//! Replies are written as they came (they are Markdown already), user messages as
//! plain paragraphs, attached files folded in `<details>` blocks, and each reply's cited
//! sources under it. Messages no longer in the model's context are exported too: the
//! file is a record of the whole conversation.

use crate::state::{Message, MessageStatus, Role};

/// The document for `messages`, titled `title`; `model` is shown under the title.
pub fn markdown(title: &str, model: &str, messages: &[Message]) -> String {
    let mut out = format!(
        "# {}\n\n*Conversation exportée de chatatui · {model}*\n",
        title.trim()
    );
    for message in messages {
        let heading = match message.role {
            Role::User => "Vous",
            Role::Assistant => "Assistant",
            Role::System => "Système",
            Role::Summary => "Résumé de la conversation",
            Role::Attachment => {
                let source = message.source.as_deref().unwrap_or("fichier");
                out.push_str(&format!(
                    "\n<details>\n<summary>📎 Fichier joint : {source}</summary>\n\n{}\n</details>\n",
                    fenced(&message.content)
                ));
                continue;
            }
        };
        out.push_str(&format!(
            "\n## {heading}\n\n{}\n",
            message.content.trim_end()
        ));
        match &message.status {
            MessageStatus::Complete | MessageStatus::Streaming => {}
            MessageStatus::Cancelled => out.push_str("\n*(réponse interrompue)*\n"),
            MessageStatus::Failed(error) => out.push_str(&format!("\n*(erreur : {error})*\n")),
        }
        if !message.citations.is_empty() {
            let sources: Vec<String> = message
                .citations
                .iter()
                .map(|c| format!("[{}] {}", c.number, c.label()))
                .collect();
            out.push_str(&format!("\n> Sources : {}\n", sources.join(" · ")));
        }
    }
    out
}

/// `text` in a code fence long enough not to be closed by fences inside it.
fn fenced(text: &str) -> String {
    let longest = text
        .lines()
        .map(|l| l.trim_start().chars().take_while(|c| *c == '`').count())
        .max()
        .unwrap_or(0);
    let fence = "`".repeat(longest.max(2) + 1);
    format!("{fence}\n{}\n{fence}", text.trim_end())
}

/// A file name for the conversation: its title in lowercase ASCII words joined by `-`.
pub fn file_name(title: &str) -> String {
    let mut slug = String::new();
    for c in title.chars().flat_map(fold_accent) {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
        if slug.len() >= 50 {
            break;
        }
    }
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        "conversation.md".to_owned()
    } else {
        format!("{slug}.md")
    }
}

/// `é` → `e`, `œ` → `oe`, … (common French and Western letters).
fn fold_accent(c: char) -> Vec<char> {
    let folded = match c {
        'à' | 'á' | 'â' | 'ä' | 'ã' | 'å' => "a",
        'À' | 'Á' | 'Â' | 'Ä' | 'Ã' | 'Å' => "A",
        'ç' => "c",
        'Ç' => "C",
        'é' | 'è' | 'ê' | 'ë' => "e",
        'É' | 'È' | 'Ê' | 'Ë' => "E",
        'î' | 'ï' | 'í' | 'ì' => "i",
        'Î' | 'Ï' | 'Í' | 'Ì' => "I",
        'ô' | 'ö' | 'ó' | 'ò' | 'õ' => "o",
        'Ô' | 'Ö' | 'Ó' | 'Ò' | 'Õ' => "O",
        'ù' | 'û' | 'ü' | 'ú' => "u",
        'Ù' | 'Û' | 'Ü' | 'Ú' => "U",
        'ÿ' => "y",
        'ñ' => "n",
        'œ' => "oe",
        'Œ' => "OE",
        'æ' => "ae",
        'Æ' => "AE",
        other => return vec![other],
    };
    folded.chars().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Citation, Conversation};

    #[test]
    fn conversation_as_markdown() {
        let mut c = Conversation::new();
        c.push_attachment("notes.md", "```rust\nfn a() {}\n```");
        c.push(Role::User, "Résume mes notes", MessageStatus::Complete);
        let reply = c.push(
            Role::Assistant,
            "Une fonction **a** [1].",
            MessageStatus::Complete,
        );
        if let Some(message) = c.get_mut(reply) {
            message.citations = vec![Citation {
                number: 1,
                path: "notes.md".into(),
                location: String::new(),
            }];
        }
        c.push(Role::Assistant, "Il", MessageStatus::Cancelled);
        let doc = markdown("Mes notes", "Ollama › llama3.2", c.messages());
        assert!(doc.starts_with(
            "# Mes notes\n\n*Conversation exportée de chatatui · Ollama › llama3.2*\n"
        ));
        assert!(doc.contains("<summary>📎 Fichier joint : notes.md</summary>\n\n````\n```rust"));
        assert!(doc.contains("## Vous\n\nRésume mes notes\n"));
        assert!(
            doc.contains("## Assistant\n\nUne fonction **a** [1].\n\n> Sources : [1] notes.md\n")
        );
        assert!(doc.contains("Il\n\n*(réponse interrompue)*"));
    }

    #[test]
    fn file_names_from_titles() {
        assert_eq!(
            file_name("Qu'est-ce qu'un trait générique ?"),
            "qu-est-ce-qu-un-trait-generique.md"
        );
        assert_eq!(file_name("Œuvre : « Été »"), "oeuvre-ete.md");
        assert_eq!(file_name("???"), "conversation.md");
    }
}
