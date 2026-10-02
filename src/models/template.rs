//! Renders a conversation into the prompt string an architecture expects.
//!
//! Nothing in this repository did this before J34: every provider so far talks to a server
//! that applies the model's own chat template. In-process, it is ours.

use crate::llm::{ChatMessage, ChatRole};

use super::ModelError;

/// Renders a conversation into the prompt string `architecture` expects.
///
/// Pure: no I/O, no clock, no allocation beyond the string it returns.
pub fn render(architecture: &str, messages: &[ChatMessage]) -> Result<String, ModelError> {
    match architecture {
        "qwen3" | "qwen2" => Ok(chatml(messages)),
        other => Err(ModelError::Unsupported(format!(
            "architecture « {other} » non prise en charge par le moteur local"
        ))),
    }
}

/// ChatML, as Qwen uses it.
///
/// The trailing `<think>\n\n</think>\n\n` is not decoration: Qwen3 is a hybrid reasoning
/// model whose own template carries `enable_thinking` logic, and without it the model opens
/// its own think block and writes its reasoning into the reply. Reasoning tokens are out of
/// scope (J33), so the block is pre-filled empty — Qwen's documented way to turn thinking
/// off — and the model continues straight into its answer.
///
/// Verified against the real template shipped in a Qwen3-4B GGUF on disk
/// (`tokenizer.chat_template`): its role turns render as
/// `'<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n'`, matching
/// what is emitted below, and its generation prompt is
/// `'<|im_start|>assistant\n'` followed by `'<think>\n\n</think>\n\n'` exactly when
/// `enable_thinking is defined and enable_thinking is false` — the state this renderer always
/// produces, since reasoning is out of scope.
fn chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            // The local provider offers no tools, so a tool result is history the model
            // must see without being taught a turn shape it never trained on.
            ChatRole::Tool => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    out
}

#[cfg(test)]
mod tests {
    use crate::llm::{ChatMessage, ChatRole};

    use super::*;

    #[test]
    fn renders_chatml_and_opens_the_assistant_turn() {
        let messages = [
            ChatMessage::new(ChatRole::System, "Tu es concis."),
            ChatMessage::new(ChatRole::User, "Bonjour"),
        ];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert_eq!(
            prompt,
            "<|im_start|>system\nTu es concis.<|im_end|>\n\
             <|im_start|>user\nBonjour<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// Review focus 1: Qwen3 is a hybrid reasoning model. Its own template carries
    /// `enable_thinking` logic, and plain ChatML without it makes the model write
    /// `<think>…</think>` into the reply. Reasoning tokens are out of scope since J33, so
    /// they would arrive in the conversation as literal text. Qwen's documented way to turn
    /// thinking off is to pre-fill an empty think block, which is what the renderer emits.
    #[test]
    fn thinking_is_disabled_by_a_prefilled_empty_block() {
        let messages = [ChatMessage::new(ChatRole::User, "Salut")];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "the empty think block is pre-filled so the model does not open its own: {prompt:?}"
        );
    }

    #[test]
    fn a_prior_assistant_turn_is_closed() {
        let messages = [
            ChatMessage::new(ChatRole::User, "Un"),
            ChatMessage::new(ChatRole::Assistant, "Deux"),
            ChatMessage::new(ChatRole::User, "Trois"),
        ];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert!(
            prompt.contains("<|im_start|>assistant\nDeux<|im_end|>\n"),
            "{prompt:?}"
        );
    }

    /// A tool message has no place in a ChatML prompt for a model that was offered no tools.
    /// Rendering it as a `tool` role would teach the model a turn shape it never saw in
    /// training; folding it into the user turn keeps the transcript honest.
    #[test]
    fn a_tool_message_is_folded_into_the_user_turn() {
        let messages = [ChatMessage::tool_result("call_1", "42")];

        let prompt = render("qwen3", &messages).expect("qwen3 is supported");

        assert!(
            prompt.contains("<|im_start|>user\n42<|im_end|>\n"),
            "{prompt:?}"
        );
        assert!(
            !prompt.contains("tool"),
            "no tool role is emitted: {prompt:?}"
        );
    }

    /// The engine lists every GGUF in the store, so the user can pick a Gemma or a Llama. It
    /// must be refused by name: being silently wrong is worse than being unavailable.
    #[test]
    fn an_unsupported_architecture_is_refused_by_name() {
        let messages = [ChatMessage::new(ChatRole::User, "Bonjour")];

        let error = render("gemma3", &messages).expect_err("only qwen3 is in scope");

        let text = error.to_string();
        assert!(text.contains("gemma3"), "{text}");
    }

    #[test]
    fn an_empty_conversation_still_opens_the_assistant_turn() {
        let prompt = render("qwen3", &[]).expect("qwen3 is supported");

        assert_eq!(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}
