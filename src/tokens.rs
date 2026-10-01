//! Token estimates and number formatting.
//!
//! Servers report exact counts after each request (see `llm::Usage`); in between, the app
//! estimates. The estimate is deliberately simple — about four characters per token, plus a
//! few tokens of framing per message — and always shown as approximate.

use crate::llm::ChatMessage;

/// Tokens added by the chat template around each message (role markers, separators).
const PER_MESSAGE_OVERHEAD: u64 = 4;

/// Rough token count of a text.
pub fn estimate(text: &str) -> u64 {
    let chars = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
    chars.div_ceil(4)
}

/// Rough token count of a message, framing included.
pub fn estimate_message(content: &str) -> u64 {
    estimate(content) + PER_MESSAGE_OVERHEAD
}

/// Rough cost of an image (vision models count about a thousand tokens for a picture of
/// ordinary size).
pub const IMAGE_TOKENS: u64 = 1_000;

/// Rough token count of a whole prompt.
pub fn estimate_prompt(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .map(|m| estimate_message(&m.content) + IMAGE_TOKENS * m.images.len() as u64)
        .sum()
}

/// `3 214` (thin grouping, French style).
pub fn format_count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push('\u{202f}'); // narrow no-break space
        }
        out.push(c);
    }
    out
}

/// `812`, `3,2k`, `32k`, `1,2M` for the status bar.
pub fn format_short(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..10_000 => {
            let tenths = (n + 50) / 100;
            format!("{},{}k", tenths / 10, tenths % 10)
        }
        10_000..1_000_000 => format!("{}k", (n + 500) / 1_000),
        _ => {
            let tenths = (n + 50_000) / 100_000;
            format!("{},{}M", tenths / 10, tenths % 10)
        }
    }
}

/// A byte count for the UI: `4,4 Go`, `812 Mo`, `96 ko`.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 3] = [("Go", 1 << 30), ("Mo", 1 << 20), ("ko", 1 << 10)];
    for (unit, scale) in UNITS {
        if bytes >= scale {
            let value = bytes as f64 / scale as f64;
            return if value < 10.0 {
                format!("{value:.1} {unit}").replace('.', ",")
            } else {
                format!("{} {unit}", value.round() as u64)
            };
        }
    }
    format!("{bytes} o")
}

/// Share of `used` in `total`, in percent (0 when `total` is 0).
pub fn percent(used: u64, total: u64) -> u64 {
    used.saturating_mul(100).checked_div(total).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatRole;

    #[test]
    fn estimates_about_four_characters_per_token() {
        assert_eq!(estimate(""), 0);
        assert_eq!(estimate("abcd"), 1);
        assert_eq!(estimate("abcde"), 2);
        assert_eq!(estimate("été"), 1, "characters, not bytes");
        let prompt = [
            ChatMessage::new(ChatRole::System, "abcd"),
            ChatMessage::new(ChatRole::User, "abcdabcd"),
        ];
        assert_eq!(estimate_prompt(&prompt), 1 + 4 + 2 + 4);
    }

    #[test]
    fn formatting() {
        assert_eq!(format_count(7), "7");
        assert_eq!(format_count(3_214), "3\u{202f}214");
        assert_eq!(format_count(1_048_576), "1\u{202f}048\u{202f}576");
        assert_eq!(format_short(812), "812");
        assert_eq!(format_short(3_214), "3,2k");
        assert_eq!(format_short(8_192), "8,2k");
        assert_eq!(format_short(32_768), "33k");
        assert_eq!(format_short(200_000), "200k");
        assert_eq!(format_short(1_048_576), "1,0M");
        assert_eq!(percent(50, 200), 25);
        assert_eq!(percent(5, 0), 0);
    }

    #[test]
    fn byte_counts_read_like_a_file_manager() {
        assert_eq!(format_bytes(0), "0 o");
        assert_eq!(format_bytes(1_023), "1023 o");
        assert_eq!(format_bytes(1_024), "1,0 ko");
        assert_eq!(format_bytes(98_304), "96 ko");
        assert_eq!(format_bytes(512 * (1 << 20)), "512 Mo");
        assert_eq!(format_bytes(4_724_464_025), "4,4 Go");
        // No panic and no overflow at the top of the range.
        assert!(format_bytes(u64::MAX).ends_with(" Go"));
    }
}
