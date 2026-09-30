//! Incremental Server-Sent Events parser.
//!
//! Works on raw bytes so that a chunk boundary falling inside a line — or inside a multi-byte
//! UTF-8 character — is handled transparently: a line is only decoded once it is complete.
//!
//! Supported subset of the SSE spec (what OpenAI-compatible servers emit): `\n` and `\r\n`
//! line endings, `data:` fields (several per event are joined with `\n`), comments (lines
//! starting with `:`), other fields (`event:`, `id:`, `retry:`) ignored.

/// Parses a byte stream into the `data` payloads of complete events.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes of the current, still incomplete line.
    line: Vec<u8>,
    /// Data lines accumulated for the current event.
    data: Vec<String>,
}

impl SseParser {
    /// Creates an empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a chunk and returns the payloads of the events it completed, in order.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut events = Vec::new();
        let mut rest = chunk;
        while let Some(pos) = rest.iter().position(|&b| b == b'\n') {
            self.line.extend_from_slice(&rest[..pos]);
            rest = &rest[pos + 1..];
            let line = std::mem::take(&mut self.line);
            if let Some(event) = self.process_line(&line) {
                events.push(event);
            }
        }
        self.line.extend_from_slice(rest);
        events
    }

    /// Flushes an event left unterminated when the stream ended.
    pub fn finish(&mut self) -> Option<String> {
        let line = std::mem::take(&mut self.line);
        if !line.is_empty()
            && let Some(event) = self.process_line(&line)
        {
            return Some(event);
        }
        self.dispatch()
    }

    fn process_line(&mut self, line: &[u8]) -> Option<String> {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(b":") {
            return None; // comment / keep-alive
        }
        let (field, value) = match line.iter().position(|&b| b == b':') {
            Some(colon) => {
                let value = &line[colon + 1..];
                (&line[..colon], value.strip_prefix(b" ").unwrap_or(value))
            }
            None => (line, &b""[..]),
        };
        if field == b"data" {
            self.data.push(String::from_utf8_lossy(value).into_owned());
        }
        None
    }

    fn dispatch(&mut self) -> Option<String> {
        if self.data.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.data).join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLLAMA: &str = include_str!("../../tests/fixtures/ollama.sse");
    const LLAMA_CPP: &str = include_str!("../../tests/fixtures/llamacpp.sse");

    fn parse_all(chunks: &[&[u8]]) -> Vec<String> {
        let mut parser = SseParser::new();
        let mut events: Vec<String> = chunks.iter().flat_map(|c| parser.push(c)).collect();
        events.extend(parser.finish());
        events
    }

    #[test]
    fn parses_ollama_stream() {
        let events = parse_all(&[OLLAMA.as_bytes()]);
        assert_eq!(events.len(), 6);
        assert!(events[0].starts_with(r#"{"id":"chatcmpl-"#));
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    }

    #[test]
    fn parses_llama_cpp_stream_with_crlf() {
        let events = parse_all(&[LLAMA_CPP.as_bytes()]);
        assert_eq!(events.len(), 5);
        assert!(events.iter().all(|e| !e.contains('\r')));
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    }

    #[test]
    fn any_split_point_gives_the_same_events() {
        for fixture in [OLLAMA, LLAMA_CPP] {
            let bytes = fixture.as_bytes();
            let expected = parse_all(&[bytes]);
            // Includes splits in the middle of lines and of multi-byte characters (é, 🦀).
            for i in 0..=bytes.len() {
                assert_eq!(
                    parse_all(&[&bytes[..i], &bytes[i..]]),
                    expected,
                    "split at {i}"
                );
            }
        }
    }

    #[test]
    fn byte_by_byte_gives_the_same_events() {
        let bytes = OLLAMA.as_bytes();
        let chunks: Vec<&[u8]> = bytes.chunks(1).collect();
        assert_eq!(parse_all(&chunks), parse_all(&[bytes]));
    }

    #[test]
    fn multi_byte_character_split_across_chunks() {
        let bytes = "data: caf\u{e9}\n\n".as_bytes();
        let split = bytes.len() - 3; // inside the two bytes of 'é'
        assert_eq!(parse_all(&[&bytes[..split], &bytes[split..]]), vec!["café"]);
    }

    #[test]
    fn comments_and_other_fields_are_ignored() {
        let input = b": keep-alive\nevent: message\nid: 7\nretry: 100\ndata: hello\n\n";
        assert_eq!(parse_all(&[input]), vec!["hello"]);
    }

    #[test]
    fn multiple_data_lines_are_joined() {
        assert_eq!(parse_all(&[b"data: a\ndata: b\n\n"]), vec!["a\nb"]);
    }

    #[test]
    fn value_without_space_after_colon() {
        assert_eq!(parse_all(&[b"data:[DONE]\n\n"]), vec!["[DONE]"]);
    }

    #[test]
    fn unterminated_last_event_is_flushed() {
        assert_eq!(parse_all(&[b"data: tail"]), vec!["tail"]);
    }

    #[test]
    fn blank_lines_without_data_emit_nothing() {
        assert!(parse_all(&[b"\n\n\r\n"]).is_empty());
    }
}
