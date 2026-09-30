//! Reading web pages (`fetch_url` tool, `/add https://…`).
//!
//! HTML is turned into readable text with a light converter: scripts, styles and page
//! chrome (navigation, headers, footers, forms) are dropped, headings and list items keep
//! a Markdown marker, links keep their text, entities are decoded. PDFs are read with the
//! same extractor as indexing; plain text, Markdown and JSON are kept as they are.

use std::time::Duration;

use futures::StreamExt;

use crate::rag::extract::{self, FileKind};

/// Largest page downloaded.
const MAX_BYTES: usize = 5 * 1024 * 1024;
/// Characters of a page kept.
pub const MAX_CHARS: usize = 60_000;
/// Time allowed for a page.
const TIMEOUT: Duration = Duration::from_secs(20);

/// A downloaded page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    /// `<title>`, when there is one.
    pub title: Option<String>,
    pub text: String,
}

/// Downloads `url` (http or https) and returns its text. Errors are user-facing.
pub async fn fetch(url: &str) -> Result<Page, String> {
    let url = url.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(format!(
            "{url} : seules les adresses http(s) sont acceptées"
        ));
    }
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .user_agent(concat!("chatatui/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("client HTTP : {e}"))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("{url} : {}", describe(&e)))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("{url} : erreur HTTP {}", status.as_u16()));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("{url} : {}", describe(&e)))?;
        if body.len() + chunk.len() > MAX_BYTES {
            return Err(format!("{url} : page trop grosse (plus de 5 Mo)"));
        }
        body.extend_from_slice(&chunk);
    }
    let page = if content_type.contains("pdf") || body.starts_with(b"%PDF") {
        let extracted = tokio::task::spawn_blocking(move || extract::extract(FileKind::Pdf, &body))
            .await
            .map_err(|e| format!("lecture interrompue ({e})"))??;
        Page {
            title: None,
            text: extracted.pages.join("\n\n"),
        }
    } else if content_type.contains("html") || content_type.is_empty() {
        let html = String::from_utf8_lossy(&body);
        Page {
            title: title(&html),
            text: html_to_text(&html),
        }
    } else if content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
    {
        Page {
            title: None,
            text: String::from_utf8_lossy(&body).into_owned(),
        }
    } else {
        return Err(format!("{url} : type non pris en charge ({content_type})"));
    };
    if page.text.trim().is_empty() {
        return Err(format!("{url} : aucun texte lisible"));
    }
    Ok(Page {
        text: truncate(page.text),
        ..page
    })
}

fn describe(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "délai dépassé".into()
    } else if error.is_connect() {
        "connexion impossible".into()
    } else {
        error.to_string()
    }
}

fn truncate(text: String) -> String {
    if text.chars().count() <= MAX_CHARS {
        return text;
    }
    let mut kept: String = text.chars().take(MAX_CHARS).collect();
    kept.push_str("\n[… page tronquée]");
    kept
}

/// The page's `<title>`.
pub fn title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title")?;
    let open_end = lower[start..].find('>')? + start + 1;
    let close = lower[open_end..].find("</title>")? + open_end;
    let title = decode_entities(html[open_end..close].trim());
    (!title.is_empty()).then_some(title)
}

/// Readable text of an HTML page.
pub fn html_to_text(html: &str) -> String {
    /// Elements whose content is never text for the reader.
    const SKIPPED: &[&str] = &[
        "script", "style", "noscript", "svg", "nav", "header", "footer", "form", "iframe",
        "template", "head", "aside", "button", "select",
    ];
    let mut out = String::new();
    let mut skip_depth: Vec<String> = Vec::new();
    let mut pre = false;
    let mut rest = html;
    while !rest.is_empty() {
        let Some(open) = rest.find('<') else {
            if skip_depth.is_empty() {
                push_text(&mut out, rest, pre);
            }
            break;
        };
        if open > 0 && skip_depth.is_empty() {
            push_text(&mut out, &rest[..open], pre);
        }
        rest = &rest[open..];
        if rest.starts_with("<!--") {
            rest = rest.find("-->").map_or("", |end| &rest[end + 3..]);
            continue;
        }
        let Some(close) = rest.find('>') else {
            break;
        };
        let tag = &rest[1..close];
        rest = &rest[close + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        if name.is_empty() {
            continue;
        }
        if SKIPPED.contains(&name.as_str()) {
            if closing {
                if skip_depth.last() == Some(&name) {
                    skip_depth.pop();
                }
            } else if !tag.ends_with('/') {
                skip_depth.push(name);
            }
            continue;
        }
        if !skip_depth.is_empty() {
            continue;
        }
        match (name.as_str(), closing) {
            ("h1", false) => block(&mut out, "# "),
            ("h2", false) => block(&mut out, "## "),
            ("h3" | "h4" | "h5" | "h6", false) => block(&mut out, "### "),
            ("li", false) => line(&mut out, "- "),
            ("pre", false) => {
                pre = true;
                block(&mut out, "```\n");
            }
            ("pre", true) => {
                pre = false;
                line(&mut out, "```");
                out.push_str("\n\n");
            }
            ("br", _) => out.push('\n'),
            ("td" | "th", true) => out.push_str(" | "),
            (
                "p" | "div" | "section" | "article" | "main" | "table" | "ul" | "ol" | "blockquote"
                | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "tr" | "dt" | "dd",
                _,
            ) => block(&mut out, ""),
            _ => {}
        }
    }
    tidy(&out)
}

/// Appends text, collapsing white space outside `<pre>`.
fn push_text(out: &mut String, text: &str, pre: bool) {
    let text = decode_entities(text);
    if pre {
        out.push_str(&text);
        return;
    }
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        if text.chars().next().is_some_and(char::is_whitespace) && !out.ends_with([' ', '\n']) {
            out.push(' ');
        }
        return;
    }
    if text.starts_with(char::is_whitespace) && !out.ends_with([' ', '\n']) && !out.is_empty() {
        out.push(' ');
    }
    out.push_str(&collapsed);
    if text.ends_with(char::is_whitespace) {
        out.push(' ');
    }
}

/// Starts a new paragraph, then writes `prefix`.
fn block(out: &mut String, prefix: &str) {
    let trimmed = out.trim_end_matches(' ').len();
    out.truncate(trimmed);
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push_str(if out.ends_with('\n') { "\n" } else { "\n\n" });
    }
    out.push_str(prefix);
}

/// Starts a new line, then writes `prefix`.
fn line(out: &mut String, prefix: &str) {
    let trimmed = out.trim_end_matches(' ').len();
    out.truncate(trimmed);
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(prefix);
}

/// Trims lines and keeps at most one blank line in a row.
fn tidy(text: &str) -> String {
    let mut out = String::new();
    let mut blank = 0;
    for line in text.lines().map(str::trim_end) {
        if line.trim().is_empty() {
            blank += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank > 0 { "\n\n" } else { "\n" });
        }
        blank = 0;
        out.push_str(line);
    }
    out
}

/// Decodes the usual named entities and numeric references.
pub fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let Some(end) = rest[..rest.len().min(12)].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            "eacute" => Some('é'),
            "egrave" => Some('è'),
            "ecirc" => Some('ê'),
            "agrave" => Some('à'),
            "ccedil" => Some('ç'),
            "ocirc" => Some('ô'),
            "ugrave" => Some('ù'),
            "icirc" => Some('î'),
            "laquo" => Some('«'),
            "raquo" => Some('»'),
            "hellip" => Some('…'),
            "mdash" => Some('—'),
            "ndash" => Some('–'),
            "rsquo" => Some('’'),
            "lsquo" => Some('‘'),
            "euro" => Some('€'),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<!doctype html><html><head><title>Les traits &amp; Rust</title>
<style>body { color: red }</style><script>alert("x")</script></head>
<body><nav><a href="/">Accueil</a> | <a href="/blog">Blog</a></nav>
<main><h1>Les traits</h1><p>Un <b>trait</b> décrit un comportement
commun.<br>Deuxième&nbsp;ligne.</p>
<ul><li>Display</li><li>Debug &#8212; pour le débogage</li></ul>
<pre><code>impl Display for Point {
    fn fmt(&amp;self) {}
}</code></pre>
<!-- commentaire --><footer>© 2026</footer></main></body></html>"#;

    #[test]
    fn html_becomes_readable_text() {
        assert_eq!(title(PAGE).as_deref(), Some("Les traits & Rust"));
        let text = html_to_text(PAGE);
        assert_eq!(
            text,
            "# Les traits\n\nUn trait décrit un comportement commun.\nDeuxième ligne.\n\n\
             - Display\n- Debug — pour le débogage\n\n\
             ```\nimpl Display for Point {\n    fn fmt(&self) {}\n}\n```"
        );
    }

    #[test]
    fn entities_are_decoded() {
        assert_eq!(
            decode_entities("a &lt; b &amp;&amp; c &#x41; &unknown; &"),
            "a < b && c A &unknown; &"
        );
    }

    #[tokio::test]
    async fn fetches_a_local_page() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let body = "<html><head><title>Test</title></head><body><p>Bonjour</p></body></html>";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let page = fetch(&format!("http://{address}/")).await.expect("page");
        assert_eq!(page.title.as_deref(), Some("Test"));
        assert_eq!(page.text, "Bonjour");
        assert!(fetch("file:///etc/passwd").await.is_err());
    }
}
