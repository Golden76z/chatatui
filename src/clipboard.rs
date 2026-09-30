//! Copying text to the clipboard.
//!
//! Two ways, both used: the system clipboard tool when one is installed (`wl-copy` on
//! Wayland, `xclip` or `xsel` on X11, `pbcopy` on macOS), and the OSC 52 escape
//! sequence, which asks the terminal itself to set the clipboard (works over SSH in most
//! terminals: kitty, WezTerm, Alacritty, foot, iTerm2, Windows Terminal; not GNOME
//! Terminal). Neither can confirm that the copy happened.

use std::{
    io::Write,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

/// Longest a clipboard tool may take.
const TOOL_TIMEOUT: Duration = Duration::from_secs(3);

/// How the text was handed over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Copied {
    /// Given to this system tool (and to the terminal).
    Tool(&'static str),
    /// Only sent to the terminal (OSC 52).
    TerminalOnly,
}

/// Clipboard tools to try, for this environment, with their arguments.
fn tools(env: impl Fn(&str) -> Option<String>) -> Vec<(&'static str, &'static [&'static str])> {
    let mut tools: Vec<(&'static str, &'static [&'static str])> = Vec::new();
    if cfg!(target_os = "macos") {
        tools.push(("pbcopy", &[]));
    }
    if env("WAYLAND_DISPLAY").is_some() {
        tools.push(("wl-copy", &[]));
    }
    if env("DISPLAY").is_some() {
        tools.push(("xclip", &["-selection", "clipboard"]));
        tools.push(("xsel", &["--clipboard", "--input"]));
    }
    tools
}

/// Gives `text` to the first clipboard tool that works (blocking).
pub fn copy_with_tool(text: &str) -> Copied {
    for (program, args) in tools(|name| std::env::var(name).ok()) {
        if run_tool(program, args, text) {
            return Copied::Tool(program);
        }
    }
    Copied::TerminalOnly
}

fn run_tool(program: &str, args: &[&str], text: &str) -> bool {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false; // not installed
    };
    let written = child
        .stdin
        .take()
        .is_some_and(|mut stdin| stdin.write_all(text.as_bytes()).is_ok());
    // The tools fork a process that keeps serving the selection; the one we started exits.
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return written && status.success(),
            Ok(None) if start.elapsed() < TOOL_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(10));
            }
            _ => {
                let _ = child.kill();
                return false;
            }
        }
    }
}

/// Asks the terminal to set the clipboard (OSC 52), writing to `out`.
pub fn osc52(out: &mut impl Write, text: &str) -> std::io::Result<()> {
    crossterm::execute!(
        out,
        crossterm::clipboard::CopyToClipboard::to_clipboard_from(text)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_depend_on_the_display_server() {
        let env = |vars: &'static [&'static str]| {
            move |name: &str| vars.contains(&name).then(|| "1".to_owned())
        };
        let names = |vars| -> Vec<&str> { tools(env(vars)).into_iter().map(|t| t.0).collect() };
        if !cfg!(target_os = "macos") {
            assert!(names(&[]).is_empty(), "no display: terminal only");
            assert_eq!(names(&["WAYLAND_DISPLAY"]), vec!["wl-copy"]);
            assert_eq!(names(&["DISPLAY"]), vec!["xclip", "xsel"]);
        }
    }

    #[test]
    fn osc52_sequence() {
        let mut out = Vec::new();
        osc52(&mut out, "héllo").expect("write");
        let sequence = String::from_utf8(out).expect("ascii");
        assert!(sequence.starts_with("\u{1b}]52;c;"), "{sequence:?}");
        assert!(sequence.contains("aMOpbGxv"), "base64 of the text");
    }

    #[test]
    fn missing_tool_is_not_an_error() {
        assert!(!run_tool("definitely-not-a-clipboard-tool", &[], "x"));
    }
}
