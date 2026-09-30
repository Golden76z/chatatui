//! Copying text to the clipboard.
//!
//! Two ways, both used: the system clipboard tool when one is installed (`wl-copy` on
//! Wayland, `xclip` or `xsel` on X11, `pbcopy` on macOS, `clip.exe` on Windows and in
//! WSL), and the OSC 52 escape
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

/// Windows' clipboard tool. It reads its input in the console code page unless the text
/// starts with a UTF-16LE byte order mark, so it is given UTF-16 (see [`encode_for`]).
const CLIP: &str = "clip.exe";

/// Clipboard tools to try, for this environment, with their arguments.
fn tools(env: impl Fn(&str) -> Option<String>) -> Vec<(&'static str, &'static [&'static str])> {
    let mut tools: Vec<(&'static str, &'static [&'static str])> = Vec::new();
    if cfg!(target_os = "macos") {
        tools.push(("pbcopy", &[]));
    }
    // Native Windows, or WSL where Windows programs can be run and share the clipboard.
    if cfg!(windows) || env("WSL_DISTRO_NAME").is_some() {
        tools.push((CLIP, &[]));
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
        .is_some_and(|mut stdin| stdin.write_all(&encode_for(program, text)).is_ok());
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

/// Bytes to give `program`: UTF-16LE with a byte order mark for `clip.exe`, UTF-8
/// otherwise.
fn encode_for(program: &str, text: &str) -> Vec<u8> {
    if program != CLIP {
        return text.as_bytes().to_vec();
    }
    let mut bytes = vec![0xFF, 0xFE];
    for unit in text.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    bytes
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
        if cfg!(windows) {
            assert_eq!(names(&[]), vec!["clip.exe"]);
        } else if !cfg!(target_os = "macos") {
            assert!(names(&[]).is_empty(), "no display: terminal only");
            assert_eq!(names(&["WAYLAND_DISPLAY"]), vec!["wl-copy"]);
            assert_eq!(names(&["DISPLAY"]), vec!["xclip", "xsel"]);
            assert_eq!(names(&["WSL_DISTRO_NAME"]), vec!["clip.exe"], "WSL");
        }
    }

    #[test]
    fn clip_exe_gets_utf16() {
        assert_eq!(encode_for("xclip", "é"), "é".as_bytes());
        assert_eq!(
            encode_for(CLIP, "é€"),
            vec![0xFF, 0xFE, 0xE9, 0x00, 0xAC, 0x20]
        );
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
