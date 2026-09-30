//! Terminal setup and teardown, including the optional kitty keyboard protocol.
//!
//! The kitty protocol ("keyboard enhancement") lets us tell `Shift+Enter` and `Ctrl+M`
//! apart from a plain `Enter`. Terminals that do not support it still work: the app falls
//! back to `Alt+Enter` for new lines and `F2` for the model picker.

use std::io::stdout;
use std::sync::atomic::{AtomicBool, Ordering};

use color_eyre::Result;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::supports_keyboard_enhancement,
};
use ratatui::DefaultTerminal;

/// Whether we pushed keyboard enhancement flags and must pop them on exit.
static ENHANCED: AtomicBool = AtomicBool::new(false);
/// Whether we enabled mouse capture and must disable it on exit.
static MOUSE: AtomicBool = AtomicBool::new(false);

/// An initialized terminal.
pub struct Tty {
    pub terminal: DefaultTerminal,
    /// `true` when the terminal reports disambiguated modifier keys.
    pub keyboard_enhanced: bool,
}

/// Puts the terminal in raw/alternate-screen mode and enables keyboard enhancement when
/// supported, and mouse capture when requested.
pub fn init(mouse_capture: bool) -> Result<Tty> {
    // `ratatui::init` also installs a panic hook that restores the terminal.
    let terminal = ratatui::try_init()?;
    install_panic_hook();
    // Bracketed paste delivers pasted text in one event, so its newlines do not submit.
    execute!(stdout(), EnableBracketedPaste)?;
    if mouse_capture {
        execute!(stdout(), EnableMouseCapture)?;
        MOUSE.store(true, Ordering::SeqCst);
    }

    let keyboard_enhanced = matches!(supports_keyboard_enhancement(), Ok(true))
        && execute!(
            stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok();
    ENHANCED.store(keyboard_enhanced, Ordering::SeqCst);

    Ok(Tty {
        terminal,
        keyboard_enhanced,
    })
}

/// Restores the terminal to its original state. Safe to call more than once.
pub fn restore() {
    reset_modes();
    ratatui::restore();
}

/// Undoes the terminal modes enabled by [`init`]. Errors are ignored: there is nothing
/// sensible to do about them while shutting down.
fn reset_modes() {
    if ENHANCED.swap(false, Ordering::SeqCst) {
        let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
    }
    if MOUSE.swap(false, Ordering::SeqCst) {
        let _ = execute!(stdout(), DisableMouseCapture);
    }
    let _ = execute!(stdout(), DisableBracketedPaste);
}

thread_local! {
    /// Set while [`quietly`] runs: panics are expected and contained there.
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f`, turning a panic into `None` without touching the terminal or printing
/// anything. For third-party parsers (PDF…) that may panic on unusual input.
pub fn quietly<T>(f: impl FnOnce() -> T) -> Option<T> {
    QUIET.with(|quiet| quiet.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    QUIET.with(|quiet| quiet.set(false));
    result.ok()
}

/// Chains a hook that resets our terminal modes before ratatui's own restoring hook runs.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if QUIET.with(std::cell::Cell::get) {
            return; // contained by `quietly`
        }
        reset_modes();
        previous(info);
    }));
}
