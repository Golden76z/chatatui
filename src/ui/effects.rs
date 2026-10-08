//! Bounded visual effects, applied to the frame after the interface has drawn itself.
//!
//! The effects live here rather than in `App` because they need a clock, and `App::update` is
//! pure by design. `Runtime` measures the time between frames and hands it to [`Effects::render`],
//! which draws over the finished frame. Nothing here can change application state.

use std::time::Instant;

use ratatui::{Frame, layout::Rect};
use tachyonfx::{Duration, Effect, EffectRenderer, Motion, fx};

/// A moment worth marking. Independent of how it is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Animation {
    /// An overlay appeared over the conversation.
    PopupOpened,
    /// The first token of a reply arrived, replacing the waiting line.
    FirstToken,
    /// `/clear` dropped the context.
    ContextCleared,
}

impl Animation {
    /// The effect this animation is drawn with, over `area`.
    ///
    /// A fresh effect every time: `tachyonfx::Effect` is `Debug` and not `Clone`, so there is
    /// no template to copy.
    fn effect(self) -> Effect {
        let palette = crate::theme::palette();
        match self {
            // 150 ms: long enough to be seen as motion, short enough that a popup the user
            // asked for does not feel slow to arrive.
            // The timers are `u32` milliseconds on purpose: `EffectTimer` has `From<u32>` and
            // no `From<i32>`, so a bare `150` would not compile.
            Self::PopupOpened => fx::sweep_in(Motion::UpToDown, 10, 0, palette.bar_bg, 150u32),
            Self::FirstToken => fx::coalesce(120u32),
            Self::ContextCleared => fx::dissolve(200u32),
        }
    }

    /// The part of the screen this animation plays over.
    ///
    /// Here rather than in `Runtime` so that one place answers, for every animation, both what
    /// it looks like and where it happens. `App` is read for its layout only — the same layout
    /// `ui::render` uses, so an effect covers exactly what the frame drew.
    pub fn area(self, app: &crate::app::App) -> Rect {
        match self {
            Self::PopupOpened => crate::ui::popup_area(app.viewport),
            // The reply's first line, which is the row the waiting line occupies right now: a
            // token does not refresh the transcript, so its last line is still that waiting
            // line. Scrolled away from it, the row falls outside the pane and nothing plays.
            Self::FirstToken => {
                let content = crate::layout::chat_content(app.layout().chat);
                let last = app.transcript.total_lines().saturating_sub(1);
                match last
                    .checked_sub(app.scroll_offset())
                    .and_then(|row| u16::try_from(row).ok())
                    .map(|row| content.y.saturating_add(row))
                {
                    Some(y) if y < content.bottom() => Rect {
                        y,
                        height: 1,
                        ..content
                    },
                    _ => Rect::ZERO,
                }
            }
            // `/clear` moves the boundary past every message, so the lines leaving the context
            // are the whole conversation.
            Self::ContextCleared => app.layout().chat,
        }
    }
}

/// The effects in flight, and the clock that drives them.
///
/// `Debug` because `Runtime` derives it; `tachyonfx::Effect` is `Debug` too.
#[derive(Debug)]
pub struct Effects {
    running: Vec<(Animation, Rect, Effect)>,
    last_frame: Instant,
}

impl Default for Effects {
    fn default() -> Self {
        Self::new()
    }
}

impl Effects {
    pub fn new() -> Self {
        Self {
            running: Vec::new(),
            last_frame: Instant::now(),
        }
    }

    /// Time since the previous frame, for [`render`](Effects::render).
    ///
    /// Capped at a quarter of a second. A suspended process or an unfocused terminal can leave
    /// minutes between two frames; without the cap the first frame back would advance every
    /// effect to its end in one step, which looks like a flash rather than a transition.
    pub fn tick(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = now
            .duration_since(self.last_frame)
            .min(std::time::Duration::from_millis(250));
        self.last_frame = now;
        Duration::from_millis(u32::try_from(elapsed.as_millis()).unwrap_or(u32::MAX))
    }

    /// Starts `animation` over `area`, replacing one of the same kind already running.
    ///
    /// An empty area starts nothing: a terminal too small to show a popup has nothing to shade.
    pub fn start(&mut self, animation: Animation, area: Rect) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        self.running.retain(|(kind, _, _)| *kind != animation);
        self.running.push((animation, area, animation.effect()));
    }

    /// Draws every running effect over the frame.
    ///
    /// Finished effects are dropped here, *before* drawing, not after: an effect's last frame is
    /// its end state, and `fx::dissolve` ends on blank cells. Dropping it on the way out would
    /// leave that blank frame on screen, because `in_flight` would go false in the same breath
    /// and the redraw gate would stop asking for frames. Kept one frame longer, the effect holds
    /// the gate open for one more pass, and that pass draws the interface plain.
    pub fn render(&mut self, frame: &mut Frame, elapsed: Duration) {
        self.running.retain(|(_, _, effect)| !effect.done());
        for (_, area, effect) in &mut self.running {
            frame.render_effect(effect, *area, elapsed);
        }
    }

    /// Whether a bounded effect is running: the redraw gate's extra condition.
    ///
    /// Every effect here is bounded. Nothing continuous is ever added — a repeating effect
    /// never reports `done()`, and the gate would stay lifted for as long as it ran.
    pub fn in_flight(&self) -> bool {
        !self.running.is_empty()
    }

    /// Drops every running effect.
    ///
    /// Called on resize: each effect holds the `Rect` it was started on, and that rect no longer
    /// means anything once the terminal changed size.
    pub fn cancel_all(&mut self) {
        self.running.clear();
    }

    /// How many effects are running. For tests.
    #[cfg(test)]
    fn running_count(&self) -> usize {
        self.running.len()
    }
}

#[cfg(test)]
mod tests {
    use ratatui::layout::Rect;

    use super::*;

    /// Pressing F2 twice quickly opens, closes and reopens an overlay. `Effect` is not `Clone`,
    /// so there is no template to copy and a naive implementation pushes a second effect onto
    /// the first. They would then both run, and the gate would stay lifted for twice as long.
    #[test]
    fn starting_the_same_animation_twice_replaces_it() {
        let mut effects = Effects::new();
        let area = Rect::new(0, 0, 40, 10);

        effects.start(Animation::PopupOpened, area);
        effects.start(Animation::PopupOpened, area);

        assert_eq!(effects.running_count(), 1);
    }

    /// A terminal too small to show a popup produces an empty rect. tachyonfx has no reason to
    /// be asked to shade nothing, and being unavailable beats panicking.
    #[test]
    fn an_empty_area_starts_nothing() {
        let mut effects = Effects::new();

        effects.start(Animation::PopupOpened, Rect::new(0, 0, 0, 0));
        effects.start(Animation::PopupOpened, Rect::new(5, 5, 20, 0));

        assert!(!effects.in_flight());
    }

    /// Nothing running, nothing to redraw for.
    #[test]
    fn a_fresh_layer_is_not_in_flight() {
        assert!(!Effects::new().in_flight());
    }

    /// A reply arrives on one line of the conversation, `/clear` empties the whole of it, and
    /// only the sweep belongs to the popup. Until each animation answered for its own area they
    /// all shaded the popup rect, so the first token of a reply lit up the middle of the screen.
    #[test]
    fn each_animation_plays_where_it_belongs() {
        let mut app = crate::app::App::new(&crate::config::Config::default(), false);
        app.update(crate::action::Action::Resize {
            width: 80,
            height: 30,
        });

        let popup = Animation::PopupOpened.area(&app);
        let first_token = Animation::FirstToken.area(&app);
        let cleared = Animation::ContextCleared.area(&app);

        assert_eq!(cleared, app.layout().chat, "`/clear` drops every message");
        assert_eq!(first_token.height, 1, "one line, not the whole pane");
        assert!(
            cleared.height > first_token.height,
            "{cleared:?} vs {first_token:?}"
        );
        assert_ne!(popup, cleared, "a reply does not arrive in a popup");
        assert!(popup.width < cleared.width, "{popup:?}");
    }

    /// Scrolled up, or on a terminal with no room for the conversation, the reply's line is not
    /// on screen. Both reach the same guard: an effect over a row the pane does not contain is
    /// an effect over someone else's row, so there is none.
    #[test]
    fn a_line_outside_the_pane_animates_nothing() {
        let mut app = crate::app::App::new(&crate::config::Config::default(), false);
        app.update(crate::action::Action::Resize {
            width: 80,
            height: 0,
        });

        let area = Animation::FirstToken.area(&app);

        assert!(area.height == 0 || area.width == 0, "{area:?}");
        // And `start` refuses it, so nothing reaches tachyonfx.
        let mut effects = Effects::new();
        effects.start(Animation::FirstToken, area);
        assert!(!effects.in_flight());
    }

    use ratatui::{Terminal, backend::TestBackend};

    /// The test the whole design rests on: an effect changes the frame, and then stops changing
    /// it. If it never stopped, `in_flight` would never drop and the 30 fps cap would be gone.
    #[test]
    fn an_effect_changes_the_frame_and_then_stops() {
        let mut terminal = Terminal::new(TestBackend::new(20, 5)).expect("a test terminal");
        let area = Rect::new(0, 0, 20, 5);
        let paint = |frame: &mut Frame| {
            frame.render_widget(
                ratatui::widgets::Paragraph::new("bonjour").style(
                    ratatui::style::Style::default().fg(ratatui::style::Color::Indexed(110)),
                ),
                area,
            );
        };

        terminal.draw(|frame| paint(frame)).expect("draws");
        let plain = terminal.backend().buffer().clone();

        let mut effects = Effects::new();
        effects.start(Animation::PopupOpened, area);
        terminal
            .draw(|frame| {
                paint(frame);
                effects.render(frame, Duration::from_millis(40));
            })
            .expect("draws");
        assert_ne!(
            terminal.backend().buffer(),
            &plain,
            "mid-effect the frame differs"
        );
        assert!(effects.in_flight());

        // Past the effect's 150 ms.
        terminal
            .draw(|frame| {
                paint(frame);
                effects.render(frame, Duration::from_millis(200));
            })
            .expect("draws");
        assert!(
            effects.in_flight(),
            "a finished effect still asks for the frame that undoes it"
        );

        terminal
            .draw(|frame| {
                paint(frame);
                effects.render(frame, Duration::from_millis(40));
            })
            .expect("draws");
        assert_eq!(
            terminal.backend().buffer(),
            &plain,
            "once finished the frame is the plain one again"
        );
        assert!(!effects.in_flight(), "and only then does the gate drop");
    }

    /// `fx::dissolve` ends on blank cells, so its last frame is an empty conversation. Dropping
    /// a finished effect before drawing — rather than after — is what brings the screen back:
    /// the gate stops asking for frames the moment nothing is in flight, so a blank last frame
    /// would stay on screen until the next keystroke. `/clear` is that effect.
    #[test]
    fn an_effect_ending_on_blank_cells_still_leaves_the_frame_plain() {
        let mut terminal = Terminal::new(TestBackend::new(20, 2)).expect("a test terminal");
        let area = Rect::new(0, 0, 20, 2);
        let paint = |frame: &mut Frame| {
            frame.render_widget(ratatui::widgets::Paragraph::new("bonjour"), area);
        };

        terminal.draw(|frame| paint(frame)).expect("draws");
        let plain = terminal.backend().buffer().clone();

        let mut effects = Effects::new();
        effects.start(Animation::ContextCleared, area);
        let mut frames = 0;
        while effects.in_flight() && frames < 20 {
            terminal
                .draw(|frame| {
                    paint(frame);
                    effects.render(frame, Duration::from_millis(60));
                })
                .expect("draws");
            frames += 1;
        }

        assert!(frames < 20, "the dissolve ends on its own");
        assert_eq!(
            terminal.backend().buffer(),
            &plain,
            "the conversation is back on the last frame, not dissolved away"
        );
    }

    /// A suspended process or an unfocused terminal can leave minutes between frames. The cap in
    /// `tick` keeps that from advancing an effect to its end in a single step.
    #[test]
    fn a_long_gap_between_frames_is_capped() {
        let mut effects = Effects::new();
        effects.last_frame = Instant::now() - std::time::Duration::from_secs(600);

        let elapsed = effects.tick();

        assert_eq!(elapsed, Duration::from_millis(250));
    }

    /// The area an effect was started on is meaningless after a resize, and an effect finishing
    /// against the old geometry paints over the new layout. Dropping them is cheaper to reason
    /// about than re-aiming them, and a 150 ms effect is not worth rescuing.
    #[test]
    fn a_resize_cancels_everything_in_flight() {
        let mut effects = Effects::new();
        effects.start(Animation::PopupOpened, Rect::new(0, 0, 40, 10));
        assert!(effects.in_flight());

        effects.cancel_all();

        assert!(!effects.in_flight());
    }
}
