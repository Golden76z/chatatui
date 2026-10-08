# J35 — Bounded visual effects: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give four moments of the interface — a popup opening, the first token arriving, `/clear`, and waiting — a visible transition, without letting any of them break the 30 fps markdown cap.

**Architecture:** `Runtime` owns the effects because `Runtime` owns the clock; they are applied to the frame after `ui::render`, so `App` stays pure and `ui::render` stays read-only. Two effects are declared by `App` as `Effect::Animate(..)`; the popup sweep is detected by `Runtime` diffing `app.overlay.is_some()` across a dispatch, because eleven sites open overlays and none of them would remember a trigger. The waiting line's shimmer is not a tachyonfx effect at all — it is span styling in `Waiting::line`, driven by the spinner counter that already ticks.

**Tech Stack:** Rust 2024 (`rust-version = "1.88"`), ratatui 0.30.2, tachyonfx 0.25.2.

**Spec:** `docs/superpowers/specs/2026-10-08-tachyonfx-design.md`

## Execution Order

```
Task 1 (effect layer, gate, popup sweep)  ┐
Task 2 (transcript revision comparison)   ┘ in parallel, isolated worktrees
   └── Task 3 (the two Animate triggers, the shimmer)
          └── Task 4 (docs, dependency gate, manual check)
```

Tasks 1 and 2 share no file. **They must run in *isolated worktrees*, not the shared one.** In Rust the crate is a single compilation unit: each task's TDD red phase leaves the lib test binary uncompilable, so one agent's `cargo test` would fail on the other's half-written file. Hit for real in J32, confirmed in J33 and J34.

## Global Constraints

- Rust edition 2024, `rust-version = "1.88"`. No `unwrap()` or `expect()` outside `#[cfg(test)]` code.
- UI strings in French; code, comments, doc comments and the documentation files in English.
- **`App::update` stays pure: no I/O, and no clock.** The elapsed time an effect needs is measured in `Runtime`. A task that gives `App` a clock has failed even if its tests pass.
- **`ui::render(&App, frame)` stays read-only** and keeps its signature. Effects run after it.
- **The 30 fps markdown cap must survive.** The gate in `src/runtime.rs:281-290` redraws a tick only when `(transcript.revision(), spinner_frame())` changed. It gains `|| effects.in_flight()`, and **`in_flight()` counts bounded effects only** — nothing continuous ever enters it.
- **No `*-sys` crate is added.** The tree holds six today (`aws-lc-sys`, `dirs-sys`, `inotify-sys`, `libsqlite3-sys`, `linux-raw-sys`, `onig_sys`). Check with `cargo tree | grep -oE '[a-z0-9_-]+[-_]sys v[0-9.]+' | sort -u` — the plain `grep -i -- '-sys'` cannot match an underscore and has already lied to two agents.
- **One commit for the whole milestone.** Do not commit per task; each task ends at its checks. The orchestrator squashes the branch into one commit, per the repo's `J<N>: …` convention, in English like `J31`/`J32`/`J33`/`J34`.
- Effects are invisible to insta snapshots, which capture characters only. Test them by comparing `TestBackend` buffers.

## Review Focus

Input classes the spec implies but no task's own tests would otherwise exercise. Each line's test is added to the task that owns the code.

1. **The same effect is started while it is already running.** `Effect` is not `Clone`, and pressing F2 twice quickly opens, closes and reopens an overlay. `start` must replace rather than accumulate, or effects pile up and the gate never drops. (Task 1)
2. **A terminal too small for the popup.** `centered` clamps, and on a very small terminal the area can come out zero-width or zero-height. Starting an effect on it must be a no-op, not a panic inside tachyonfx. (Task 1)
3. **A long gap between two frames.** The process is suspended (`Ctrl+Z`), or the terminal is unfocused and the host throttles it. `tick()` then returns minutes. tachyonfx's `Duration` is millisecond-resolution and the effect must simply finish, not overflow or hang. (Task 1)
4. **`/clear` on an already-empty context.** `clear_context` returns early with `"le contexte est déjà vide"` and no context moves. Animating a dissolve over nothing would be a visible lie. (Task 3)
5. **The first token of a `/compact` summary.** The reply is a `Role::Summary` message, not an assistant turn. Whichever behaviour is chosen must be pinned, so it is a decision rather than an accident. (Task 3)

---

### Task 1: The effect layer, the redraw gate, and the popup sweep

**Files:**
- Create: `src/ui/effects.rs`
- Modify: `src/ui/mod.rs` (add `pub mod effects;` to the module list at `:9-18`)
- Modify: `src/action.rs:173` (the `Effect` enum gains `Animate(Animation)`)
- Modify: `src/runtime.rs:253` (the draw closure), `:281-290` (the gate), `:378` (`dispatch`: the overlay diff and the resize cancellation), `:384` (`execute`)
- Modify: `Cargo.toml` (add `tachyonfx`)
- Test: appended `#[cfg(test)] mod tests` inside `src/ui/effects.rs`

**Interfaces:**
- Consumes: nothing from other tasks. Existing and verified: `pub overlay: Option<Overlay>` is a **public field** of `App` (`src/app.rs:336`); `terminal.draw(|frame| ui::render(&self.app, frame))?` is the single draw call (`src/runtime.rs:253`); `fn execute(&mut self, effect: Effect)` matches on the effect enum (`src/runtime.rs:384`).
- Produces:
  ```rust
  // src/ui/effects.rs
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum Animation { PopupOpened, FirstToken, ContextCleared }

  pub struct Effects { /* … */ }

  impl Effects {
      pub fn new() -> Self;
      pub fn tick(&mut self) -> tachyonfx::Duration;
      pub fn start(&mut self, animation: Animation, area: ratatui::layout::Rect);
      pub fn render(&mut self, frame: &mut ratatui::Frame, elapsed: tachyonfx::Duration);
      pub fn in_flight(&self) -> bool;
  }
  ```
  Task 3 calls `start(Animation::FirstToken, area)` and `start(Animation::ContextCleared, area)`.
  `Animation` is re-exported through `crate::action::Effect::Animate(Animation)`.

- [ ] **Step 1: Add the dependency and confirm it costs what the spec says**

```bash
cargo add tachyonfx@0.25.2
cargo tree | grep -oE '[a-z0-9_-]+[-_]sys v[0-9.]+' | sort -u
```

Expected: exactly six `*-sys` lines — `aws-lc-sys`, `dirs-sys`, `inotify-sys`, `libsqlite3-sys`, `linux-raw-sys`, `onig_sys`. If a seventh appears, stop and report it: the spec's dependency claim is wrong and the milestone's cost has changed.

Then confirm the net new crates:

```bash
git diff Cargo.lock | grep '^+name = ' | sort
```

Expected: `tachyonfx`, `bon`, `micromath` and nothing else of substance.

- [ ] **Step 2: Write the failing test for `start` and `in_flight`**

Append to `src/ui/effects.rs` (create the file with just this test module and the `use` lines for now):

```rust
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
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test --lib ui::effects`
Expected: FAIL to compile, `cannot find type Effects in this scope` (and `Animation`, `running_count`).

- [ ] **Step 4: Implement the layer**

Write above the test module in `src/ui/effects.rs`:

```rust
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
}

/// The effects in flight, and the clock that drives them.
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
        let elapsed = now.duration_since(self.last_frame).min(std::time::Duration::from_millis(250));
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

    /// Draws every running effect over the frame, and drops the finished ones.
    pub fn render(&mut self, frame: &mut Frame, elapsed: Duration) {
        for (_, area, effect) in &mut self.running {
            frame.render_effect(effect, *area, elapsed);
        }
        self.running.retain(|(_, _, effect)| !effect.done());
    }

    /// Whether a bounded effect is running: the redraw gate's extra condition.
    ///
    /// Every effect here is bounded. Nothing continuous is ever added — a repeating effect
    /// never reports `done()`, and the gate would stay lifted for as long as it ran.
    pub fn in_flight(&self) -> bool {
        !self.running.is_empty()
    }

    /// How many effects are running. For tests.
    #[cfg(test)]
    fn running_count(&self) -> usize {
        self.running.len()
    }
}
```

Add `pub mod effects;` to `src/ui/mod.rs`, in alphabetical order among the existing `mod` lines at `:9-18` (between `context_view` and `find_bar`).

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test --lib ui::effects`
Expected: PASS, 3 tests.

- [ ] **Step 6: Write the failing test for boundedness — the property everything else rests on**

Append inside `src/ui/effects.rs`'s `mod tests`:

```rust
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
        assert!(!effects.in_flight(), "the effect finished");

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
```

- [ ] **Step 7: Run it to verify it fails**

Run: `cargo test --lib ui::effects`
Expected: FAIL. `an_effect_changes_the_frame_and_then_stops` may already pass if the implementation is right; `a_long_gap_between_frames_is_capped` fails to compile until `last_frame` is reachable from the test module (it is, same module). If both pass immediately, that is a finding about the tests, not a success — report it.

- [ ] **Step 8: Make them pass**

The Step 4 implementation already satisfies both. If `an_effect_changes_the_frame_and_then_stops` fails because `sweep_in`'s gradient leaves the buffer identical on a 20×5 area, widen the test terminal to 40×10 rather than weakening the assertion.

Run: `cargo test --lib ui::effects`
Expected: PASS, 5 tests.

- [ ] **Step 9: Wire the layer into `Runtime`**

In `src/runtime.rs`, add the field to the struct (next to `app`):

```rust
    effects: crate::ui::effects::Effects,
```

initialised with `effects: crate::ui::effects::Effects::new(),` in `Runtime::new`.

Replace the draw call at `:253`:

```rust
                let elapsed = self.effects.tick();
                terminal.draw(|frame| {
                    ui::render(&self.app, frame);
                    self.effects.render(frame, elapsed);
                })?;
```

If the borrow checker objects to `self.effects` being borrowed mutably inside a closure that also borrows `self.app`, split the borrow before the closure:

```rust
                let elapsed = self.effects.tick();
                let (app, effects) = (&self.app, &mut self.effects);
                terminal.draw(|frame| {
                    ui::render(app, frame);
                    effects.render(frame, elapsed);
                })?;
```

- [ ] **Step 10: Write the failing test for the gate**

Append to `src/runtime.rs`'s existing `mod tests` (if the module has no constructor for a bare `Runtime`, place this test in `src/ui/effects.rs` instead and assert `in_flight` directly — note which you chose in your report):

```rust
    /// The gate redraws a tick only when the screen changed. An effect changes the screen
    /// without touching the transcript or the spinner, so it has to be asked about.
    #[test]
    fn a_running_effect_asks_for_a_redraw() {
        let mut effects = crate::ui::effects::Effects::new();
        assert!(!effects.in_flight());

        effects.start(
            crate::ui::effects::Animation::PopupOpened,
            ratatui::layout::Rect::new(0, 0, 40, 10),
        );

        assert!(effects.in_flight(), "the tick arm redraws while this is true");
    }
```

- [ ] **Step 11: Run it to verify it fails, then wire the gate**

Run: `cargo test --lib a_running_effect_asks_for_a_redraw`
Expected: FAIL to compile until Step 9 landed; PASS after.

Then change the tick arm at `src/runtime.rs:287-289`:

```rust
                let before = (self.app.transcript.revision(), self.app.spinner_frame());
                self.dispatch(Action::Tick);
                let changed = (self.app.transcript.revision(), self.app.spinner_frame()) != before;
                // An effect repaints without dirtying the transcript or turning the spinner, so
                // the gate has to ask it too. Only bounded effects answer yes: a continuous one
                // would hold the gate open and undo the 30 fps cap.
                return changed || self.effects.in_flight();
```

- [ ] **Step 12: Detect the popup opening**

`App` does not declare this one: eleven sites assign `self.overlay = Some(…)` and none of them would remember a trigger. `Runtime` diffs instead, where it already diffs for the gate.

In `src/runtime.rs`, in the function that dispatches an action (around `:378`, `for effect in self.app.update(action)`), wrap the dispatch:

```rust
    fn dispatch(&mut self, action: Action) {
        let had_overlay = self.app.overlay.is_some();
        for effect in self.app.update(action) {
            self.execute(effect);
        }
        if !had_overlay && self.app.overlay.is_some() {
            self.effects.start(
                crate::ui::effects::Animation::PopupOpened,
                crate::ui::popup_area(self.app.viewport()),
            );
        }
    }
```

`popup_area` does not exist yet. Add it to `src/ui/mod.rs` next to `popup_width` and `centered`, reusing them so the effect covers exactly what the modal covers:

```rust
/// The rect a modal occupies, for the effect layer — the same geometry `centered` gives the
/// modals themselves, so a sweep covers what the popup will cover and nothing else.
pub fn popup_area(area: Rect) -> Rect {
    centered(area, popup_width(area), area.height.saturating_sub(4))
}
```

If `App` has no `viewport()` accessor, use the `viewport` field directly — `src/app.rs` keeps it as `viewport: Rect`; make it `pub` if it is not, and say so in your report.

- [ ] **Step 13: Add the `Animate` effect so Task 3 has it**

In `src/action.rs`, add to the `Effect` enum at `:173`:

```rust
    /// Play a visual effect. Declared here because the moment is the application's to know;
    /// drawing it is `Runtime`'s.
    Animate(crate::ui::effects::Animation),
```

and the arm in `src/runtime.rs`'s `execute` at `:384`:

```rust
            Effect::Animate(animation) => {
                let area = crate::ui::popup_area(self.app.viewport());
                self.effects.start(animation, area);
            }
```

Task 3 narrows that area for the two animations that are not popups. Leaving it as the popup area for now keeps this task's deliverable compiling and visible.

- [ ] **Step 14: Write the failing test for cancellation on resize**

A stored `Rect` is stale the moment the terminal changes size, and a sweep finishing against the
old geometry leaves debris on the new one. The spec requires every running effect to be dropped.

Append to `src/ui/effects.rs`'s `mod tests`:

```rust
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
```

- [ ] **Step 15: Run it, then implement**

Run: `cargo test --lib ui::effects::tests::a_resize_cancels_everything_in_flight`
Expected: FAIL, `no method named cancel_all`.

Add to `impl Effects`:

```rust
    /// Drops every running effect.
    ///
    /// Called on resize: each effect holds the `Rect` it was started on, and that rect no longer
    /// means anything once the terminal changed size.
    pub fn cancel_all(&mut self) {
        self.running.clear();
    }
```

and call it from `Runtime`'s resize handling. The action arrives as `Action::Resize { width, height }` (`src/app.rs:1285`); cancel in `dispatch`, beside the overlay diff, so one place owns reacting to a dispatch:

```rust
    fn dispatch(&mut self, action: Action) {
        let had_overlay = self.app.overlay.is_some();
        let resized = matches!(action, Action::Resize { .. });
        for effect in self.app.update(action) {
            self.execute(effect);
        }
        if resized {
            self.effects.cancel_all();
        } else if !had_overlay && self.app.overlay.is_some() {
            self.effects.start(
                crate::ui::effects::Animation::PopupOpened,
                crate::ui::popup_area(self.app.viewport()),
            );
        }
    }
```

Run: `cargo test --lib ui::effects`
Expected: PASS.

- [ ] **Step 16: Final checks**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green, and `cargo test` shows at least 6 more tests than the 593 the branch starts with. No `unwrap()`/`expect()` outside `#[cfg(test)]` in `src/ui/effects.rs` — check by reading the file above its test module.

---

### Task 2: The transcript's revision comparison

**Files:**
- Modify: `src/transcript.rs:183-228` (`refresh`'s render loop and the `changed` computation)
- Test: appended to the existing `#[cfg(test)] mod tests` in `src/transcript.rs`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: no new names. `Transcript::refresh(&mut self, messages: &[Message], width: usize, context_start: u64) -> bool` and `revision() -> u64` keep their signatures; only *when* `revision` moves changes.

This is the debt J33's review deferred. Today `refresh` counts an entry as rendered whenever it was dirty, so `revision` moves whether or not the lines differ. The redraw gate reads `revision`, so every spurious bump is a repaint of identical pixels. J33 found two bugs of that shape and fixed them one at a time; this kills the class.

- [ ] **Step 1: Write the failing test**

Append to `src/transcript.rs`'s `mod tests`:

```rust
    /// Re-rendering a message that produces the same lines must not move `revision`. The redraw
    /// gate reads `revision`, so a spurious bump is a full repaint of an identical screen — the
    /// shape of two bugs J33's review had to fix one at a time.
    #[test]
    fn re_rendering_identical_content_leaves_the_revision_alone() {
        let messages = vec![
            Message::new(Role::User, "Bonjour"),
            Message::new(Role::Assistant, "Salut !"),
        ];
        let mut transcript = Transcript::default();
        transcript.refresh(&messages, 40, 0);
        let settled = transcript.revision();

        // Nothing changed, but the entry is dirtied as if something had.
        transcript.invalidate(messages[1].id);
        let changed = transcript.refresh(&messages, 40, 0);

        assert_eq!(transcript.revision(), settled, "identical lines, same revision");
        assert!(!changed, "and `refresh` says nothing changed");
    }

    /// The other half: real changes must still move it, or the screen would stop updating.
    #[test]
    fn different_content_still_moves_the_revision() {
        let mut messages = vec![Message::new(Role::Assistant, "Salut")];
        let mut transcript = Transcript::default();
        transcript.refresh(&messages, 40, 0);
        let settled = transcript.revision();

        messages[0].content.push_str(" !");
        transcript.invalidate(messages[0].id);
        let changed = transcript.refresh(&messages, 40, 0);

        assert!(transcript.revision() > settled);
        assert!(changed);
    }
```

If `Message::new(role, content)` is not the constructor this file's other tests use, copy whichever they use — check the top of `mod tests` before writing.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib transcript::tests::re_rendering_identical_content`
Expected: FAIL, `assertion failed: transcript.revision() == settled` — the revision moved.

- [ ] **Step 3: Compare before assigning**

In `src/transcript.rs`, in the render loop that currently ends:

```rust
                entry.lines = lines;
                entry.dirty = false;
                rendered += 1;
```

replace those three lines with:

```rust
                // `rendered` feeds `changed`, which moves `revision`, which lifts the redraw
                // gate. Counting a re-render that produced the same lines repaints an identical
                // screen — so compare, and only count a real difference.
                if entry.lines != lines {
                    entry.lines = lines;
                    rendered += 1;
                }
                entry.dirty = false;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib transcript::`
Expected: PASS, including `different_content_still_moves_the_revision`.

- [ ] **Step 5: Run the whole suite — this changes when the screen repaints**

Run: `cargo test`
Expected: all green. If a test elsewhere fails, read it before changing it: a test that depended on `revision` moving for an identical re-render was depending on the bug. Report any such test and what you did.

Note that `last_rendered()` now reports entries whose lines actually changed. Check its callers with `grep -rn 'last_rendered' src/` and say in your report whether any of them wanted the old meaning.

- [ ] **Step 6: Final checks**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

---

### Task 3: The two declared triggers, and the waiting line's shimmer

**Files:**
- Modify: `src/app.rs:3063-3070` (the `LlmEvent::Token` arm), `:2825-2838` (`clear_context`)
- Modify: `src/ui/effects.rs` (narrow the area per animation)
- Modify: `src/transcript.rs:56-76` (`Waiting::line`)
- Test: appended to `mod tests` in `src/app.rs` and `src/transcript.rs`

**Interfaces:**
- Consumes: from Task 1, `crate::ui::effects::Animation::{FirstToken, ContextCleared}` and `Effect::Animate(Animation)` in `src/action.rs`.
- Produces: no new public names.

- [ ] **Step 1: Write the failing tests for the two triggers**

Append to `src/app.rs`'s `mod tests`:

```rust
    /// The first token is the moment the waiting line gives way to the reply. Later tokens are
    /// not: animating each one would repaint the conversation thirty times a second.
    #[test]
    fn only_the_first_token_animates() {
        let mut app = app();
        let job = send(&mut app, "?");

        let first = app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Token("Bon".into()),
        });
        let second = app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Token("jour".into()),
        });

        assert!(
            first.iter().any(|e| matches!(
                e,
                Effect::Animate(crate::ui::effects::Animation::FirstToken)
            )),
            "the first token animates: {first:?}"
        );
        assert!(
            !second.iter().any(|e| matches!(e, Effect::Animate(_))),
            "the second does not: {second:?}"
        );
    }

    /// `/clear` on an empty context changes nothing and says so. A dissolve over nothing would
    /// be a visible lie.
    #[test]
    fn clearing_an_empty_context_animates_nothing() {
        let mut app = app();

        let effects = submit(&mut app, "/clear");

        assert!(
            !effects.iter().any(|e| matches!(e, Effect::Animate(_))),
            "nothing was dropped, so nothing dissolves: {effects:?}"
        );
        assert!(matches!(&app.status, Status::Info(m) if m.contains("déjà vide")));
    }

    /// And when it does drop something, it animates.
    #[test]
    fn clearing_a_real_context_animates() {
        let mut app = app();
        let job = send(&mut app, "une question");
        llm(&mut app, job.request_id, LlmEvent::Token("une réponse".into()));
        llm(&mut app, job.request_id, LlmEvent::Done);

        let effects = submit(&mut app, "/clear");

        assert!(effects.iter().any(|e| matches!(
            e,
            Effect::Animate(crate::ui::effects::Animation::ContextCleared)
        )));
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib app::tests::only_the_first_token_animates app::tests::clearing_`
Expected: FAIL — `only_the_first_token_animates` with "the first token animates: []".

- [ ] **Step 3: Declare the two animations**

In `src/app.rs`, the `LlmEvent::Token` arm at `:3063`:

```rust
            LlmEvent::Token(token) => {
                self.wait = None;
                let mut effects = Vec::new();
                if let Some(message) = self.conversation.get_mut(generation.message_id) {
                    // The first token is where the waiting line gives way to the reply; the
                    // rest is ordinary streaming and must not repaint the conversation.
                    if message.content.is_empty() {
                        effects.push(Effect::Animate(crate::ui::effects::Animation::FirstToken));
                    }
                    message.content.push_str(&token);
                }
                self.transcript.invalidate(generation.message_id);
                effects
            }
```

This animates a `/compact` summary's first token too — a summary is a reply arriving in the same place, and Review Focus 5 asks for the behaviour to be deliberate rather than accidental. Add the test that pins it:

```rust
    /// A `/compact` summary arrives through the same path and animates the same way: it is a
    /// reply landing where the waiting line was. Pinned so that a future change to summaries is
    /// a decision rather than a surprise.
    #[test]
    fn a_summary_s_first_token_animates_like_any_reply() {
        let mut app = app();
        let job = send(&mut app, "a");
        llm(&mut app, job.request_id, LlmEvent::Token("b".into()));
        llm(&mut app, job.request_id, LlmEvent::Done);
        let effects = submit(&mut app, "/compact");
        let Some(Effect::StartCompletion(summary)) = effects.first() else {
            panic!("a summary job started");
        };

        let first = app.update(Action::Llm {
            request_id: summary.request_id,
            event: LlmEvent::Token("résumé".into()),
        });

        assert!(first.iter().any(|e| matches!(
            e,
            Effect::Animate(crate::ui::effects::Animation::FirstToken)
        )));
    }
```

In `clear_context` at `:2825`, after `effects.extend(self.move_context_start(start));`:

```rust
        effects.push(Effect::Animate(crate::ui::effects::Animation::ContextCleared));
```

The early return for an empty context is already above it, so an empty `/clear` animates nothing without any extra condition.

- [ ] **Step 4: Run them to verify they pass**

Run: `cargo test --lib app::`
Expected: PASS.

- [ ] **Step 5: Give each animation its own area**

Task 1 left every animation covering the popup rect. In `src/runtime.rs`'s `Effect::Animate` arm, choose by animation:

```rust
            Effect::Animate(animation) => {
                let viewport = self.app.viewport();
                let area = match animation {
                    crate::ui::effects::Animation::PopupOpened => crate::ui::popup_area(viewport),
                    // The conversation area: where a reply appears, and where cleared context
                    // disappears from.
                    crate::ui::effects::Animation::FirstToken
                    | crate::ui::effects::Animation::ContextCleared => {
                        crate::layout::chat_area(viewport)
                    }
                };
                self.effects.start(animation, area);
            }
```

`crate::layout` already computes the chat area for `ui::render`. Read `src/layout.rs` and use whatever function or field gives the conversation's rect; if it is not public, make it public rather than recomputing the layout here, and say so in your report.

- [ ] **Step 6: Write the failing test for the shimmer**

Append to `src/transcript.rs`'s `mod tests`:

```rust
    /// The waiting line's Braille glyph is replaced by a travelling highlight. Braille is the
    /// one block a terminal font is most likely to lack, and the highlight says "working"
    /// without needing it.
    #[test]
    fn the_waiting_line_shimmers_without_a_braille_glyph() {
        let text = |frame: usize| -> String {
            Waiting {
                phase: crate::llm::Phase::Waiting {
                    model: "llama3.2".into(),
                },
                frame,
                elapsed_s: None,
            }
            .line()
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect()
        };

        for frame in 0..10 {
            assert!(
                !text(frame).chars().any(|c| ('\u{2800}'..='\u{28FF}').contains(&c)),
                "no Braille at frame {frame}: {}",
                text(frame)
            );
        }
    }

    /// And it has to move, or it is not a shimmer. The styles differ between frames even though
    /// the text does not.
    #[test]
    fn the_shimmer_advances_with_the_frame() {
        let styles = |frame: usize| -> Vec<ratatui::style::Style> {
            Waiting {
                phase: crate::llm::Phase::Loading,
                frame,
                elapsed_s: None,
            }
            .line()
            .spans
            .iter()
            .map(|s| s.style)
            .collect()
        };

        assert_eq!(
            styles(0).len(),
            styles(3).len(),
            "the same text, cut the same way"
        );
        assert_ne!(styles(0), styles(3), "but lit differently");
    }
```

- [ ] **Step 7: Run them to verify they fail**

Run: `cargo test --lib transcript::tests::the_waiting_line_shimmers transcript::tests::the_shimmer_advances`
Expected: FAIL — the Braille assertion fails at every frame, since `SPINNER` is Braille.

- [ ] **Step 8: Build the shimmer in `Waiting::line`**

Replace the two lines in `src/transcript.rs:70-71` that build the glyph and the first span:

```rust
        let glyph = SPINNER[self.frame % SPINNER.len()];
        let mut spans = vec![Span::styled(format!("{glyph} {what}"), dim)];
```

with:

```rust
        // A highlight travelling through the label, rather than a spinning glyph. It rides the
        // frame counter the spinner already turned on, so it costs no extra redraw — and it
        // needs no Braille, the block a terminal font is most likely to be missing.
        let mut spans = shimmer(&what, self.frame);
```

and add, next to `Waiting`:

```rust
/// Cuts `text` into spans with one lit character, moving with `frame`.
///
/// The lit character is drawn in the accent colour over the dim rest, so the line reads as
/// working without the eye being pulled away from the conversation.
fn shimmer(text: &str, frame: usize) -> Vec<Span<'static>> {
    let dim = Style::default().fg(crate::theme::palette().dim);
    let accent = Style::default().fg(crate::theme::palette().accent);
    let characters: Vec<char> = text.chars().collect();
    if characters.is_empty() {
        return Vec::new();
    }
    let lit = frame % characters.len();
    let piece = |range: std::ops::Range<usize>, style: Style| {
        Span::styled(characters[range].iter().collect::<String>(), style)
    };
    let mut spans = Vec::new();
    if lit > 0 {
        spans.push(piece(0..lit, dim));
    }
    spans.push(piece(lit..lit + 1, accent));
    if lit + 1 < characters.len() {
        spans.push(piece(lit + 1..characters.len(), dim));
    }
    spans
}
```

`SPINNER` is still used by the tool card at `src/transcript.rs:466` — leave that one alone; it is a different affordance and out of this milestone's scope. Say in your report whether leaving it looks inconsistent on screen.

- [ ] **Step 9: Run them to verify they pass**

Run: `cargo test --lib transcript::`
Expected: PASS.

- [ ] **Step 10: Accept the snapshots, one at a time**

The waiting line appears in several insta snapshots. `cargo-insta` is **not installed**: accept each one individually and read it before you do.

```bash
cargo test --lib 2>&1 | grep -E '^test .* FAILED|snapshot'
INSTA_UPDATE=always cargo test --lib <exact test name>
cat src/ui/snapshots/<the file>.snap
```

Read every accepted snapshot. A snapshot captures characters, not colour, so what you are checking is that the *text* of the waiting line is what you expect and that nothing else moved. Never run `INSTA_UPDATE` over the whole suite.

- [ ] **Step 11: Final checks**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

---

### Task 4: Documentation, the dependency gate, and the manual check

**Files:**
- Modify: `README.md`, `PLAN.md`, `docs/roadmap.md`
- Test: none new; this task runs the gates

**Interfaces:**
- Consumes: everything the previous three tasks produced.
- Produces: a green suite, and a `cargo tree` with six `*-sys` crates and no more.

- [ ] **Step 1: The dependency gate**

```bash
cargo tree | grep -oE '[a-z0-9_-]+[-_]sys v[0-9.]+' | sort -u
```

Expected, exactly: `aws-lc-sys`, `dirs-sys`, `inotify-sys`, `libsqlite3-sys`, `linux-raw-sys`, `onig_sys`. Six. Use this command, not `grep -i -- '-sys'`, which cannot match an underscore and reported a green gate while a C library was linked — twice, during J34.

```bash
cargo tree -i tachyonfx
```

Expected: tachyonfx appears once, pulled only by `chatatui`.

- [ ] **Step 2: Document the effects**

In `README.md`, in the section describing the interface, add a short paragraph in English:

```markdown
**Motion.** Four moments are animated, briefly: a popup sweeps in, the first token of a reply
coalesces where the waiting line stood, `/clear` dissolves the context it drops, and a highlight
travels through the waiting line instead of a spinning glyph. Each lasts under a quarter of a
second, and nothing animates continuously — the renderer only redraws a frame when the screen
actually changed, and an effect is the only thing allowed to lift that rule, for as long as it
runs.
```

In `PLAN.md`, add J35 to the milestone list in the file's existing style, and add `effects.rs` to the `ui/` entry of the module map. In `docs/roadmap.md`, move the tachyonfx line out of `Next` and into the shipped list.

- [ ] **Step 3: Run everything**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Expected: all green.

- [ ] **Step 4: The manual check this milestone cannot automate**

Not automatable, and not performed by you. Report it as the user's to do, with this list — the durations in particular cannot be judged any other way:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release && ./target/release/chatatui
```

1. `F2` and `Ctrl+P`: the popup sweeps in. Does 150 ms read as motion, or as lag? Press F2 twice
   quickly — one sweep, not two overlapping.
2. Send a message: at the first token the reply coalesces where the waiting line stood. 120 ms.
3. Watch the waiting line: the highlight travels at about 10 Hz. Stuttery or alive?
4. `/clear` with a real context: the dropped lines dissolve. 200 ms. Then `/clear` again on the
   now-empty context: nothing animates.
5. Resize the terminal while a popup is open, and during a reply. Nothing should streak or
   leave debris.
6. A terminal small enough that a popup barely fits: no panic, and no effect is better than a
   broken one.
