# J35 — Bounded visual effects with tachyonfx: design

## Why

J33 gave the interface an editorial direction: position says who speaks, a rule replaces a
colour, the waiting line stands where the reply will appear. What it has no vocabulary for is
*change*. A popup appears between two frames with nothing to say it arrived; the waiting line
is replaced by the first token with no seam; `/clear` drops the context and the screen simply
has less in it afterwards.

This milestone adds motion to four of those moments. Three of them decorate; one of them —
`/clear` — carries information the interface currently loses.

It also pays a debt J33's review deferred: `Transcript::refresh` bumps `revision` whenever it
re-renders, whether or not the result differs. That is the redraw gate's one weakness, and
effects are about to depend on that gate being trustworthy.

## Scope

**In:**

- Four effects: popup opening, first token arriving, `/clear`, and the waiting line's glyph.
- A `Runtime`-owned effect layer applied after `ui::render`.
- `Effect::Animate(Animation)` as the trigger idiom, matching how this repository already
  declares work in `App` and executes it in `Runtime`.
- The redraw gate learning about effects in flight.
- `Transcript::refresh` comparing rendered lines before bumping `revision`.

**Out:**

- Any effect not in the list of four. No ambient decoration, no effects on scrolling, typing,
  or message arrival beyond the first token.
- Configuration. The effects are either right or they are removed; a toggle would mean two
  interfaces to look after. (Revisit only if someone's terminal cannot keep up.)
- Reduced-motion detection. No terminal reports it.

## What already exists, precisely

- `src/runtime.rs:253` — `terminal.draw(|frame| ui::render(&self.app, frame))?`. One draw call,
  one closure, the only place a frame is produced.
- `src/runtime.rs:281-290` — the redraw gate. On `Event::Tick` it captures
  `(self.app.transcript.revision(), self.app.spinner_frame())`, dispatches `Action::Tick`, and
  returns whether the tuple changed. This is the 30 fps markdown cap: `update` deliberately
  does not dirty the transcript per token, and the spinner is checked separately so it can keep
  turning while the transcript stays clean.
- `App::update(Action) -> Vec<Effect>` is pure: no I/O, **no clock**. `Runtime` executes the
  effects. `ui::render(&App, &mut Frame)` is read-only.
- `src/transcript.rs` — `revision()` is the gate's input; `refresh(&[Message], width,
  context_start) -> bool` re-renders dirty entries.
- The waiting line already redraws at about 10 Hz, because `spinner_frame()` changes and the
  gate notices. **This is the only state in the application that ticks visibly today.**

## The tachyonfx API this design relies on

Read from the vendored source of 0.25.2, not from memory:

- `impl EffectRenderer<Effect> for Frame<'_>` — `frame.render_effect(&mut effect, area: Rect,
  last_tick: Duration)`. The hook takes a `Frame`, so the effect layer needs no access to the
  buffer directly.
- `Effect::done() -> bool`, `Effect::running() -> bool`.
- `tachyonfx::Duration` is tachyonfx's own millisecond-resolution type, not `std::time::Duration`.
- `Effect` derives `Debug` and **not `Clone`**: every trigger constructs a fresh effect. There
  is no template to copy.
- Constructors, exact signatures:
  - `fx::sweep_in<T: Into<EffectTimer>, C: Into<Color>>(direction: Motion, gradient_length: u16,
    randomness: u16, faded_color: C, timer: T) -> Effect`
  - `fx::coalesce<T: Into<EffectTimer>>(timer: T) -> Effect`
  - `fx::dissolve<T: Into<EffectTimer>>(timer: T) -> Effect`
  - `fx::fade_from_fg<T: Into<EffectTimer>, C: Into<Color>>(fg: C, timer: T) -> Effect`
  - `fx::hsl_shift_fg<T: Into<EffectTimer>>(hsl_fg_change: [f32; 3], timer: T) -> Effect`
  - `fx::repeating(effect: Effect) -> Effect`, `fx::ping_pong`, `fx::parallel(&[Effect])`,
    `fx::sequence(&[Effect])`
- `Motion` is `LeftToRight | RightToLeft | UpToDown | DownToUp`.

tachyonfx 0.25.2 targets ratatui 0.30.2, which is what `Cargo.lock` holds.

## Approach

`Runtime` owns the effects, because `Runtime` owns the clock. The draw closure becomes:

```rust
let elapsed = self.effects.tick();              // wall time since the previous frame
terminal.draw(|frame| {
    ui::render(&self.app, frame);
    self.effects.render(frame, elapsed);
})?;
```

Two invariants are preserved by construction rather than by care: `App` never sees a clock
because the elapsed time is measured here, and `ui::render` stays read-only because the effects
run after it and write to the frame, not to `App`.

Triggers use the idiom the repository already has. `App::update` returns `Vec<Effect>` and
`Runtime` executes them, so a visual animation is one more kind of effect:

```rust
Effect::Animate(Animation::FirstToken)      // the first token of a reply arrived
Effect::Animate(Animation::ContextCleared)  // /clear dropped the context
```

One line each, in the arm that already handles the action — `clear_context` already returns
`Vec<Effect>`. A reviewer reading `app.rs` sees which actions animate without opening the effect
layer.

### The popup sweep is the exception: `Runtime` detects it

Eleven sites in `app.rs` assign `self.overlay = Some(…)` and there is no helper to put a trigger
in. Declaring the animation at each would be eleven lines to forget at the twelfth overlay.

`Runtime` already diffs `App` across a dispatch — that is what the redraw gate does. Comparing
`app.overlay().is_some()` before and after covers all eleven sites and every overlay added
later, in one place. The intent being reconstructed here is trivial and total: an overlay exists
now that did not before.

So the rule is by cardinality, not by principle: one intentional site declares its animation,
many incidental sites are observed.

## Components

### `src/ui/effects.rs` (new)

```rust
/// A visual effect the application asked for, independent of how it is drawn.
pub enum Animation { PopupOpened, FirstToken, ContextCleared }

/// The effects in flight, and the clock that drives them.
pub struct Effects { /* … */ }

impl Effects {
    pub fn new() -> Self;
    /// Elapsed time since the previous frame, and the value `render` consumes.
    pub fn tick(&mut self) -> Duration;
    /// Starts `animation` over `area`. Replaces an effect of the same kind already running.
    pub fn start(&mut self, animation: Animation, area: Rect);
    /// Draws every running effect and drops the finished ones.
    pub fn render(&mut self, frame: &mut Frame, elapsed: Duration);
    /// Whether a bounded effect is running — the redraw gate's extra condition.
    pub fn in_flight(&self) -> bool;
}
```

`start` taking the `Rect` is what keeps `App` out of layout: `Runtime` knows the popup's area
from the same layout `ui::render` uses.

### The four effects

| Moment | Effect | Duration |
|---|---|---|
| A popup opens | `fx::sweep_in(Motion::UpToDown, …, palette().bar_bg, …)` over the popup's rect | 150 ms |
| The first token arrives | `fx::coalesce` over the reply's first line | 120 ms |
| `/clear` runs | `fx::dissolve` over the lines leaving the context | 200 ms |
| Waiting | not a tachyonfx effect — see below | continuous |

Durations are the starting point, not a measurement: they are read on screen during the manual
check and adjusted once. Anything above 250 ms makes the interface feel slow rather than alive.

### The waiting line does not use tachyonfx, and that is the design

Two things rule it out. The effect layer needs a `Rect`, and the waiting line's position depends
on the scroll offset — only `ui::render` knows it, and handing it back would mean changing
`ui::render`'s signature to carry layout out of the renderer. And a shimmer is *continuous*:
`fx::repeating` never reports `done()`, so `in_flight()` would either have to special-case it or
lift the gate forever, which is the regression the 30 fps cap exists to prevent.

Both disappear if the shimmer is built where the line is built. `Waiting::line` in
`src/transcript.rs` already assembles the spans and already holds `frame`, the counter the
spinner turns on. A shimmer is a colour ramp across those spans, offset by `frame` — about
fifteen lines, no rect, no effect state, no gate change, and the Braille glyph goes away with
it, which is what fixes terminals whose font lacks that block.

It advances at the spinner's cadence, roughly 10 Hz, because that is what already redraws. If
that looks stuttery on screen the fix is a finer counter for the waiting state alone, tripling
redraws during waits only. **Do not take that step without looking at the 10 Hz version first.**

`in_flight()` therefore counts bounded effects only, and nothing continuous ever enters it.

### The redraw gate

```rust
let before = (self.app.transcript.revision(), self.app.spinner_frame());
self.dispatch(Action::Tick);
(self.app.transcript.revision(), self.app.spinner_frame()) != before || self.effects.in_flight()
```

### `Transcript::refresh` — the J33 debt

Today `refresh` bumps `revision` whenever it re-renders an entry, whether or not the lines
changed. J33's review found two bugs of that shape and fixed them one at a time; the review
reported that comparing the rendered lines kills the class instead. It lands here because
effects are about to trust the gate, and because debugging two redraw mechanisms at once is
worse than debugging either.

After: an entry whose re-render produces identical lines leaves `revision` alone.

## Error handling

Effects cannot fail in a way worth reporting: tachyonfx writes into a buffer that already holds
a complete frame, so the worst case is a frame that looks wrong for 150 ms. Two cases are
guarded rather than reported:

- **A zero-sized or out-of-bounds area.** `start` drops the request if `area` is empty. A popup
  on a terminal too small to show it should not animate.
- **A resize mid-effect.** The stored `Rect` is stale after a resize. `Action::Resize` clears
  every running effect: a sweep finishing against the old geometry is worse than no sweep.

## Testing

Every effect test runs on a `TestBackend` buffer, and asserts behaviour rather than appearance —
insta snapshots capture characters only and would see nothing of a colour fade.

- **An effect changes the frame and then stops changing it.** Render once with no effect, keep
  the buffer. Start the effect, render mid-way, assert the buffer differs. Advance past the
  duration, render, assert the buffer equals the first one. This is the test that proves effects
  are *bounded* — the property the whole design rests on.
- **The gate lifts for a bounded effect and drops again.** A tick redraws while an effect runs
  and stops once it is done.
- **The shimmer never lifts the gate.** With only the waiting-line effect running,
  `in_flight()` is false. Written as a test because getting this wrong is invisible: the
  interface looks fine and burns a core.
- **A resize cancels running effects.**
- **`revision` does not move when the rendered lines are identical** — the J33 debt, pinned.
- **An empty area starts nothing.**

## Dependencies

`tachyonfx = "0.25.2"`. Net new crates in the tree: `tachyonfx`, `bon`, `micromath`
(`ratatui-core`, `compact_str`, `unicode-width` and `thiserror` are already there). The gate
this milestone checks: nothing else appears, and no `*-sys` crate is added — the six already in
the tree stay six.

## Accepted limitations

- **The shimmer advances at 10 Hz**, the spinner's cadence, and is span styling rather than a
  tachyonfx effect. Smoother needs a finer counter and more redraws during waits.
- **No way to turn effects off.** If one of the four is wrong, it is removed rather than
  configured.
- **Durations are guesses until seen.** They are tuned once, on screen, during the manual check.
- **Effects are invisible to the test suite's snapshots.** They are tested through buffer
  comparison, never through insta.

## Tasks

1. **The effect layer and the gate.** `src/ui/effects.rs`, the draw closure, `in_flight()` in
   the tick arm, the resize cancellation. One effect — the popup sweep — to prove the path end
   to end — including its trigger, so `Effect::Animate` and the `Animation` enum land here.
   Files: `Cargo.toml`, `src/ui/effects.rs`, `src/runtime.rs`, `src/app.rs` (the `Effect` enum
   and the overlay arm).
2. **The transcript's revision comparison.** The J33 debt, standalone and testable on its own.
   Files: `src/transcript.rs`.
3. **The three remaining effects and their triggers.** `Effect::Animate` and its two arms in
   `app.rs` (first token, `/clear`), their effects, and the waiting line's shimmer — which is
   span styling in `Waiting::line`, not an effect. Files: `src/app.rs`, `src/ui/effects.rs`,
   `src/transcript.rs`.
4. **Documentation and the manual check.** `README.md`, `PLAN.md`, `docs/roadmap.md`, the
   dependency gate, and the list of what to look at on screen — durations included, since they
   cannot be judged any other way.

Tasks 1 and 2 share no file and can run together. Task 3 needs Task 1's layer. Task 4 is last.
