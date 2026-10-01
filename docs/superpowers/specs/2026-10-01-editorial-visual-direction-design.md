# J33 — Editorial visual direction, waiting states included

## Why

Two complaints, one answer.

The interface is tidy but anonymous, and its dark theme is not really a theme: `DARK`
names the eight basic ANSI colours (`Color::Cyan`, `Color::Magenta`, `Color::Yellow`),
whose actual tones come from whatever palette the user's terminal happens to use. The same
build looks different — and sometimes clashes — under Gruvbox, Dracula or Solarized. The
light palette, by contrast, pins exact indexed tones. One half of the product was designed
and the other was inherited.

And the application goes silent exactly when it is working hardest. `start_job_with`
pushes an empty assistant message, the screen shows a cursor and `◐ Génération…`, and then
**nothing changes until the first token arrives** — no pixel, no redraw. `◐` is a fixed
glyph; nothing anywhere cycles it. On a local 8B that is several seconds of a screen that
reads as hung. Retrieval runs before the first HTTP byte (`stream_task.rs:137`) behind the
same still screen, and a tool that is actually executing still says
`en attente de votre accord…` — the user approved it long ago.

These are not two projects. "What the application looks like while it waits" is a question
of visual vocabulary, and the plumbing under it is thin. Doing them separately would mean
drawing the status bar and the transcript twice.

## Scope

In:

- A new visual vocabulary for the conversation — role by position instead of by label,
  whitespace instead of chrome.
- Both palettes pinned to indexed colours, reduced to monochrome plus one accent.
- A live waiting line in the conversation, animated, naming the phase and its elapsed time.
- A tool card that distinguishes *awaiting approval* from *running*.
- The status bar's spinner actually turning.

Out, deliberately:

- **Reasoning tokens.** `reasoning_content`, `reasoning`, `thinking` and `thinking_delta`
  are dropped today (`openai.rs:282`, `anthropic.rs:284`, the latter asserted by
  `thinking_and_unknown_events_are_ignored`). Displaying them is a new content type across
  both clients, `stream_task`, `App`, `Message` and the transcript. Its own milestone.
- **A per-message metadata line** (model, duration, speed). Rejected: it would cost a line
  per exchange. `/context` keeps carrying the detail.
- **Progress inside retrieval** (extraction, embedding, rerank). Would require passing a
  `Sender` into the `ContextProvider` trait, which sits behind `Arc<dyn …>`. The phase
  reported around the await says `recherche dans N collections…`, which is enough for a
  step that rarely exceeds a second.
- **The open code block re-highlighted at every tick.** `render.rs:486` sends an unclosed
  fence to `highlight()` uncached, on purpose, so the cache is not filled with partial
  versions — up to 30 times a second over the whole block. A performance question, not a
  visual one; noted, not fixed here.

## The vocabulary

Role is read from **position**. The `▌ Vous` and `▌ Assistant` headers go away.

```
        Comment trier un Vec ?


  Tri en Rust
  ───────────

  Utilise sort_unstable si l'ordre des égaux n'importe pas :

    rust                                                  [1]
      let mut v = vec![3, 1, 2];
      v.sort_unstable();

  · plus rapide
  · pas d'allocation


        Et pour trier à l'envers ?
```

| Element | Column | Treatment |
|---|---|---|
| The user's message | 8 | `dim` |
| The assistant's reply | 2 | default foreground |
| Heading | 2 | text, then a `─` rule the width of the text, `dim` |
| List item | 4 (`·`), text at 6 | marker `dim` |
| Block quote | 4 (`│`), text at 6 | marker `dim` |
| Code block | 6 | language and `[n]` on a `dim` line above |
| Rare roles | 2 | a `dim` label line (see below) |

Two blank lines between turns, one inside a turn. The rhythm replaces the frames.

**The code block loses its per-line gutter but keeps what it is for.** The language and the
block's number sit on one dim line above it, because `/copy code 2` is unusable if the
numbers are invisible. The number is shown only when the reply holds several blocks, as
`render.rs:43` already decides. syntect highlighting is unchanged.

### The roles that cannot be a position

`Système`, `Fichier joint`, `Résumé de la conversation` and `Outil` keep a dim label, as an
acknowledged exception: they are rare, and expressing four more roles through indentation
would be unreadable.

```
  fichier joint · cours/ch04-ownership.pdf
  🔧 read_file · ≈ 340 tokens transmis
```

The out-of-context separator and the version markers (`‹ 2/3 ›`) keep their current places
and become `dim`.

## The waiting states

The waiting line sits **where the reply will appear** — column 2, `dim` — and the text
replaces it in place:

```
        Résume-moi le cours sur l'ownership


  ⠻ recherche dans 2 collections…
```

then

```
  ⠼ llama3.2 réfléchit…  3 s
```

then

```
  Chaque valeur a un seul propriétaire▍
```

Four phases, in the order they occur — and they are **not all drawn in the same place**:

| Phase | Text | Reported | Drawn |
|---|---|---|---|
| `Connecting` | `connexion…` | when the job starts | waiting line |
| `Retrieving { collections }` | `recherche dans 2 collections…`, or `recherche dans les documents…` when the count is 1 | before `provide().await` | waiting line |
| `Waiting { model }` | `llama3.2 réfléchit…` | after `provide()` returns | waiting line |
| `RunningTool { name }` | `exécution de read_file…` | when a tool starts running | the tool card |

`RunningTool` is the exception, and it has to be. `on_tool_call` (`app.rs:2881`) closes the
partial reply — `status = Complete`, or truncates it when empty — pushes a `Role::Tool`
card, and only pushes a fresh `Streaming` message when the result comes back. So while a
tool runs **there is no empty streaming message at all**, and a waiting line would have
nothing to attach to. The phase is drawn on the card instead, which is where the user is
already looking.

That also fixes a wrong label. `transcript.rs:302` prints `en attente de votre accord…` for
`MessageStatus::Streaming`, which today covers both waiting for the user *and* executing.
The card says `en attente de votre accord…` only while the confirmation overlay is open,
and `⠼ exécution…` once the call is actually running.

The elapsed time appears on the waiting line **only past one second**, so a fast reply never
flashes a number, and is rendered in whole seconds (`3 s`). The tool card carries the
spinner but no timer: a tool's duration is not something the user can act on.

The status bar does not repeat any of this. It keeps `Génération…`, the model and the
gauge, with its `◐` cycling through `◐◓◑◒`.

### How the spinner actually turns

This is the real cost, and it is not obvious. `runtime.rs:278` redraws on a tick **only if
`transcript.revision()` changed**, and `update` skips `refresh_view()` for a token
(`app.rs:739`) so that markdown work is capped at the tick rate. During a wait there are no
tokens, nothing is dirty, `refresh` changes nothing, and so nothing redraws. A naive
animated glyph would sit as still as today's `◐`.

The fix follows an existing pattern exactly. `Transcript` already owns `trailer`, a line
vector compared on every refresh whose change bumps `revision` (`transcript.rs:144-148`),
and `set_marks` already marks the affected entries dirty when the markers change
(`transcript.rs:46-60`). The waiting line gets the same shape:

```rust
/// What the application is doing while no token has arrived yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waiting {
    pub phase: Phase,
    /// Spinner frame, already reduced from the tick count.
    pub frame: usize,
    /// Whole seconds waited; `None` under one second.
    pub elapsed_s: Option<u32>,
}

impl Transcript {
    /// Sets the waiting line; the streaming message is re-rendered when it changes.
    pub fn set_waiting(&mut self, waiting: Option<Waiting>);
}
```

**Time comes from the tick count, not from a clock.** `App::update` is pure and reads no
clock anywhere — elapsed time already enters through `LlmEvent::Timing`, which the stream
task only emits inside the token loop (`stream_task.rs:186`), every 500 ms. Extending that
to cover the wait would mean wrapping three separate `await`s in `select!`/`interval`, and
a 500 ms feed would animate a spinner at 2 fps. So instead `App` keeps one counter,
`waiting_ticks: u32`, incremented on `Action::Tick` while a wait is in progress, and derives
both numbers from it at the known tick rate (`TICK_FPS = 30.0`, `event.rs:21`):

```rust
frame     = (ticks / 3) % FRAMES.len()      // ~10 frames per second
elapsed_s = (ticks >= 30).then(|| ticks / 30)
```

The counter is deliberately **not** part of `Waiting`. Only the derived values are, so
`set_waiting` sees a change every third tick rather than every tick: the line is re-rendered
and redrawn about ten times a second, and the 30 fps markdown cap is preserved. Deriving
from ticks also makes the whole thing deterministic in tests — dispatch thirty `Tick`s and
the line must read `1 s` — with no injected clock.

A tick-derived second under-reports if the runtime is busy enough to delay the interval.
That is accepted: the number is a reassurance, not a measurement, and `/context` carries the
real timing from `LlmEvent::Timing` as it does today.

A waiting line is shown only for the streaming message whose content is still empty. At the
first token the content is no longer empty, `App` clears `waiting`, and the entry re-renders
as text. Nothing is inserted into or removed from the message list, so scroll position and
`Find` are untouched.

### Reporting the phases

`stream_task` already awaits retrieval at `stream_task.rs:137` and already owns a sender to
the app. A new `LlmEvent::Phase(Phase)` is emitted around the awaits it already performs:

```rust
pub enum Phase {
    Connecting,
    Retrieving { collections: usize },
    Waiting { model: String },
    RunningTool { name: String },
}
```

Nothing in the `ContextProvider` or `LlmClient` traits changes. `App::on_llm_event` treats
`Phase` like the other events — it sets `self.waiting` and returns no effect — and the
stale-`request_id` guard that already discards late events covers it unchanged.

## The colours

The principle: **monochrome plus one accent.** Body text in the terminal's default
foreground, structure in `dim`, one accent for interaction and selection. `warn` and
`error` keep their own.

The semantic field names of `Palette` do not change — every call site keeps working. Only
the values do, and **both palettes pin indexed tones**:

```rust
pub const DARK: Palette = Palette {
    dim: Color::Indexed(245),          // was DarkGray
    accent: Color::Indexed(110),       // was Cyan
    warn: Color::Indexed(179),         // was Yellow
    error: Color::Indexed(167),        // was Red
    ok: Color::Indexed(108),           // was Green
    info: Color::Indexed(109),         // was Blue
    assistant: Color::Indexed(110),    // was Magenta
    selection_bg: Color::Indexed(238), // was DarkGray
    bar_bg: Color::Indexed(234),       // was Black
    badge_fg: Color::Indexed(234),
    badge_bg: Color::Indexed(245),
    code_theme: "base16-ocean.dark",
};
```

`LIGHT` is already indexed and mostly stands; `assistant` moves to the accent so that the
two palettes carry the same reduced vocabulary, and `dim` goes to 245 for contrast against
white.

`assistant` survives as a field because `status_bar.rs:181` and `prompt_view.rs:50` still
use it; the role header that was its main consumer is gone.

## Architecture

The control flow does not change. Everything here is either a palette value, a rendering
decision, or one new `LlmEvent` variant travelling the existing channel.

```
LlmEvent::Phase → AppEvent::Llm → App::update → App.waiting
                                              → Transcript::set_waiting
                                              → revision bumps → Runtime redraws
Action::Tick    → App advances `frame` every 80 ms → same path
```

`App::update` stays pure; the rendering stays read-only.

Files, and what each owns:

| File | Change |
|---|---|
| `src/theme.rs` | both palettes pinned to indexed tones |
| `src/transcript.rs` | positional layout, `Waiting`, `set_waiting`, the rare-role labels, the tool card's two states |
| `src/markdown/render.rs` | headings with a rule, `·` lists, code blocks without a gutter |
| `src/llm/mod.rs` | `Phase`, `LlmEvent::Phase` |
| `src/llm/stream_task.rs` | emit the four phases around the awaits already there |
| `src/app.rs` | `waiting`, `waiting_ticks`, `on_llm_event` handling |
| `src/ui/chat.rs` | margins |
| `src/ui/prompt_view.rs` | input chrome reduced to a single rule |
| `src/ui/status_bar.rs` | `◐◓◑◒` cycling |
| `src/ui/snapshots/*` | all thirty, re-read and re-accepted one by one |

## Errors

Nothing here can fail in a new way. A phase that never arrives leaves the previous one on
screen, which is what the user already sees today. A phase arriving after `Done` is
discarded by the existing `request_id` guard. `Waiting` is cleared by `Done`, `Error`,
cancellation, and by the first token.

## Testing

Unit tests:

- `theme.rs`: every colour of both palettes is `Color::Indexed`, so the regression to a
  terminal-dependent basic colour is caught mechanically.
- `transcript.rs`: a user message is indented and dim; an assistant message is not; the
  four rare roles keep their label; a waiting line appears only for an empty streaming
  message; `set_waiting` bumps `revision` when the frame changes and does not when it does
  not; the tool card says `exécution` while running and `en attente de votre accord` while
  the overlay is open.
- `markdown/render.rs`: a heading is followed by a rule the width of its text; a single code
  block carries no number and several do; the gutter is gone.
- `app.rs`: thirty `Tick`s produce `elapsed_s == Some(1)` and ten frame changes, not thirty;
  the first token clears `waiting`; `Done`, `Error` and cancellation clear it.
- `stream_task.rs`: the four phases are emitted in order, `Retrieving` before `provide` and
  `Waiting` after.

Snapshots. The thirty existing ones all change; each is re-read before being accepted, not
accepted in bulk. Four are added for states nothing covers today:

- the wait before the first token, with its phase and elapsed time;
- a retrieval wait naming the collection count;
- a tool actually running, as opposed to awaiting approval;
- a code block still open while it streams.

## Accepted limitation

The editorial direction loses scanning landmarks, and no per-message metadata was added
back. In a fifty-message conversation, finding your own question rests entirely on
indentation and the dim colour. `Ctrl+F` and the conversation list's preview soften it.
This is a deliberate choice; if it proves tiring in use, the remedy is a dim rule between
turns, not a return to role labels.

## Task breakdown

| # | Task | Files | Depends on |
|---|---|---|---|
| 1 | Palettes pinned and reduced | `theme.rs` | — |
| 2 | `Phase`, `LlmEvent::Phase`, emission | `llm/mod.rs`, `llm/stream_task.rs` | — |
| 3 | Positional layout and the rare roles | `transcript.rs`, `markdown/render.rs` | 1 |
| 4 | `Waiting`, the tick counter, the tool card's two states | `transcript.rs`, `app.rs`, `ui/status_bar.rs` | 2, 3 |
| 5 | Margins, input chrome, the thirty snapshots, docs | `ui/chat.rs`, `ui/prompt_view.rs`, `ui/snapshots/*`, `README.md`, `PLAN.md`, `docs/roadmap.md` | 3, 4 |

Tasks 1 and 2 share no file and can run at once. Task 3 needs the palette, task 4 needs
both, task 5 comes last because the snapshots must be written against the final rendering.
