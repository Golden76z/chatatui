# J33 — Editorial visual direction: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the conversation's label-and-chrome layout with a positional, whitespace-led one, pin both colour palettes to exact tones, and make the application say what it is doing while it waits instead of freezing.

**Architecture:** Nothing moves in the control flow. `App::update(Action) -> Vec<Effect>` stays pure and reads no clock; the renderer stays read-only. One new `LlmEvent` variant (`Phase`) travels the channel that already exists, and the animated waiting line reuses the `set_marks` pattern: `App` hands the transcript a value that only changes every third tick, so `revision` — and therefore the redraw — moves about ten times a second while the 30 fps markdown cap is preserved.

**Tech Stack:** Rust 2024 (`rust-version = "1.88"`), ratatui 0.30, pulldown-cmark, syntect, insta for snapshots, tokio.

**Spec:** `docs/superpowers/specs/2026-10-01-editorial-visual-direction-design.md`

## Execution Order

```
Task 1 (palettes)   ┐
Task 2 (Phase)      ┘ in parallel, disjoint files
   └── Task 3 (positional layout + markdown)
          └── Task 4 (Waiting + tick counter + tool card)
                 └── Task 5 (margins, input chrome, the thirty snapshots, docs)
```

Tasks 1 and 2 share no file: Task 1 owns `src/theme.rs`, Task 2 owns `src/llm/mod.rs` and
`src/llm/stream_task.rs`. Task 3 needs the palette. Task 4 needs both. Task 5 comes last
because the snapshots can only be written against the final rendering — writing them earlier
means writing them twice.

**Snapshots are a Task 5 concern on purpose.** Tasks 3 and 4 change the rendering, so the
thirty existing snapshots will fail from Task 3 onward. Tasks 3 and 4 run
`cargo test --lib transcript`, `--lib markdown`, `--lib app` and `--lib theme` — not the
whole suite — and Task 5 is where the suite goes green again. Each task says exactly which
command it must pass.

## Global Constraints

- Rust edition 2024, `rust-version = "1.88"`. No `unwrap()` or `expect()` outside `#[cfg(test)]` code.
- UI strings in French; code, comments and doc comments in English.
- **No new dependency.** Not one.
- **`App::update` stays pure: no I/O, and no clock.** Elapsed time is derived from the tick count, never from `Instant::now()` inside `App`. This is what makes the waiting line deterministic in tests.
- `ui::render(&App, frame)` stays read-only; a view never mutates state.
- **The 30 fps markdown cap must survive.** `update` skips `refresh_view()` for a token (`app.rs:739`) and `runtime.rs:278` only redraws when `transcript.revision()` changed. Any change that makes the transcript dirty on every tick is a regression, even if it looks right on screen.
- **Snapshots are re-read one by one before being accepted.** Never `INSTA_UPDATE=always` over the whole suite: a snapshot accepted unread freezes whatever bug is in it. `cargo-insta` is not installed; accept one at a time with `INSTA_UPDATE=always cargo test --lib <exact test path>` after reading its diff.
- **One commit for the whole milestone.** Do not commit per task — each task ends at its checks. The orchestrator squashes the branch into a single commit, per the repo's `J<N>: …` convention.
- Semantic palette field names do not change; only their values. Every existing `palette().x` call site keeps compiling.

## Review Focus

Input classes the spec implies but no task's own tests exercise. Each line's test is added to the task that owns the code.

1. **A terminal narrower than the indentation.** The reply sits at column 2 and the question at column 8; at 20 columns — or 8 — the remaining text width reaches zero and `wrap_spans` is called with it. It must not panic, divide by zero, or loop. (Task 3)
2. **An out-of-context message that is the user's.** `dimmed()` repaints every span `dim`, and the user's messages are now `dim` by default — so the two states look identical. The separator becomes the only distinction and must still be drawn. (Task 3)
3. **A long unbroken token in an indented message.** A pasted URL of 200 characters in a question rendered at column 8 must wrap inside the pane, not push the line past the right edge. (Task 3)
4. **A phase arriving after `Done`, or out of order.** The `request_id` guard catches a stale *request*, not a stale *phase*: a `Retrieving` arriving after the first token must not resurrect a waiting line on a reply that is already streaming or finished. (Task 4)
5. **A terminal that cannot draw braille.** `⠻` becomes a replacement box in fonts without the Braille Patterns block. The waiting line must still say what is happening without it — the glyph carries no information the text does not. (Task 4)

---

### Task 1: Both palettes pinned and reduced

**Files:**
- Modify: `src/theme.rs:51-80` (the two palette constants), tests appended to its existing `mod tests`

**Interfaces:**
- Consumes: nothing.
- Produces: `theme::DARK` and `theme::LIGHT` with unchanged field names and new values. Every `crate::theme::palette().<field>` call site in the crate keeps compiling; Tasks 3, 4 and 5 read these colours through `palette()` as they do today.

- [ ] **Step 1: Write the failing test**

Append inside the existing `#[cfg(test)] mod tests` in `src/theme.rs`:

```rust
    /// A basic ANSI colour is not a colour: its tone comes from the terminal's own palette,
    /// so the same build looks different under Gruvbox, Dracula or Solarized. Both palettes
    /// must pin indexed tones. This test is the guard against the regression.
    #[test]
    fn every_colour_is_pinned_not_inherited() {
        for (name, palette) in [("DARK", DARK), ("LIGHT", LIGHT)] {
            for (field, colour) in [
                ("dim", palette.dim),
                ("accent", palette.accent),
                ("warn", palette.warn),
                ("error", palette.error),
                ("ok", palette.ok),
                ("info", palette.info),
                ("assistant", palette.assistant),
                ("selection_bg", palette.selection_bg),
                ("bar_bg", palette.bar_bg),
                ("badge_fg", palette.badge_fg),
                ("badge_bg", palette.badge_bg),
            ] {
                assert!(
                    matches!(colour, Color::Indexed(_)),
                    "{name}.{field} is {colour:?}, not an indexed tone"
                );
            }
        }
    }

    /// Monochrome plus one accent: `accent`, `info` and `assistant` are the same family, so
    /// the interface does not read as three competing hues.
    #[test]
    fn the_accent_family_is_one_hue() {
        assert_eq!(DARK.accent, DARK.assistant);
        assert_eq!(LIGHT.accent, LIGHT.assistant);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib theme`
Expected: FAIL — `every_colour_is_pinned_not_inherited` reports `DARK.dim is DarkGray, not an indexed tone`.

- [ ] **Step 3: Replace both palette constants**

Replace `src/theme.rs:51-80` (from `/// For dark backgrounds` through the end of `LIGHT`) with:

```rust
/// For dark backgrounds (the default).
///
/// Every tone is indexed on purpose. The eight basic ANSI colours are not colours: their
/// actual tone is whatever the user's terminal palette says, so `Color::Cyan` is a
/// different blue under every popular theme. Monochrome plus one accent: body text uses the
/// terminal's default foreground and never appears here.
pub const DARK: Palette = Palette {
    dim: Color::Indexed(245),
    accent: Color::Indexed(110),
    warn: Color::Indexed(179),
    error: Color::Indexed(167),
    ok: Color::Indexed(108),
    info: Color::Indexed(110),
    assistant: Color::Indexed(110),
    selection_bg: Color::Indexed(238),
    bar_bg: Color::Indexed(234),
    badge_fg: Color::Indexed(234),
    badge_bg: Color::Indexed(245),
    code_theme: "base16-ocean.dark",
};

/// For light backgrounds: the same reduced vocabulary in tones that stay readable on white.
pub const LIGHT: Palette = Palette {
    dim: Color::Indexed(245),
    accent: Color::Indexed(25),
    warn: Color::Indexed(130),
    error: Color::Indexed(160),
    ok: Color::Indexed(28),
    info: Color::Indexed(25),
    assistant: Color::Indexed(25),
    selection_bg: Color::Indexed(254),
    bar_bg: Color::Indexed(254),
    badge_fg: Color::Indexed(255),
    badge_bg: Color::Indexed(245),
    code_theme: "InspiredGitHub",
};
```

Note `badge_fg` changes from `Color::Black` / `Color::White` to indexed equivalents, and
`info` collapses onto the accent in both palettes — three hues become one.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib theme`
Expected: PASS — all tests in the module, including the pre-existing
`auto_follows_the_terminal_background`.

- [ ] **Step 5: Check the whole crate still builds and nothing else broke**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test --lib theme`
Expected: no warnings; theme tests pass.

The full suite is **not** green at this point and is not expected to be: snapshots capture
characters, not colours, so they are unaffected by this task — but run
`cargo test` anyway and record the result. If a snapshot fails here, something other than
colour changed and it must be investigated before moving on.

---

### Task 2: `Phase`, `LlmEvent::Phase`, and its emission

**Files:**
- Modify: `src/llm/mod.rs:269-298` (the `LlmEvent` enum), plus a new `Phase` enum above it
- Modify: `src/llm/stream_task.rs:125-160` (around the awaits already there), and the tool-call path near `stream_task.rs:243`
- Test: appended to the existing `#[cfg(test)] mod tests` in `src/llm/stream_task.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```rust
  pub enum Phase {
      Connecting,
      Retrieving { collections: usize },
      Waiting { model: String },
      RunningTool { name: String },
  }
  pub enum LlmEvent { /* … existing variants … */ Phase(Phase) }
  ```
  Task 4 matches on `LlmEvent::Phase(phase)` in `App::on_llm_event` and renders each variant.
  The guarantee that a *refused* tool emits no `RunningTool` is asserted in Task 4, from
  `tests/app_flow.rs`, where a refusal path already exists — there is no helper in this
  module for driving a refusal.

- [ ] **Step 1: Write the failing test**

Append inside `#[cfg(test)] mod tests` in `src/llm/stream_task.rs`:

```rust
    /// The screen is still until the first token, so the task has to say which wait the
    /// user is in: connecting, retrieving, or waiting on the model.
    #[tokio::test]
    async fn the_phases_are_announced_in_order() {
        let llm = Arc::new(MockLlmClient::new(vec!["bonjour"]));
        let events = run_to_end(backends(llm)).await;
        let phases: Vec<Phase> = events
            .into_iter()
            .filter_map(|e| match e {
                LlmEvent::Phase(phase) => Some(phase),
                _ => None,
            })
            .collect();
        assert_eq!(phases.len(), 3, "{phases:?}");
        assert_eq!(phases[0], Phase::Connecting);
        // `NoContext` and a job with no `rag_collection`: zero, so the line will read
        // "recherche dans les documents…".
        assert_eq!(phases[1], Phase::Retrieving { collections: 0 });
        // The model is read from the job rather than hard-coded, so `job()` stays free to
        // change its fixture.
        assert_eq!(
            phases[2],
            Phase::Waiting {
                model: job().model.clone()
            }
        );
    }
```

`run_to_end(backends)`, `backends(llm)` and `job()` are the three helpers this module's
tests already use (`src/llm/stream_task.rs:291-311`); `MockLlmClient::new(vec![…])` is the
mock the rest of the crate uses. Do not add a fourth helper.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib llm::stream_task`
Expected: FAIL to compile — `cannot find type Phase in this scope` and
`no variant named Phase found for enum LlmEvent`.

- [ ] **Step 3: Add `Phase` and the enum variant**

In `src/llm/mod.rs`, immediately above `pub enum LlmEvent` (line 269):

```rust
/// Which wait the user is currently in, while no token has arrived yet.
///
/// The reply is a sequence of waits — connecting, retrieving documents, waiting on the
/// model, running a tool — and until the first token none of them is visible on screen.
/// `stream_task` reports each one around the `await` it already performs, so neither the
/// `LlmClient` nor the `ContextProvider` trait has to change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The job has started; nothing has been asked of anyone yet.
    Connecting,
    /// Retrieval is running over `collections` collections (0 when none is selected).
    Retrieving { collections: usize },
    /// The request is out and the model has not answered yet.
    Waiting { model: String },
    /// A tool the user approved is executing.
    RunningTool { name: String },
}
```

Then add the variant to `LlmEvent`, after `Token`:

```rust
    /// Which wait is in progress, before the first token (see [`Phase`]).
    Phase(Phase),
```

- [ ] **Step 4: Emit the three pre-token phases**

In `src/llm/stream_task.rs`, inside `generate`, replace the opening of the `JobKind::Reply`
arm (currently `let query = ContextQuery { … };` through the `provide(query).await` match)
with:

```rust
        JobKind::Reply => {
            send(LlmEvent::Phase(Phase::Connecting));
            let query = ContextQuery {
                collection: job.rag_collection.as_deref(),
                history: &job.history,
            };
            // `/rag a,b` stores the selection comma-joined; the count is what the line says.
            let collections = job
                .rag_collection
                .as_deref()
                .map_or(0, |names| names.split(',').filter(|n| !n.is_empty()).count());
            send(LlmEvent::Phase(Phase::Retrieving { collections }));
            let context = match backends.context.provide(query).await {
                Ok(context) => context,
                Err(error) => return send(LlmEvent::Error(error.to_string())),
            };
            if !context.is_empty() {
                send(LlmEvent::Retrieved {
                    first_number: prompt::first_context_number(&job.history),
                    chunks: context.chunks.clone(),
                });
            }
            prompt::build_messages(&job.system_prompt, &context, &job.history)
        }
```

Then, immediately after the `let Some(llm) = backends.clients.get(&job.provider) else { … };`
block that follows, add:

```rust
    send(LlmEvent::Phase(Phase::Waiting {
        model: job.model.clone(),
    }));
```

Add `Phase` to the `use super::{…}` list at the top of the file alongside `LlmEvent`.

- [ ] **Step 5: Emit `RunningTool` where a tool actually starts**

At `src/llm/stream_task.rs:243`, where the approved call is dispatched to `mcp.call` /
`tools::run`, emit the phase immediately before the await:

```rust
            send(LlmEvent::Phase(Phase::RunningTool {
                name: call.name.clone(),
            }));
```

Place it after the user's decision has been read and before the call is awaited — the point
is to distinguish *waiting for the user* from *executing*. Read the surrounding ten lines
first: if the decision and the dispatch are in separate branches, the emission belongs in the
branch that runs the tool, not the one that refuses it. A refused call must emit no
`RunningTool`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test --lib llm::stream_task`
Expected: PASS, including the pre-existing `streams_tokens_then_done`,
`mid_stream_failure_keeps_previous_tokens` and
`cancellation_stops_the_task_and_drops_the_stream`.

- [ ] **Step 7: Run the checks**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`
Expected: all three clean. `App` ignores the new event for now — `on_llm_event` matches on
the variants it knows and its `_ =>` arm, if any, swallows `Phase`. If the match is
exhaustive and the crate does not compile, add `LlmEvent::Phase(_) => Vec::new(),` to
`App::on_llm_event` as a placeholder; Task 4 replaces it with the real handling.

---

### Task 3: Positional layout and the markdown vocabulary

**Files:**
- Modify: `src/transcript.rs` — `message_lines_marked` (211-262), `header` (283-296, deleted), `plain` (386-391), `tool_card` (299-318), `attachment_card` (323-347)
- Modify: `src/markdown/render.rs` — `BULLETS` (14), `heading_style` (24-33), `TagEnd::Heading` (279-283), `render_code` (466-499)
- Test: appended to the existing `mod tests` in both files

**Interfaces:**
- Consumes: `theme::DARK` / `theme::LIGHT` from Task 1, through `crate::theme::palette()`.
- Produces:
  ```rust
  // src/transcript.rs
  pub fn message_lines(message: &Message, width: usize) -> Vec<Line<'static>>;
  pub fn message_lines_marked(
      message: &Message, width: usize, mark: Option<(usize, usize)>,
  ) -> Vec<Line<'static>>;
  /// Left margin of each role, in columns.
  fn indent(role: Role) -> usize;
  ```
  Both public signatures are unchanged. Task 4 calls `message_lines_marked` with a fourth
  argument added in that task; Task 5 renders the result.

- [ ] **Step 1: Write the failing layout tests**

Append inside `#[cfg(test)] mod tests` in `src/transcript.rs`:

`Message` is a plain struct with seven public fields and no constructor
(`src/state/conversation.rs:85-96`), so build it literally:

```rust
    fn message(id: u64, role: Role, content: &str) -> Message {
        Message {
            id: MessageId(id),
            role,
            content: content.to_owned(),
            status: MessageStatus::Complete,
            source: None,
            citations: Vec::new(),
            image: None,
        }
    }

    fn user(text: &str) -> Message {
        message(1, Role::User, text)
    }

    fn assistant(text: &str) -> Message {
        message(2, Role::Assistant, text)
    }

    /// Role is read from position: the question is pushed right and dimmed, the reply sits
    /// at the left margin in the terminal's own foreground. No label on either.
    #[test]
    fn the_role_is_read_from_the_indentation() {
        let question = message_lines(&user("Comment trier un Vec ?"), 60);
        let text: Vec<String> = question.iter().map(ToString::to_string).collect();
        assert!(
            text.iter().all(|l| !l.contains("Vous")),
            "the role label is gone: {text:?}"
        );
        assert!(
            text[0].starts_with("        Comment"),
            "the question sits at column 8: {:?}",
            text[0]
        );

        let reply = message_lines(&assistant("Utilise sort_unstable."), 60);
        let text: Vec<String> = reply.iter().map(ToString::to_string).collect();
        assert!(
            text.iter().all(|l| !l.contains("Assistant")),
            "the role label is gone: {text:?}"
        );
        assert!(
            text[0].starts_with("  Utilise"),
            "the reply sits at column 2: {:?}",
            text[0]
        );
    }

    /// Four roles are too rare to be expressed by position and keep a dim label.
    #[test]
    fn the_rare_roles_keep_a_label() {
        for (role, expected) in [
            (Role::System, "système"),
            (Role::Summary, "résumé de la conversation"),
        ] {
            let message = message(3, role, "texte");
            let text = message_lines(&message, 60)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains(expected), "{role:?}: {text}");
        }
    }

    /// The version marker lost its home when the header went away; it lands on its own dim
    /// line under the body, with the model that wrote that version.
    #[test]
    fn the_version_marker_sits_under_the_body() {
        let mut message = assistant("Première réponse.");
        message.source = Some("llama3.2".to_owned());
        let lines = message_lines_marked(&message, 60, Some((2, 3)));
        let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
        let marker = text
            .iter()
            .position(|l| l.contains("‹ 2/3 ›"))
            .expect("the marker is shown");
        let body = text
            .iter()
            .position(|l| l.contains("Première"))
            .expect("the body is shown");
        assert!(marker > body, "the marker comes after the body: {text:?}");
        assert!(text[marker].contains("llama3.2"), "{:?}", text[marker]);
    }

    /// Two blank lines between turns, one inside a turn — the rhythm that replaces the
    /// frames. A question carries the extra one, so the rule holds wherever a turn starts.
    #[test]
    fn a_turn_is_separated_by_two_blank_lines_and_its_halves_by_one() {
        let blanks = |lines: &[Line<'static>]| -> (usize, usize) {
            let leading = lines.iter().take_while(|l| l.spans.is_empty()).count();
            let trailing = lines.iter().rev().take_while(|l| l.spans.is_empty()).count();
            (leading, trailing)
        };
        assert_eq!(blanks(&message_lines(&user("Question ?"), 60)), (1, 1));
        assert_eq!(blanks(&message_lines(&assistant("Réponse."), 60)), (0, 1));
    }

    /// Without versions there is no metadata at all: nothing by default.
    #[test]
    fn without_versions_there_is_no_metadata_line() {
        let mut message = assistant("Réponse.");
        message.source = Some("llama3.2".to_owned());
        let text = message_lines_marked(&message, 60, None)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("llama3.2"), "{text}");
        assert!(!text.contains('‹'), "{text}");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib transcript`
Expected: FAIL — `the_role_is_read_from_the_indentation` reports that line 0 is
`▌ Vous`, not `        Comment`.

- [ ] **Step 3: Add the indentation helpers**

In `src/transcript.rs`, replace `fn header(role: Role) -> Line<'static>` (283-296) with:

```rust
/// Left margin of each role, in columns.
///
/// This is the layout: role is read from position, not from a label. The reply sits at the
/// left margin, the question is pushed right and dimmed, and the rare roles — which cannot
/// be told apart by position — share the reply's margin and carry a label instead.
fn indent(role: Role) -> usize {
    match role {
        Role::Assistant | Role::Summary | Role::System | Role::Attachment | Role::Tool => 2,
        Role::User => 8,
    }
}

/// The dim label of a role too rare to be read from its position, or `None` when position
/// says it already.
fn label(role: Role) -> Option<&'static str> {
    match role {
        Role::User | Role::Assistant => None,
        Role::System => Some("système"),
        Role::Summary => Some("résumé de la conversation"),
        Role::Attachment => Some("fichier joint"),
        Role::Tool => Some("outil"),
    }
}

/// Shifts every line right by `columns`, leaving blank lines blank so no trailing spaces
/// land in the snapshots.
fn shift(lines: Vec<Line<'static>>, columns: usize) -> Vec<Line<'static>> {
    let pad = " ".repeat(columns);
    lines
        .into_iter()
        .map(|line| {
            if line.spans.is_empty() {
                return line;
            }
            let style = line.style;
            let mut spans = vec![Span::raw(pad.clone())];
            spans.extend(line.spans);
            Line::from(spans).style(style)
        })
        .collect()
}
```

- [ ] **Step 4: Rewrite `message_lines_marked`**

Replace the body of `message_lines_marked` (211-262) with:

```rust
pub fn message_lines_marked(
    message: &Message,
    width: usize,
    mark: Option<(usize, usize)>,
) -> Vec<Line<'static>> {
    let palette = crate::theme::palette();
    let dim = Style::default().fg(palette.dim);
    let columns = indent(message.role);
    // The body is wrapped to what is left once the margin is taken, then shifted into it.
    let inner = width.saturating_sub(columns).max(1);

    let mut lines = Vec::new();
    if let Some(label) = label(message.role) {
        lines.push(Line::styled(label.to_owned(), dim));
    }
    let mut body = match message.role {
        Role::Assistant | Role::Summary => markdown::render(&message.content, inner),
        Role::User => plain(&message.content, inner, dim),
        Role::System => plain(&message.content, inner, dim),
        Role::Attachment => attachment_card(message, inner),
        Role::Tool => tool_card(message, inner),
    };

    match &message.status {
        MessageStatus::Complete => {}
        MessageStatus::Streaming => match body.last_mut() {
            Some(last) if last.width() < inner => last.push_span(Span::styled("▍", dim)),
            _ => body.push(Line::styled("▍", dim)),
        },
        MessageStatus::Cancelled => body.push(Line::styled("[interrompu]", dim.italic())),
        MessageStatus::Failed(error) => {
            let red = Style::default().fg(palette.error);
            let spans = [Span::styled(format!("✖ {error}"), red)];
            body.extend(wrap_spans(&spans, inner).into_iter().map(Line::from));
        }
    }
    lines.extend(body);
    if !message.citations.is_empty() {
        lines.extend(citation_lines(message, inner));
    }
    // Metadata only on demand: the model is named only when there is another version to
    // compare it against. This is where the deleted header's marker went.
    if let Some((shown, total)) = mark {
        let mut spans = Vec::new();
        if message.role == Role::Assistant
            && let Some(model) = &message.source
        {
            spans.push(Span::styled(format!("{model} · "), dim));
        }
        spans.push(Span::styled(
            format!("‹ {shown}/{total} ›  Alt+← Alt+→"),
            dim,
        ));
        lines.push(Line::from(spans));
    }
    let mut lines = shift(lines, columns);
    // The rhythm replaces the frames: one blank line after every message, and one more
    // before a question, so a turn is separated from the next by two and its own halves by
    // one. The conversation therefore opens on a blank line, which is wanted.
    if message.role == Role::User {
        lines.insert(0, Line::default());
    }
    lines.push(Line::default());
    lines
}
```

- [ ] **Step 5: Make `plain` carry a style**

Replace `fn plain` (386-391) with:

```rust
/// User text is shown as typed: line breaks kept, words wrapped, in `style`.
fn plain(text: &str, width: usize, style: Style) -> Vec<Line<'static>> {
    text.lines()
        .flat_map(|line| wrap_spans(&[Span::styled(line.to_owned(), style)], width))
        .map(Line::from)
        .collect()
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib transcript`
Expected: PASS for the four new tests. Pre-existing tests in the module that assert on
`▌ Vous` or `▌ Assistant` will fail — those assertions are now wrong, not the code. Update
each one to the new layout, reading it first to be sure it was testing the label and not
something else that happens to mention it.

- [ ] **Step 7: Write the failing Review Focus tests**

```rust
    /// Review Focus 1: a pane narrower than the indentation leaves no room for text. It
    /// must not panic, and it must still produce lines.
    #[test]
    fn a_pane_narrower_than_the_indentation_does_not_panic() {
        for width in [0, 1, 2, 8, 9, 20] {
            let lines = message_lines(&user("Comment trier un Vec ?"), width);
            assert!(!lines.is_empty(), "width {width} produced nothing");
            let lines = message_lines(&assistant("# Titre\n\nTexte."), width);
            assert!(!lines.is_empty(), "width {width} produced nothing");
        }
    }

    /// Review Focus 3: a pasted URL has no break point. Wrapped at an indentation, it must
    /// stay inside the pane.
    #[test]
    fn a_long_unbroken_token_stays_inside_the_pane() {
        let url = "https://example.com/".to_owned() + &"a".repeat(200);
        let width = 40;
        for line in message_lines(&user(&url), width) {
            assert!(
                line.width() <= width,
                "line of {} columns in a pane of {width}: {line:?}",
                line.width()
            );
        }
    }
```

- [ ] **Step 8: Run them and fix what they catch**

Run: `cargo test --lib transcript`
Expected: both PASS if `saturating_sub(columns).max(1)` and `wrap_spans` already behave.
If `a_long_unbroken_token_stays_inside_the_pane` fails, the bug is real and is in the
interaction between `shift` and the wrap width — the body was wrapped to `inner` but a span
longer than `inner` was not broken. Fix it in `wrap_spans`' caller by wrapping to `inner`
and never to `width`; do not widen the assertion.

- [ ] **Step 9: Write the failing out-of-context test**

Review Focus 2. This one belongs with `refresh`, which is what applies `dimmed`:

```rust
    /// Review Focus 2: the user's messages are dim now, and `dimmed()` repaints every span
    /// dim — so an out-of-context question looks exactly like an in-context one. The
    /// separator is the only remaining distinction and must be drawn.
    #[test]
    fn the_out_of_context_separator_is_the_only_distinction_left() {
        let mut transcript = Transcript::default();
        let messages = vec![user("ancienne question"), assistant("ancienne réponse")];
        transcript.refresh(&messages, 60, 3);
        let text = transcript.line_texts().join("\n");
        assert!(
            text.contains("ne sont plus envoyés au modèle"),
            "the separator must be drawn: {text}"
        );
    }
```

- [ ] **Step 10: Run it**

Run: `cargo test --lib transcript::tests::the_out_of_context_separator`
Expected: PASS — `boundary()` is untouched by this task. If it fails, `refresh`'s
`cleared_all` condition does not hold for this input; read `refresh` (118-154) and build the
message ids so that all of them fall before `context_start`, rather than changing `refresh`.

- [ ] **Step 11: Write the failing markdown tests**

Append inside `#[cfg(test)] mod tests` in `src/markdown/render.rs`:

```rust
    /// A heading is weight plus a rule the width of its own text — not a colour.
    #[test]
    fn a_heading_is_followed_by_a_rule_its_own_width() {
        let lines = render("## Tri en Rust\n\ntexte", 60);
        let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
        let title = text
            .iter()
            .position(|l| l.contains("Tri en Rust"))
            .expect("the heading is rendered");
        let rule = text[title + 1].trim_end();
        assert_eq!(rule, "─".repeat("Tri en Rust".chars().count()), "{text:?}");
    }

    /// The code block loses its per-line gutter and keeps its label line, because
    /// `/copy code N` is unusable when the numbers are invisible.
    #[test]
    fn a_code_block_is_indented_without_a_gutter() {
        let lines = render("```rust\nlet v = 1;\n```\n", 60);
        let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
        assert!(
            text.iter().all(|l| !l.contains('▎')),
            "the gutter is gone: {text:?}"
        );
        let code = text
            .iter()
            .find(|l| l.contains("let v = 1;"))
            .expect("the code is rendered");
        assert!(code.starts_with("    "), "code is indented: {code:?}");
    }

    /// With several blocks the number is shown at the right of the label line; with one it
    /// is not shown at all.
    #[test]
    fn the_block_number_appears_only_when_there_are_several() {
        let one = render("```rust\na\n```\n", 60)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!one.contains('['), "{one}");

        let two = render("```rust\na\n```\n\n```sh\nb\n```\n", 60)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(two.contains("[1]"), "{two}");
        assert!(two.contains("[2]"), "{two}");
    }

    /// Lists use `·`, the calmest marker available.
    #[test]
    fn a_list_uses_a_middle_dot() {
        let text = render("- plus rapide\n- pas d'allocation\n", 60)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("· plus rapide"), "{text}");
        assert!(!text.contains('•'), "{text}");
    }
```

- [ ] **Step 12: Run them to verify they fail**

Run: `cargo test --lib markdown::render`
Expected: FAIL — `a_list_uses_a_middle_dot` reports `• plus rapide`,
`a_code_block_is_indented_without_a_gutter` finds `▎`.

- [ ] **Step 13: Change the markdown vocabulary**

In `src/markdown/render.rs`:

Line 14 — the top-level marker becomes a middle dot, nesting stays distinguishable:

```rust
const BULLETS: [&str; 3] = ["· ", "◦ ", "▪ "];
```

Lines 24-33 — a heading is weight and a rule, not a hue:

```rust
fn heading_style(level: HeadingLevel) -> Style {
    let bold = Style::default().add_modifier(Modifier::BOLD);
    match level {
        // The rule under H1 and H2 carries the hierarchy; the text itself stays
        // monochrome so a reply does not read as three competing colours.
        HeadingLevel::H1 | HeadingLevel::H2 => bold,
        _ => bold,
    }
}
```

Keep the function rather than inlining it: `render.rs` calls it from `Tag::Heading` and the
level is still the decision point for the rule below.

Lines 279-283 — emit the rule. Replace the `TagEnd::Heading(_)` arm with:

```rust
            TagEnd::Heading(level) => {
                // The width of the heading's own text, before `flush_inline` clears it.
                let columns: usize = self
                    .inline
                    .iter()
                    .map(|span| display_width(&span.content))
                    .sum();
                self.flush_inline();
                self.pop_style();
                if matches!(level, HeadingLevel::H1 | HeadingLevel::H2) && columns > 0 {
                    let available = self.width.saturating_sub(self.prefix_width()).max(1);
                    self.emit(vec![Span::styled(
                        "─".repeat(columns.min(available)),
                        dim(),
                    )]);
                }
                self.end_block();
            }
```

`self.inline` is the `Vec<Span<'static>>` buffer that `flush_inline` drains with
`std::mem::take` (`src/markdown/render.rs:115` and `:403-412`), so the width has to be
measured **before** calling it — after the take it is empty and the rule would be zero wide.

- [ ] **Step 14: Replace the code block's gutter with an indent**

In `render_code` (466-499), replace the constant, the closure and the two `gutter()` uses:

```rust
        // The block is set in from the text, with no per-line marker: the indent and the
        // syntax colours say it is code. The label line above carries the language and,
        // when the reply holds several blocks, the number `/copy code N` needs.
        const INDENT: &str = "    ";
        let indent = || Span::raw(INDENT);
        let available = self
            .width
            .saturating_sub(self.prefix_width() + display_width(INDENT))
            .max(1);
        let lang = code.lang.trim();
        self.code_count += 1;
        let number = self.number_code.then(|| format!("[{}]", self.code_count));
        if lang.is_empty() && number.is_none() {
            // Nothing to label: no line.
        } else {
            let label_width = self.width.saturating_sub(self.prefix_width() + 2).max(1);
            let left = if lang.is_empty() { "" } else { lang };
            let right = number.unwrap_or_default();
            let gap = label_width
                .saturating_sub(display_width(left) + display_width(&right))
                .max(1);
            self.emit(vec![
                Span::raw("  "),
                Span::styled(left.to_owned(), dim()),
                Span::raw(" ".repeat(gap)),
                Span::styled(right, dim()),
            ]);
        }
        let highlighted = if self.code_closed {
            highlight_cached(&code.text, lang)
        } else {
            highlight(&code.text, lang)
        };
        for source_line in highlighted {
            for wrapped in wrap_plain(&source_line, available) {
                let mut spans = vec![indent()];
                spans.extend(wrapped);
                self.emit(spans);
            }
        }
```

- [ ] **Step 15: Run the markdown tests to verify they pass**

Run: `cargo test --lib markdown`
Expected: PASS for the four new tests. The pre-existing
`code_block_is_highlighted_with_gutter` will fail — it asserts the behaviour this task
removes. Rename it `code_block_is_highlighted_and_indented` and change its assertion from
`▎` to the four-space indent. Read `unterminated_code_block_while_streaming` and confirm it
still passes untouched; if it asserted on the gutter too, update it the same way.

- [ ] **Step 16: Run this task's checks**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test --lib transcript --lib markdown --lib theme`
Expected: no warnings, those modules green.

Then run `cargo test` and record the output. **Snapshots will fail — around twenty of them —
and that is expected at this point.** Do not accept any of them here: Task 4 changes the
rendering again, and Task 5 is where they are read and accepted once. List the failing
snapshot names in your report so Task 5 knows the scope.

---

### Task 4: `Waiting`, the tick counter, and the tool card's two states

**Files:**
- Modify: `src/transcript.rs` — the `Transcript` struct (28-42), a new `Waiting` type, `set_waiting`, `refresh` (78-154), `tool_card` (299-318)
- Modify: `src/app.rs` — the `App` struct, `Action::Tick` (1245), `on_llm_event` (2990+), `start_job_with` (2008), `refresh_view` (745)
- Modify: `src/ui/status_bar.rs:20-28` (the spinner)
- Test: appended to `mod tests` in `src/transcript.rs`, `src/app.rs`, and `tests/app_flow.rs`

**Interfaces:**
- Consumes: `crate::llm::Phase` and `LlmEvent::Phase` from Task 2; `indent`, `shift` and `message_lines_marked` from Task 3.
- **This task changes two signatures Task 3 wrote**, which is expected and not a conflict:
  `message_lines_marked` gains a fourth parameter and `tool_card` a third. Two of Task 3's
  own tests call `message_lines_marked` with three arguments —
  `the_version_marker_sits_under_the_body` and `without_versions_there_is_no_metadata_line` —
  and must be updated to pass `None` as the fourth. `message_lines` keeps its three-argument
  shape as a wrapper, so every other Task 3 test compiles untouched.
- Produces:
  ```rust
  // src/transcript.rs
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct Waiting { pub phase: Phase, pub frame: usize, pub elapsed_s: Option<u32> }
  impl Transcript { pub fn set_waiting(&mut self, waiting: Option<Waiting>); }
  pub const SPINNER: [&str; 10];
  // src/app.rs
  pub wait: Option<(crate::llm::Phase, u32)>,  // field, like `pulling`
  impl App { pub fn waiting(&self) -> Option<Waiting>; }
  ```
  Task 5 renders nothing new from these — the waiting line is produced inside the
  transcript — but its snapshots exercise them.

- [ ] **Step 1: Write the failing transcript tests**

Append inside `#[cfg(test)] mod tests` in `src/transcript.rs`:

```rust
    use crate::llm::Phase;

    fn streaming(text: &str) -> Message {
        let mut message = Message::new(MessageId(2), Role::Assistant, text.to_owned());
        message.status = MessageStatus::Streaming;
        message
    }

    /// The waiting line stands where the reply will appear, so the text replaces it in
    /// place when the first token lands.
    #[test]
    fn the_waiting_line_stands_where_the_reply_will_appear() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Waiting {
                model: "llama3.2".to_owned(),
            },
            frame: 0,
            elapsed_s: Some(3),
        }));
        transcript.refresh(&[user("Raconte"), streaming("")], 60, 0);
        let text = transcript.line_texts().join("\n");

        assert!(text.contains("llama3.2 réfléchit…"), "{text}");
        assert!(text.contains("3 s"), "{text}");
        let line = transcript
            .line_texts()
            .into_iter()
            .find(|l| l.contains("réfléchit"))
            .expect("the line is there");
        assert!(line.starts_with("  "), "at column 2: {line:?}");
    }

    /// Once a token has arrived the message is no longer empty, and the waiting line has no
    /// business being there even if the phase was not cleared yet.
    #[test]
    fn a_message_with_content_shows_no_waiting_line() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Waiting {
                model: "llama3.2".to_owned(),
            },
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[user("Raconte"), streaming("Il était")], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(!text.contains("réfléchit"), "{text}");
        assert!(text.contains("Il était"), "{text}");
    }

    /// Under a second, no number: a fast reply must not flash one.
    #[test]
    fn under_a_second_no_duration_is_shown() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::Connecting,
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[user("Salut"), streaming("")], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(text.contains("connexion…"), "{text}");
        assert!(!text.contains(" s"), "{text}");
    }

    /// One collection reads differently from several, and zero reads as the documents.
    #[test]
    fn the_retrieval_line_counts_its_collections() {
        for (collections, expected) in [
            (0, "recherche dans les documents…"),
            (1, "recherche dans 1 collection…"),
            (2, "recherche dans 2 collections…"),
        ] {
            let mut transcript = Transcript::default();
            transcript.set_waiting(Some(Waiting {
                phase: Phase::Retrieving { collections },
                frame: 0,
                elapsed_s: None,
            }));
            transcript.refresh(&[user("q"), streaming("")], 60, 0);
            let text = transcript.line_texts().join("\n");
            assert!(text.contains(expected), "{collections}: {text}");
        }
    }

    /// The redraw gate is `revision`, so a frame change must move it — and an unchanged
    /// value must not, or the 30 fps markdown cap is gone.
    #[test]
    fn only_a_real_change_bumps_the_revision() {
        let mut transcript = Transcript::default();
        let messages = [user("q"), streaming("")];
        let waiting = |frame| {
            Some(Waiting {
                phase: Phase::Connecting,
                frame,
                elapsed_s: None,
            })
        };

        transcript.set_waiting(waiting(0));
        transcript.refresh(&messages, 60, 0);
        let first = transcript.revision();

        transcript.set_waiting(waiting(0));
        transcript.refresh(&messages, 60, 0);
        assert_eq!(transcript.revision(), first, "same frame, no redraw");

        transcript.set_waiting(waiting(1));
        transcript.refresh(&messages, 60, 0);
        assert!(transcript.revision() > first, "new frame, redraw");
    }

    /// Review Focus 5: the braille glyph carries no information the text does not. A
    /// terminal that cannot draw it must still show a readable line.
    #[test]
    fn the_waiting_line_reads_without_its_glyph() {
        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: Phase::RunningTool {
                name: "read_file".to_owned(),
            },
            frame: 4,
            elapsed_s: Some(2),
        }));
        transcript.refresh(&[user("q"), streaming("")], 60, 0);
        let line = transcript
            .line_texts()
            .into_iter()
            .find(|l| l.contains("read_file"))
            .expect("the line is there");
        let without_glyph: String = line.chars().filter(|c| !SPINNER.contains(&&*c.to_string())).collect();
        assert!(
            without_glyph.contains("exécution de read_file"),
            "{without_glyph:?}"
        );
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --lib transcript`
Expected: FAIL to compile — `cannot find type Waiting in this scope`,
`no method named set_waiting`.

- [ ] **Step 3: Add `Waiting`, `SPINNER` and `set_waiting`**

In `src/transcript.rs`, after the `Entry` struct:

```rust
/// Spinner frames. Braille is one cell wide in every font that has the block, and the text
/// beside it carries the meaning for the fonts that do not.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// What the application is doing while no token has arrived yet.
///
/// `frame` and `elapsed_s` are already reduced from `App`'s tick counter, which is
/// deliberately not stored here: the comparison in [`Transcript::set_waiting`] must see a
/// change about ten times a second, not thirty, or the markdown cap goes with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waiting {
    pub phase: crate::llm::Phase,
    pub frame: usize,
    /// Whole seconds waited; `None` under one second, so a fast reply shows no number.
    pub elapsed_s: Option<u32>,
}

impl Waiting {
    /// The line as the user reads it, without the margin.
    fn line(&self) -> Line<'static> {
        use crate::llm::Phase;
        let dim = Style::default().fg(crate::theme::palette().dim);
        let what = match &self.phase {
            Phase::Connecting => "connexion…".to_owned(),
            Phase::Retrieving { collections: 0 } => "recherche dans les documents…".to_owned(),
            Phase::Retrieving { collections: 1 } => "recherche dans 1 collection…".to_owned(),
            Phase::Retrieving { collections } => {
                format!("recherche dans {collections} collections…")
            }
            Phase::Waiting { model } => format!("{model} réfléchit…"),
            Phase::RunningTool { name } => format!("exécution de {name}…"),
        };
        let glyph = SPINNER[self.frame % SPINNER.len()];
        let mut spans = vec![Span::styled(format!("{glyph} {what}"), dim)];
        if let Some(seconds) = self.elapsed_s {
            spans.push(Span::styled(format!("  {seconds} s"), dim));
        }
        Line::from(spans)
    }
}
```

Add `waiting: Option<Waiting>` to the `Transcript` struct with a doc comment, and the setter
beside `set_marks`:

```rust
    /// Sets the waiting line. The streaming message is re-rendered when it changes, which
    /// is what makes the spinner turn: `refresh` bumps `revision`, and `runtime.rs` redraws
    /// on a tick only when `revision` moved.
    pub fn set_waiting(&mut self, waiting: Option<Waiting>) {
        if waiting == self.waiting {
            return;
        }
        self.waiting = waiting;
        // Only the message being streamed shows it.
        if let Some(entry) = self.entries.last() {
            let id = entry.id;
            self.invalidate(id);
        }
    }
```

- [ ] **Step 4: Render the waiting line in `refresh`**

`refresh` already renders each dirty entry through `message_lines_marked`. The waiting line
belongs to the streaming message whose content is empty, so give
`message_lines_marked` a fourth parameter and pass it from `refresh`:

```rust
pub fn message_lines_marked(
    message: &Message,
    width: usize,
    mark: Option<(usize, usize)>,
    waiting: Option<&Waiting>,
) -> Vec<Line<'static>> {
```

and inside it, replacing the `MessageStatus::Streaming` arm written in Task 3:

```rust
        MessageStatus::Streaming => match waiting {
            // Nothing has arrived yet: say what is being waited for, where the text will go.
            Some(waiting) if message.content.is_empty() => body.push(waiting.line()),
            _ => match body.last_mut() {
                Some(last) if last.width() < inner => last.push_span(Span::styled("▍", dim)),
                _ => body.push(Line::styled("▍", dim)),
            },
        },
```

In `refresh`, pass `self.waiting.as_ref()` when the message is the last one and
`MessageStatus::Streaming`, and `None` otherwise — read the loop at 108-130 and thread it
through the existing `message_lines_marked` call. Keep `pub fn message_lines` as the
three-argument convenience wrapper by giving it `None` for both `mark` and `waiting`, so the
tests written in Task 3 keep compiling.

- [ ] **Step 5: Run the transcript tests to verify they pass**

Run: `cargo test --lib transcript`
Expected: PASS, the six new tests included.

- [ ] **Step 6: Write the failing app tests**

Append inside `#[cfg(test)] mod tests` in `src/app.rs`:

```rust
    /// Time must come from the tick count, never from a clock inside `App`: that is what
    /// makes this deterministic. Thirty ticks is one second at `TICK_FPS = 30`.
    #[test]
    fn the_waiting_clock_is_driven_by_ticks() {
        let mut app = sized_app();
        let job = send(&mut app, "Raconte");
        app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Phase(crate::llm::Phase::Connecting),
        });
        assert_eq!(app.waiting().map(|w| w.elapsed_s), Some(None));

        for _ in 0..29 {
            app.update(Action::Tick);
        }
        assert_eq!(
            app.waiting().and_then(|w| w.elapsed_s),
            None,
            "under a second, no number"
        );
        app.update(Action::Tick);
        assert_eq!(app.waiting().and_then(|w| w.elapsed_s), Some(1));
    }

    /// The frame advances every third tick, not every tick: ten frames a second, and the
    /// 30 fps markdown cap survives.
    #[test]
    fn the_frame_advances_every_third_tick() {
        let mut app = sized_app();
        let job = send(&mut app, "Raconte");
        app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Phase(crate::llm::Phase::Connecting),
        });
        let frame = app.waiting().map(|w| w.frame).expect("waiting");
        app.update(Action::Tick);
        app.update(Action::Tick);
        assert_eq!(app.waiting().map(|w| w.frame), Some(frame), "same frame");
        app.update(Action::Tick);
        assert_eq!(app.waiting().map(|w| w.frame), Some(frame + 1));
    }

    /// Review Focus 4: a phase that arrives late must not resurrect a waiting line on a
    /// reply that is already streaming or finished.
    #[test]
    fn a_late_phase_does_not_resurrect_the_waiting_line() {
        let mut app = sized_app();
        let job = send(&mut app, "Raconte");
        token(&mut app, job.request_id, "Il était");
        assert!(app.waiting().is_none(), "the first token clears it");

        app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Phase(crate::llm::Phase::Retrieving { collections: 2 }),
        });
        assert!(
            app.waiting().is_none(),
            "a phase after the first token is stale"
        );
    }

    /// Done, Error and cancellation all end the wait.
    #[test]
    fn finishing_clears_the_waiting_line() {
        for event in [LlmEvent::Done, LlmEvent::Error("panne".to_owned())] {
            let mut app = sized_app();
            let job = send(&mut app, "Raconte");
            app.update(Action::Llm {
                request_id: job.request_id,
                event: LlmEvent::Phase(crate::llm::Phase::Connecting),
            });
            assert!(app.waiting().is_some());
            app.update(Action::Llm {
                request_id: job.request_id,
                event,
            });
            assert!(app.waiting().is_none());
        }

        let mut app = sized_app();
        let job = send(&mut app, "Raconte");
        app.update(Action::Llm {
            request_id: job.request_id,
            event: LlmEvent::Phase(crate::llm::Phase::Connecting),
        });
        app.update(Action::Cancel);
        assert!(app.waiting().is_none(), "Esc ends the wait");
    }
```

`sized_app()`, `send(&mut app, "text")` — which returns the job, whose `request_id` field is
what `Action::Llm` needs — and `token(&mut app, request_id, "text")` are the three helpers
this module's streaming tests already use; read
`tokens_are_rendered_on_the_next_tick_only` (`src/app.rs:3654`) for the exact shape. `app()`
also exists but returns an idle `App` with no job, so it is the wrong one here.

- [ ] **Step 7: Run them to verify they fail**

Run: `cargo test --lib app`
Expected: FAIL to compile — `no method named waiting found for struct App`.

- [ ] **Step 8: Add the counter and the handling**

In `src/app.rs`, add to the `App` struct:

```rust
    /// Which wait is in progress, and how long it has lasted in ticks. `None` outside a
    /// wait. The tick count is the only clock `App` has: `update` must stay pure, so
    /// nothing here reads `Instant::now()`.
    ///
    /// Named `wait`, not `waiting`, because `waiting()` is the method that reduces it for
    /// the renderer — a field and a method of the same name compile but read as a trap.
    /// Public like `pulling`: the views read it through `waiting()`.
    pub wait: Option<(crate::llm::Phase, u32)>,
```

Initialise it to `None` in the constructor beside the other fields.

Replace `Action::Tick => Vec::new(),` (1245) with:

```rust
            Action::Tick => {
                if let Some((_, ticks)) = &mut self.wait {
                    *ticks = ticks.saturating_add(1);
                }
                Vec::new()
            }
```

In `on_llm_event`, add the `Phase` arm and clear the wait where a reply starts or ends:

```rust
            LlmEvent::Phase(phase) => {
                // A phase arriving after the first token is stale: the reply is already on
                // screen and a waiting line would contradict it.
                let started = self
                    .conversation
                    .get(generation.message_id)
                    .is_some_and(|m| !m.content.is_empty());
                if !started {
                    self.wait = Some((phase, 0));
                }
                Vec::new()
            }
```

and in the existing `LlmEvent::Token` arm, as its first statement:

```rust
                self.wait = None;
```

`LlmEvent::Done` and `LlmEvent::Error` must each set `self.wait = None;`. Cancellation goes
through `cancel_generation`; set it to `None` there too, so `Esc` leaves no line behind. A
refused tool call ends up on the same path — `on_tool_call`'s refusal branch — and must also
clear it, which the flow test in Step 15 pins.

Add the accessor:

```rust
    /// The waiting line to show, derived from the tick count. `frame` turns about ten times
    /// a second and the seconds appear only past one, so the value only changes every third
    /// tick — which is what keeps the redraw off the 30 fps path.
    pub fn waiting(&self) -> Option<crate::transcript::Waiting> {
        let (phase, ticks) = self.wait.as_ref()?;
        Some(crate::transcript::Waiting {
            phase: phase.clone(),
            frame: (ticks / 3) as usize,
            elapsed_s: (*ticks >= 30).then(|| ticks / 30),
        })
    }
```

Finally, in `refresh_view` (745), hand it to the transcript beside `set_marks`:

```rust
        self.transcript.set_waiting(self.waiting());
```

- [ ] **Step 9: Run the app tests to verify they pass**

Run: `cargo test --lib app`
Expected: PASS for the four new tests, and `tokens_are_rendered_on_the_next_tick_only`
still passing — if that one breaks, the markdown cap has been disturbed and the cause must
be found before continuing.

- [ ] **Step 10: Write the failing tool-card test**

In `src/transcript.rs`'s tests:

```rust
    /// A tool the user already approved is executing, and must stop asking for approval.
    #[test]
    fn the_tool_card_tells_waiting_from_running() {
        let mut message = Message::new(MessageId(4), Role::Tool, String::new());
        message.source = Some("read_file".to_owned());
        message.status = MessageStatus::Streaming;

        let mut transcript = Transcript::default();
        transcript.set_waiting(None);
        transcript.refresh(&[message.clone()], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(text.contains("en attente de votre accord"), "{text}");

        let mut transcript = Transcript::default();
        transcript.set_waiting(Some(Waiting {
            phase: crate::llm::Phase::RunningTool {
                name: "read_file".to_owned(),
            },
            frame: 0,
            elapsed_s: None,
        }));
        transcript.refresh(&[message], 60, 0);
        let text = transcript.line_texts().join("\n");
        assert!(text.contains("exécution"), "{text}");
        assert!(!text.contains("en attente de votre accord"), "{text}");
    }
```

- [ ] **Step 11: Run it, then make the card read the phase**

Run: `cargo test --lib transcript::tests::the_tool_card`
Expected: FAIL — the card says `en attente de votre accord…` in both cases.

Change `tool_card` to take the waiting state and pick its wording:

```rust
/// A tool call is shown as a one-line card: what it is doing, or what it returned.
fn tool_card(message: &Message, width: usize, waiting: Option<&Waiting>) -> Vec<Line<'static>> {
    let palette = crate::theme::palette();
    let what = message.source.as_deref().unwrap_or("outil");
    let running = matches!(
        waiting.map(|w| &w.phase),
        Some(crate::llm::Phase::RunningTool { .. })
    );
    let (text, color) = match &message.status {
        // `Streaming` covers both halves of a tool call: waiting for the user's decision,
        // and running once it is given. Only the phase can tell them apart.
        MessageStatus::Streaming if running => {
            let glyph = SPINNER[waiting.map_or(0, |w| w.frame) % SPINNER.len()];
            (format!("🔧 {what} · {glyph} exécution…"), palette.warn)
        }
        MessageStatus::Streaming => (
            format!("🔧 {what} · en attente de votre accord…"),
            palette.warn,
        ),
        MessageStatus::Complete => (
            format!(
                "🔧 {what} · ≈ {} tokens transmis",
                tokens::format_count(tokens::estimate(&message.content))
            ),
            palette.warn,
        ),
        MessageStatus::Failed(reason) => (format!("🔧 {what} · {reason}"), palette.dim),
        MessageStatus::Cancelled => (format!("🔧 {what} · annulé"), palette.dim),
    };
    wrap_spans(&[Span::styled(text, Style::default().fg(color))], width)
        .into_iter()
        .map(Line::from)
        .collect()
}
```

and pass `waiting` to it from `message_lines_marked`'s `Role::Tool` arm. A `Role::Tool`
message is never the empty streaming assistant message, so `refresh` must pass the waiting
state to the tool card even when it would pass `None` to the body — thread it to every
message, and let `message_lines_marked` decide which one uses it.

- [ ] **Step 12: Run it to verify it passes**

Run: `cargo test --lib transcript`
Expected: PASS.

- [ ] **Step 13: Make the status bar's spinner turn**

Replace `src/ui/status_bar.rs:20-28` with:

```rust
    let (label, color) = if app.is_generating() {
        let speed = app
            .live_speed()
            .map(|s| format!(" {s} t/s"))
            .unwrap_or_default();
        // The glyph was fixed, which reads as a hung program. It turns with the same clock
        // as the waiting line, so the two never disagree.
        const FRAMES: [&str; 4] = ["◐", "◓", "◑", "◒"];
        let frame = app.waiting().map_or(0, |w| w.frame) % FRAMES.len();
        (
            format!("{} Génération…{speed}", FRAMES[frame]),
            crate::theme::palette().warn,
        )
    } else {
```

Once the first token arrives `app.waiting()` is `None` and the glyph settles on `◐`. That is
deliberate: during generation the text itself is moving, so a turning glyph beside it adds
nothing.

- [ ] **Step 14: Do not add a status-bar unit test — and know why**

There is deliberately no unit test for the turning glyph in `src/ui/status_bar.rs`. Its
`mod tests` drives the bar by writing public fields directly (`app.pulling = …`,
`status_bar.rs:347`), and `test_app()` returns an idle `App`. The spinner branch only runs
when `app.is_generating()` is true, which needs a real generation — so the test would need
either a test-only setter on `App` or a scaffold this module has never had, and it would
prove nothing that two other tests do not already prove: `the_frame_advances_every_third_tick`
(Step 6) pins the arithmetic, and the `waiting_before_the_first_token` snapshot (Task 5)
pins the rendering. A test that needs scaffolding to exist and duplicates its neighbours is
worse than its absence.

For the same reason, make the field public rather than reaching around it:

```rust
    pub wait: Option<(crate::llm::Phase, u32)>,
```

`App` already exposes the state its views read as public fields — `pulling`, `models`,
`collections`. `wait` follows them.

- [ ] **Step 15: Write the end-to-end flow test**

In `tests/app_flow.rs`:

```rust
/// The whole point: the screen must stop being still. A reply that has not started yet
/// says which wait it is in, and the line is replaced by the text in place.
#[tokio::test]
async fn the_wait_before_the_first_token_is_visible() {
    let mut harness = Harness::new(Arc::new(MockLlmClient::new(vec!["Il était une fois"])));
    submit(&mut harness.app, "Raconte une histoire");

    harness.app.update(Action::Llm {
        request_id: 1,
        event: chatatui::llm::LlmEvent::Phase(chatatui::llm::Phase::Waiting {
            model: "llama3.2".to_owned(),
        }),
    });
    for _ in 0..60 {
        harness.app.update(Action::Tick);
    }
    let text = harness.app.transcript.line_texts().join("\n");
    assert!(text.contains("llama3.2 réfléchit…"), "{text}");
    assert!(text.contains("2 s"), "{text}");

    harness.app.update(Action::Llm {
        request_id: 1,
        event: chatatui::llm::LlmEvent::Token("Il était".to_owned()),
    });
    harness.app.update(Action::Tick);
    let text = harness.app.transcript.line_texts().join("\n");
    assert!(!text.contains("réfléchit"), "replaced in place: {text}");
    assert!(text.contains("Il était"), "{text}");
}
```

`request_id: 1` must match what the harness's first job uses; read `step()` and the
`reply_is_streamed_into_the_conversation` test (286) and take the id the same way they do
rather than hard-coding it if they do not.

- [ ] **Step 16: Run this task's checks**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test --lib transcript --lib app --lib markdown --lib theme --lib ui::status_bar && cargo test --test app_flow`
Expected: no warnings, those modules and `app_flow` green.

Then run `cargo test` and record it. **The snapshots still fail; that is Task 5's job.** List
the failing names in your report.

---

### Task 5: Margins, input chrome, the thirty snapshots and the docs

**Files:**
- Modify: `src/ui/chat.rs` (the chat area's margins)
- Modify: `src/ui/prompt_view.rs` (the input's chrome)
- Modify: every file under `src/ui/snapshots/`
- Modify: `README.md`, `PLAN.md`, `docs/roadmap.md`

**Interfaces:**
- Consumes: everything Tasks 1-4 produce.
- Produces: a green `cargo test`.

- [ ] **Step 1: Reduce the input's chrome to a single rule**

`src/ui/prompt_view.rs` draws a full bordered block titled `Message`. The editorial
direction keeps one horizontal rule above the input instead of a box around it. Read the
file — it is 76 lines — and replace the `Block::bordered()` with
`Block::new().borders(Borders::TOP)`, keeping the same inner area computation so the
textarea's height does not change. Keep the title: it moves onto the rule.

- [ ] **Step 2: Check the layout by eye before touching any snapshot**

Run: `cargo test --lib ui::tests::empty_screen 2>&1 | head -40`
Expected: a diff. **Read it.** The input must still be one line tall for an empty message,
the rule must span the full width, and the status bar must be untouched. If the textarea
lost or gained a row, fix `prompt_view.rs` before going further — a snapshot accepted now
freezes the mistake.

- [ ] **Step 3: Accept the snapshots one at a time**

For each failing snapshot, in this order — simplest layouts first, so a systematic error
shows up early:

```
empty_screen  markdown_reply  streaming_reply  cancelled_reply  network_error
conversation_with_multiline_input  scrolled_up_shows_hint  status_bar_gauge
reply_with_sources_and_rag_segment  cleared_context_with_attachment
tool_call_waiting_for_confirmation  find_bar_highlights_matches  slash_suggestions
compare_streaming  compare_done  help_screen  command_palette
model_picker_with_filter  model_picker_error  sidebar_with_conversations
sidebar_loading  sidebar_search_with_excerpt_and_delete_confirmation
sidebar_previews_the_highlighted_conversation  context_popup  prompt_popup
collections_popup_with_last_report  mcp_popup_lists_servers_and_tools
models_popup_lists_what_is_downloaded  gguf_picker_lists_the_files_of_a_repository
status_bar_shows_indexing_progress
```

For each one:

```bash
cargo test --lib ui::tests::<name> 2>&1 | sed -n '/Snapshot Summary/,/^To update/p'
```

Read the diff. Ask of it: is the question at column 8 and the reply at column 2, is the
heading's rule the width of its text, did any line pass the right edge, did a popup grow
past the pane. Only then:

```bash
INSTA_UPDATE=always cargo test --lib ui::tests::<name>
```

**Never run `INSTA_UPDATE=always cargo test`.** Accepting thirty snapshots unread is
accepting whatever bug is in them, and the whole point of this task is that the rendering
changed on purpose and must be checked on purpose.

- [ ] **Step 4: Add the four missing snapshots**

Four states nothing covers today. Add each as a test in `src/ui/mod.rs`'s `mod tests`,
following the shape of `streaming_reply` (166) — which calls `draw(&mut app, …)`, and `draw`
dispatches `Action::Tick` before drawing (101-110):

```rust
    /// The wait before the first token: the state that used to be a frozen screen.
    #[test]
    fn waiting_before_the_first_token() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Raconte une histoire", &[]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Phase(crate::llm::Phase::Waiting {
                model: "llama3.2".into(),
            }),
        });
        // 90 ticks at 30 fps: the line must read `3 s`.
        for _ in 0..90 {
            app.update(Action::Tick);
        }
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }

    /// Retrieval, which runs before the first HTTP byte and used to show nothing.
    #[test]
    fn waiting_on_retrieval_names_the_collections() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Résume le cours", &[]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Phase(crate::llm::Phase::Retrieving { collections: 2 }),
        });
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }

    /// A tool actually running, as opposed to awaiting approval — the label that was wrong.
    #[test]
    fn a_tool_that_is_running_says_so() {
        let mut config = Config::default();
        config.tools.enabled = true;
        let mut app = App::new(&config, false);
        let id = stream(&mut app, "Que dit mon plan ?", &["Je regarde."]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::ToolCall(crate::llm::ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"~/cours/plan.md"}"#.into(),
            }),
        });
        app.update(Action::ToolAnswer {
            allow: true,
            always: false,
        });
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Phase(crate::llm::Phase::RunningTool {
                name: "read_file".into(),
            }),
        });
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }

    /// A code block whose fence has not closed yet: the label line is there, the code is
    /// indented, and the cursor sits after the last code line.
    #[test]
    fn an_open_code_block_while_it_streams() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Montre-moi", &[]);
        for token in ["Voici :\n\n```rust\n", "let v = vec![1];\n"] {
            app.update(Action::Llm {
                request_id: id,
                event: LlmEvent::Token(token.into()),
            });
        }
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }
```

`stream(&mut app, question, replies) -> request_id` is this module's own helper, used by
`tool_call_waiting_for_confirmation` (`src/ui/mod.rs:761`); passing an empty `replies` slice
starts the job without feeding any token, which is exactly the state these snapshots want.
`ToolCall` has no constructor and is built literally, the way that same test does it.

- [ ] **Step 5: Read and accept the four new snapshots**

Run each with `cargo test --lib ui::tests::<name>`, read the generated frame in full, then
accept it with `INSTA_UPDATE=always cargo test --lib ui::tests::<name>`.

Check in particular: the waiting line is at column 2 and dim; `90` ticks reads `3 s`; the
open code block shows `rust` with no number (one block) and its code at a four-space
indent; the tool card shows `exécution…` and not `en attente de votre accord…`.

- [ ] **Step 6: Run the whole suite**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`
Expected: all three clean, and more tests than the 491 the branch started with.

- [ ] **Step 7: Update the documentation**

`README.md`: the section describing the interface. Say that role is read from position —
the question indented and dim, the reply at the margin — that headings carry a rule, that
code blocks are indented with their language and number above, and that the application
names the wait it is in before the first token. Drop any sentence describing the `▌ Vous` /
`▌ Assistant` headers or the `▎` code gutter.

`docs/roadmap.md`: add J33 to the `## Done` list, in the style of the J29-J32 entries —
one or two lines, English, factual.

`PLAN.md`: add the J33 row to the Milestones table in the exact style of J29-J32, and update
the `## Decisions` bullet that reads `UI strings in French, code and comments in English. No
unwrap() outside tests.` only if this milestone changed it — it did not, so leave it. Add a
bullet recording the two decisions a reader would otherwise have to reverse-engineer: both
palettes pin indexed tones because basic ANSI colours are terminal-dependent, and the
waiting line's clock is the tick count because `App::update` reads no clock.

- [ ] **Step 8: Run the final checks and show them**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`
Expected: all clean. Paste the test counts.

- [ ] **Step 9: Manual end-to-end check — the orchestrator's, not a subagent's**

This one needs a real terminal and cannot be automated. It is reported, not performed:

```
cargo run
```

Ask something of a slow local model and watch the wait: the glyph must turn, the phase must
name itself, the seconds must climb, and the text must replace the line in place. Then
resize the terminal narrow and confirm nothing overflows, and run it once with
`theme = "light"` in the configuration to see the light palette.
