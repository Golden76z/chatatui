//! Rendering. Every function here takes `&App` and only reads it.

use ratatui::Frame;

use ratatui::layout::{Constraint, Flex, Layout, Rect};

use crate::{app::App, layout, state::Overlay};

mod chat;
mod collections_view;
mod context_view;
mod help;
mod model_picker;
mod palette;
mod prompt_view;
mod sidebar;
mod status_bar;
mod suggestions;
mod text_popup;
mod tool_confirm;

pub use text_popup::max_scroll as popup_max_scroll;

/// Draws the whole screen.
pub fn render(app: &App, frame: &mut Frame) {
    let layout = layout::compute(frame.area(), app.input.lines().len(), app.sidebar.is_some());
    if let (Some(sidebar), Some(area)) = (&app.sidebar, layout.sidebar) {
        sidebar::render(app, sidebar, frame, area);
    }
    chat::render(app, frame, layout.chat);
    frame.render_widget(&app.input, layout.input);
    status_bar::render(app, frame, layout.status);
    let suggestions = app.suggestions();
    if !suggestions.is_empty() {
        suggestions::render(app, &suggestions, frame, layout.chat, layout.input);
    }
    match &app.overlay {
        Some(Overlay::ModelPicker(picker)) => {
            model_picker::render(app, picker, frame, frame.area());
        }
        Some(Overlay::Palette(palette)) => palette::render(app, palette, frame, frame.area()),
        Some(Overlay::ToolConfirm {
            description,
            tool,
            cloud,
        }) => tool_confirm::render(app, description, tool, *cloud, frame, frame.area()),
        Some(
            Overlay::Help { .. }
            | Overlay::Context { .. }
            | Overlay::Prompt { .. }
            | Overlay::Collections { .. },
        ) => {
            text_popup::render(app, frame, frame.area());
        }
        None => {}
    }
}

/// A `width` × `height` rectangle centred in `area` (clamped to it), for popups.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [cell] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(row);
    cell
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::{Terminal, backend::TestBackend};

    use super::*;
    use crate::{
        action::{Action, Effect},
        config::Config,
        llm::{LlmEvent, ModelInfo, ProviderModels, RequestId},
        storage::{ConversationId, ConversationSummary, StoreEvent},
    };

    fn draw(app: &mut App, width: u16, height: u16) -> Terminal<TestBackend> {
        app.update(Action::Resize { width, height });
        app.update(Action::Tick);
        let mut terminal =
            Terminal::new(TestBackend::new(width, height)).expect("test backend never fails");
        terminal
            .draw(|frame| render(app, frame))
            .expect("test backend never fails");
        terminal
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            app.update(Action::Edit(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
    }

    /// Sends `question` and streams `reply` into the app, leaving the generation running.
    fn stream(app: &mut App, question: &str, reply: &[&str]) -> RequestId {
        type_text(app, question);
        let request_id = app
            .update(Action::Submit)
            .into_iter()
            .find_map(|c| match c {
                Effect::StartCompletion(job) => Some(job.request_id),
                _ => None,
            })
            .expect("StartCompletion");
        for token in reply {
            app.update(Action::Llm {
                request_id,
                event: LlmEvent::Token((*token).to_owned()),
            });
        }
        request_id
    }

    #[test]
    fn empty_screen() {
        let mut app = App::new(&Config::default(), false);
        insta::assert_snapshot!(draw(&mut app, 70, 16).backend());
    }

    #[test]
    fn conversation_with_multiline_input() {
        let mut app = App::new(&Config::default(), true);
        let id = stream(
            &mut app,
            "Bonjour, peux-tu m'aider ?",
            &["Bien sûr !", " Que veux-tu savoir ?"],
        );
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        type_text(&mut app, "Première ligne");
        app.update(Action::InsertNewline);
        type_text(&mut app, "Deuxième ligne");
        insta::assert_snapshot!(draw(&mut app, 70, 16).backend());
    }

    #[test]
    fn streaming_reply() {
        let mut app = App::new(&Config::default(), false);
        stream(&mut app, "Raconte une histoire", &["Il était", " une fois"]);
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }

    #[test]
    fn cancelled_reply() {
        let mut app = App::new(&Config::default(), false);
        stream(&mut app, "Raconte une histoire", &["Il était"]);
        app.update(Action::Cancel);
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }

    #[test]
    fn network_error() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Allô ?", &[]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Error("Ollama injoignable sur http://localhost:11434/v1".into()),
        });
        insta::assert_snapshot!(draw(&mut app, 70, 12).backend());
    }

    const MARKDOWN_REPLY: &str = "## Tri en Rust\n\nUtilise **`sort_unstable`** si l'ordre des égaux *n'importe pas* :\n\n```rust\nlet mut v = vec![3, 1, 2];\nv.sort_unstable();\n```\n\n- plus rapide\n- pas d'allocation";

    #[test]
    fn markdown_reply() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Comment trier un Vec ?", &[MARKDOWN_REPLY]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        insta::assert_snapshot!(draw(&mut app, 60, 22).backend());
    }

    #[test]
    fn scrolled_up_shows_hint() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Comment trier un Vec ?", &[MARKDOWN_REPLY]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        draw(&mut app, 60, 12);
        app.update(Action::ScrollToTop);
        insta::assert_snapshot!(draw(&mut app, 60, 12).backend());
    }

    fn summary(id: &str, title: &str, model: &str, age: i64) -> ConversationSummary {
        ConversationSummary {
            id: ConversationId(id.into()),
            title: title.into(),
            provider: "ollama".into(),
            model: model.into(),
            updated_at: 1_000_000 - age,
        }
    }

    #[test]
    fn sidebar_with_conversations() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Comment trier un Vec ?", &["Avec `sort`."]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        let current = app.conversation_id.clone().expect("saved").0;
        app.update(Action::ToggleSidebar);
        app.update(Action::Storage(StoreEvent::Listed {
            conversations: vec![
                summary(&current, "Comment trier un Vec ?", "llama3.2", 10),
                summary(
                    "b",
                    "Recette de la tarte tatin aux pommes caramélisées",
                    "qwen2.5:7b",
                    3 * 3600,
                ),
                ConversationSummary {
                    provider: "claude".into(),
                    ..summary("c", "Plan de cours RAG", "claude-test", 3 * 86_400)
                },
            ],
            now: 1_000_000,
        }));
        app.update(Action::SidebarDown);
        insta::assert_snapshot!(draw(&mut app, 90, 14).backend());
    }

    #[test]
    fn sidebar_loading() {
        let mut app = App::new(&Config::default(), false);
        app.update(Action::ToggleSidebar);
        insta::assert_snapshot!(draw(&mut app, 90, 8).backend());
    }

    #[test]
    fn model_picker_with_filter() {
        let mut app = App::new(&Config::default(), false);
        app.update(Action::OpenModelPicker);
        app.update(Action::ModelsListed(vec![
            ProviderModels {
                provider: "ollama".into(),
                result: Ok(vec![
                    ModelInfo::named("llama3.2"),
                    ModelInfo::named("llama3.1:70b"),
                    ModelInfo::named("qwen2.5:7b"),
                ]),
            },
            ProviderModels {
                provider: "claude".into(),
                result: Ok(vec![ModelInfo {
                    id: "claude-llama-fan".into(),
                    context_window: Some(200_000),
                }]),
            },
            ProviderModels {
                provider: "openai".into(),
                result: Err(
                    "OpenAI : clé API absente (définissez la variable OPENAI_API_KEY)".into(),
                ),
            },
        ]));
        for c in "lla".chars() {
            app.update(Action::OverlayFilter(c));
        }
        app.update(Action::OverlayDown);
        insta::assert_snapshot!(draw(&mut app, 70, 14).backend());
    }

    #[test]
    fn model_picker_error() {
        let mut app = App::new(&Config::default(), false);
        app.update(Action::OpenModelPicker);
        app.update(Action::ModelsListed(vec![ProviderModels {
            provider: "ollama".into(),
            result: Err("Ollama injoignable sur http://localhost:11434/v1".into()),
        }]));
        insta::assert_snapshot!(draw(&mut app, 70, 14).backend());
    }

    #[test]
    fn slash_suggestions() {
        let mut app = App::new(&Config::default(), false);
        type_text(&mut app, "/h");
        app.update(Action::SuggestionDown);
        insta::assert_snapshot!(draw(&mut app, 80, 14).backend());
    }

    #[test]
    fn command_palette() {
        let mut app = App::new(&Config::default(), true);
        app.update(Action::OpenPalette);
        insta::assert_snapshot!(draw(&mut app, 80, 16).backend());
    }

    #[test]
    fn help_screen() {
        let mut app = App::new(&Config::default(), false);
        app.update(Action::OpenHelp);
        insta::assert_snapshot!(draw(&mut app, 80, 26).backend());
    }

    /// A finished exchange with measured usage and a known window.
    fn measured_app(width: u16, height: u16) -> App {
        let mut app = App::new(&Config::default(), false);
        let id = stream(
            &mut app,
            "Explique-moi les lifetimes en Rust",
            &[MARKDOWN_REPLY],
        );
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Usage(crate::llm::Usage {
                input_tokens: Some(3_214),
                output_tokens: Some(186),
            }),
        });
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        app.update(Action::ContextWindowDetected {
            provider: "ollama".into(),
            model: "llama3.2".into(),
            tokens: Some(4_096),
        });
        app.update(Action::Resize { width, height });
        app
    }

    #[test]
    fn status_bar_gauge() {
        let mut app = measured_app(90, 10);
        insta::assert_snapshot!(draw(&mut app, 90, 10).backend());
    }

    #[test]
    fn context_popup() {
        let mut app = measured_app(90, 30);
        app.update(Action::OpenPalette);
        app.update(Action::Cancel);
        type_text(&mut app, "/context");
        app.update(Action::Submit);
        insta::assert_snapshot!(draw(&mut app, 90, 30).backend());
    }

    #[test]
    fn prompt_popup() {
        let mut app = measured_app(90, 30);
        type_text(&mut app, "/prompt");
        app.update(Action::Submit);
        insta::assert_snapshot!(draw(&mut app, 90, 30).backend());
    }

    #[test]
    fn cleared_context_with_attachment() {
        let mut app = App::new(&Config::default(), false);
        let id = stream(&mut app, "Ancienne question", &["Ancienne réponse"]);
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        type_text(&mut app, "/clear");
        app.update(Action::Submit);
        app.update(Action::FileRead(Ok(crate::files::Attachment {
            source: "docs/plan.md".into(),
            content: "x".repeat(5_000),
        })));
        insta::assert_snapshot!(draw(&mut app, 90, 16).backend());
    }

    #[test]
    fn collections_popup_with_last_report() {
        use crate::rag::{indexer::IndexReport, store::CollectionSummary};
        let mut app = App::new(&Config::default(), false);
        app.update(Action::Resize {
            width: 80,
            height: 24,
        });
        let effects = app.run_command(crate::commands::CommandId::Collections, "");
        assert_eq!(
            effects,
            vec![
                Effect::Store(crate::storage::StoreRequest::ListCollections),
                Effect::CheckCollections
            ]
        );
        app.update(Action::Storage(StoreEvent::Collections {
            collections: vec![CollectionSummary {
                name: "rust".into(),
                root: "/home/damien/cours/rust".into(),
                embedding_model: "bge-m3".into(),
                documents: 42,
                chunks: 1_318,
                updated_at: 1_000,
                types: vec!["pdf".into(), "md".into()],
            }],
            now: 1_000 + 2 * 3600,
        }));
        app.stale = vec![crate::rag::indexer::Staleness {
            collection: "rust".into(),
            modified: 2,
            ..Default::default()
        }];
        app.rag_collection = Some("rust".into());
        app.last_index = Some(IndexReport {
            collection: "rust".into(),
            files: 43,
            added: 40,
            updated: 2,
            skipped: vec![("scans/tp1.pdf".into(), "PDF sans texte (scan ?)".into())],
            passages: 1_318,
            ..IndexReport::default()
        });
        insta::assert_snapshot!(draw(&mut app, 80, 24).backend());
    }

    #[test]
    fn status_bar_shows_indexing_progress() {
        use crate::rag::indexer::IndexEvent;
        let mut app = App::new(&Config::default(), false);
        let effects = app.run_command(crate::commands::CommandId::Index, "~/cours/rust");
        assert_eq!(
            effects,
            vec![Effect::StartIndex {
                collection: "rust".into(),
                root: "~/cours/rust".into(),
                types: None
            }]
        );
        app.update(Action::Index(IndexEvent::Progress {
            collection: "rust".into(),
            done: 12,
            total: 40,
            current: "ch03.pdf".into(),
        }));
        insta::assert_snapshot!(draw(&mut app, 100, 8).backend());
        assert_eq!(app.update(Action::Cancel), vec![Effect::CancelIndex]);
    }

    #[test]
    fn reply_with_sources_and_rag_segment() {
        let mut app = App::new(&Config::default(), false);
        app.rag_collection = Some("rust".into());
        let id = stream(&mut app, "Que dit le cours sur l'ownership ?", &[]);
        let chunk = |source: &str, location: &str| crate::context::ContextChunk {
            source: source.into(),
            location: location.into(),
            text: "…".into(),
        };
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Retrieved {
                first_number: 1,
                chunks: vec![
                    chunk("cours/ch04-ownership.pdf", "p. 12"),
                    chunk("plan.docx", "§ Séance 1 : ownership"),
                    chunk("notes.md", "§ Divers"),
                ],
            },
        });
        for token in [
            "Chaque valeur a **un seul** propriétaire [1],",
            " vu en séance 1 [2].",
        ] {
            app.update(Action::Llm {
                request_id: id,
                event: LlmEvent::Token(token.into()),
            });
        }
        app.update(Action::Llm {
            request_id: id,
            event: LlmEvent::Done,
        });
        insta::assert_snapshot!(draw(&mut app, 70, 14).backend());
    }

    #[test]
    fn sidebar_search_with_excerpt_and_delete_confirmation() {
        let mut app = App::new(&Config::default(), false);
        app.update(Action::ToggleSidebar);
        for c in "caramel".chars() {
            app.update(Action::SidebarType(c));
        }
        app.update(Action::Storage(StoreEvent::Searched {
            query: "caramel".into(),
            results: vec![
                (
                    summary("b", "Recette de la tarte tatin", "qwen2.5:7b", 3 * 3600),
                    Some("…faire un caramel à sec, puis…".into()),
                ),
                (
                    summary("c", "Caramel beurre salé", "llama3.2", 86_400),
                    None,
                ),
            ],
            now: 1_000_000,
        }));
        app.update(Action::SidebarDown);
        app.update(Action::SidebarDelete);
        insta::assert_snapshot!(draw(&mut app, 90, 14).backend());
    }

    #[test]
    fn tool_call_waiting_for_confirmation() {
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
        insta::assert_snapshot!(draw(&mut app, 80, 16).backend());
    }
}
