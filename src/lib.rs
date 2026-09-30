//! chatatui — a minimal terminal chat client for local LLMs.
//!
//! Architecture (see `PLAN.md`):
//!
//! ```text
//! Event ──keymap──▶ Action ──App::update──▶ Vec<Effect> ──Runtime──▶ background tasks
//!   ▲                                                                        │
//!   └──────────────────────── AppEvent (mpsc) ◀──────────────────────────────┘
//! ui::render(&App, frame) only reads state.
//! ```

pub mod action;
pub mod app;
pub mod clipboard;
pub mod commands;
pub mod config;
pub mod context;
pub mod event;
pub mod export;
pub mod files;
pub mod keymap;
pub mod layout;
pub mod llm;
pub mod markdown;
pub mod prompt;
pub mod rag;
pub mod runtime;
pub mod state;
pub mod storage;
pub mod terminal;
pub mod theme;
pub mod tokens;
pub mod transcript;
pub mod ui;
