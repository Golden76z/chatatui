//! User commands: the single registry behind slash commands (`/new`, …), the command
//! palette (Ctrl+P) and the help screen.

/// Identifies a user command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommandId {
    New,
    History,
    Model,
    Context,
    Prompt,
    Add,
    Clear,
    Compact,
    Index,
    Collections,
    Rag,
    Help,
    Quit,
}

/// Whether a command takes an argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    None,
    /// Optional argument, with its placeholder (e.g. `[nom]`).
    Optional(&'static str),
    /// Required argument, with its placeholder (e.g. `<fichier>`).
    Required(&'static str),
}

/// Description of a user command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub id: CommandId,
    /// Name without the slash.
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub arg: Arg,
    pub description: &'static str,
    /// Keyboard shortcut with the kitty keyboard protocol.
    pub shortcut: Option<&'static str>,
    /// Keyboard shortcut in other terminals, when different.
    pub legacy_shortcut: Option<&'static str>,
}

impl CommandSpec {
    /// Shortcut that works in the current terminal.
    pub fn shortcut(&self, keyboard_enhanced: bool) -> Option<&'static str> {
        if keyboard_enhanced {
            self.shortcut
        } else {
            self.legacy_shortcut.or(self.shortcut)
        }
    }

    /// `/name <arg>` as shown in menus.
    pub fn usage(&self) -> String {
        match self.arg {
            Arg::None => format!("/{}", self.name),
            Arg::Optional(arg) | Arg::Required(arg) => format!("/{} {arg}", self.name),
        }
    }

    fn matches_name(&self, name: &str) -> bool {
        self.name == name || self.aliases.contains(&name)
    }
}

/// All commands, in display order.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        id: CommandId::New,
        name: "new",
        aliases: &["nouveau"],
        arg: Arg::None,
        description: "Nouvelle conversation",
        shortcut: Some("Ctrl+N"),
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::History,
        name: "history",
        aliases: &["historique"],
        arg: Arg::None,
        description: "Liste des conversations",
        shortcut: Some("Ctrl+L"),
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Model,
        name: "model",
        aliases: &["modele"],
        arg: Arg::Optional("[nom]"),
        description: "Choisir le modèle (ou le nommer directement)",
        shortcut: Some("Ctrl+M"),
        legacy_shortcut: Some("F2"),
    },
    CommandSpec {
        id: CommandId::Context,
        name: "context",
        aliases: &["contexte", "tokens"],
        arg: Arg::None,
        description: "Taille du contexte et détail",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Prompt,
        name: "prompt",
        aliases: &[],
        arg: Arg::None,
        description: "Voir exactement ce qui part au modèle",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Add,
        name: "add",
        aliases: &["joindre", "attach"],
        arg: Arg::Required("<fichier>"),
        description: "Joindre un fichier texte au contexte (Tab complète le chemin)",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Clear,
        name: "clear",
        aliases: &["vider"],
        arg: Arg::None,
        description: "Vider le contexte (l'historique reste affiché)",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Compact,
        name: "compact",
        aliases: &["resumer", "résumer"],
        arg: Arg::None,
        description: "Résumer l'historique pour libérer le contexte",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Index,
        name: "index",
        aliases: &["indexer"],
        arg: Arg::Required("<dossier> [nom]"),
        description: "Indexer un dossier de documents pour la recherche (RAG)",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Collections,
        name: "collections",
        aliases: &["docs"],
        arg: Arg::None,
        description: "Collections de documents indexées",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Rag,
        name: "rag",
        aliases: &["documents"],
        arg: Arg::Optional("<collection>|off"),
        description: "Répondre à partir d'une collection de documents",
        shortcut: None,
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Help,
        name: "help",
        aliases: &["aide", "?"],
        arg: Arg::None,
        description: "Commandes et raccourcis",
        shortcut: Some("F1"),
        legacy_shortcut: None,
    },
    CommandSpec {
        id: CommandId::Quit,
        name: "quit",
        aliases: &["exit", "quitter"],
        arg: Arg::None,
        description: "Quitter",
        shortcut: Some("Ctrl+C"),
        legacy_shortcut: None,
    },
];

/// Splits an argument into words; `"…"` or `'…'` group words with spaces.
pub fn split_args(arg: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote: Option<char> = None;
    let mut started = false;
    for c in arg.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => word.push(c),
            (None, '"' | '\'') => {
                quote = Some(c);
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started || !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            (None, c) => word.push(c),
        }
    }
    if started || !word.is_empty() {
        words.push(word);
    }
    words
}

/// Looks a command up by name or alias.
pub fn find(name: &str) -> Option<&'static CommandSpec> {
    let name = name.to_lowercase();
    COMMANDS.iter().find(|c| c.matches_name(&name))
}

/// Result of reading the input box as a command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parsed<'a> {
    /// A known command and its (trimmed, possibly empty) argument.
    Command {
        spec: &'static CommandSpec,
        arg: &'a str,
    },
    /// Starts with `/` but names no command.
    Unknown(&'a str),
    /// `//text`: a message that starts with a slash (one slash removed).
    Escaped(&'a str),
    /// A regular message.
    Message,
}

/// Interprets the input. Only single-line input can be a command.
pub fn parse(input: &str) -> Parsed<'_> {
    let text = input.trim();
    if text.contains('\n') {
        return Parsed::Message;
    }
    let Some(rest) = text.strip_prefix('/') else {
        return Parsed::Message;
    };
    if rest.starts_with('/') {
        return Parsed::Escaped(rest);
    }
    let (name, arg) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    match find(name) {
        Some(spec) => Parsed::Command {
            spec,
            arg: arg.trim(),
        },
        None => Parsed::Unknown(name),
    }
}

/// Commands to suggest while the user types `/prefix` (no space yet).
///
/// Names starting with the prefix come first, then aliases starting with it.
pub fn suggestions(input: &str) -> Vec<&'static CommandSpec> {
    if input.contains('\n') {
        return Vec::new();
    }
    let Some(prefix) = input.strip_prefix('/') else {
        return Vec::new();
    };
    if prefix.starts_with('/') || prefix.contains(char::is_whitespace) {
        return Vec::new();
    }
    let prefix = prefix.to_lowercase();
    let by_name = COMMANDS.iter().filter(|c| c.name.starts_with(&prefix));
    let by_alias = COMMANDS.iter().filter(|c| {
        !c.name.starts_with(&prefix) && c.aliases.iter().any(|a| a.starts_with(&prefix))
    });
    by_name.chain(by_alias).collect()
}

/// Commands matching a palette filter (name, alias or description, case-insensitive).
pub fn search(filter: &str) -> Vec<&'static CommandSpec> {
    let filter = filter.trim().trim_start_matches('/').to_lowercase();
    COMMANDS
        .iter()
        .filter(|c| {
            filter.is_empty()
                || c.name.contains(&filter)
                || c.aliases.iter().any(|a| a.contains(&filter))
                || c.description.to_lowercase().contains(&filter)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(specs: &[&CommandSpec]) -> Vec<&'static str> {
        specs.iter().map(|c| c.name).collect()
    }

    #[test]
    fn names_and_aliases_are_unique() {
        let mut all: Vec<&str> = COMMANDS
            .iter()
            .flat_map(|c| std::iter::once(c.name).chain(c.aliases.iter().copied()))
            .collect();
        let count = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), count);
    }

    #[test]
    fn parses_commands_with_arguments() {
        assert_eq!(
            parse("/model  qwen2.5:7b "),
            Parsed::Command {
                spec: find("model").expect("exists"),
                arg: "qwen2.5:7b"
            }
        );
        assert!(
            matches!(parse("/NEW"), Parsed::Command { spec, arg: "" } if spec.id == CommandId::New)
        );
        assert!(
            matches!(parse("/aide"), Parsed::Command { spec, .. } if spec.id == CommandId::Help)
        );
    }

    #[test]
    fn other_inputs() {
        assert_eq!(parse("/nope"), Parsed::Unknown("nope"));
        assert_eq!(parse("//etc/hosts"), Parsed::Escaped("/etc/hosts"));
        assert_eq!(parse("bonjour /new"), Parsed::Message);
        assert_eq!(
            parse("/new\nsuite"),
            Parsed::Message,
            "multi-line is a message"
        );
    }

    #[test]
    fn suggestions_follow_the_prefix() {
        assert_eq!(
            names(&suggestions("/")),
            names(&COMMANDS.iter().collect::<Vec<_>>())
        );
        assert_eq!(names(&suggestions("/m")), vec!["model"]);
        assert_eq!(names(&suggestions("/h")), vec!["history", "help"]);
        assert_eq!(
            names(&suggestions("/co")),
            vec!["context", "compact", "collections"]
        );
        assert_eq!(
            names(&suggestions("/a")),
            vec!["add", "help"],
            "help via its « aide » alias"
        );
        assert_eq!(names(&suggestions("/p")), vec!["prompt"]);
        assert_eq!(
            names(&suggestions("/ex")),
            vec!["quit"],
            "aliases match too"
        );
        assert!(
            suggestions("/model qwen").is_empty(),
            "argument being typed"
        );
        assert!(suggestions("//").is_empty());
        assert!(suggestions("salut").is_empty());
    }

    #[test]
    fn palette_search_includes_descriptions() {
        assert_eq!(names(&search("conversation")), vec!["new", "history"]);
        assert_eq!(
            names(&search("contexte")),
            vec!["context", "add", "clear", "compact"]
        );
        assert_eq!(
            names(&search("/mod")),
            vec!["model", "prompt"],
            "« modèle » in a description"
        );
        assert_eq!(search("").len(), COMMANDS.len());
    }

    #[test]
    fn arguments_with_quotes() {
        assert_eq!(split_args("~/cours rust"), ["~/cours", "rust"]);
        assert_eq!(
            split_args("\"~/Mes documents\"  notes"),
            ["~/Mes documents", "notes"]
        );
        assert_eq!(split_args("'a b'"), ["a b"]);
        assert!(split_args("   ").is_empty());
    }

    #[test]
    fn shortcut_depends_on_the_terminal() {
        let model = find("model").expect("exists");
        assert_eq!(model.shortcut(true), Some("Ctrl+M"));
        assert_eq!(model.shortcut(false), Some("F2"));
        assert_eq!(model.usage(), "/model [nom]");
    }
}
