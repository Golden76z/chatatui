//! User configuration, stored as TOML in the platform config directory
//! (e.g. `~/.config/chatatui/config.toml` on Linux).

use std::{collections::BTreeMap, fs, path::PathBuf};

use color_eyre::{
    Result,
    eyre::{OptionExt, WrapErr},
};
use directories::ProjectDirs;
use serde::Deserialize;

/// Commented default file written on first launch. Must stay in sync with [`Config::default`].
pub const DEFAULT_CONFIG_TOML: &str = r#"# chatatui configuration

# Provider used for new conversations: a name from the [providers] section below.
default_provider = "ollama"

# System prompt sent at the start of every conversation. Use '''...''' for several
# lines, or "" to send none.
system_prompt = "You are a helpful assistant. Answer concisely and use Markdown when useful."

# Seconds to wait for a server to accept the connection.
connect_timeout_secs = 5

# Capture the mouse to scroll with the wheel. While enabled, hold Shift to select text
# with the mouse. Set to false to keep the terminal's native selection.
mouse_capture = true

# Providers. "ollama", "openai" (ChatGPT) and "claude" are predefined: the sections below
# only override their settings. Add your own OpenAI-compatible server the same way, e.g.
#   [providers.lmstudio]
#   label = "LM Studio"
#   base_url = "http://localhost:1234/v1"
#
# Cloud providers need an API key, billed separately from ChatGPT Plus / Claude Pro. It is
# read from an environment variable (recommended) or from `api_key` in this file.

[providers.ollama]
base_url = "http://localhost:11434/v1"
model = "llama3.2"

# [providers.openai]          # key from $OPENAI_API_KEY
# model = "..."

# [providers.claude]          # key from $ANTHROPIC_API_KEY
# model = "..."
# max_output_tokens = 8192
#
# Any provider: `context_window = 32768` sets the context size shown by the gauge, when
# the server cannot report it (OpenAI) or reports it wrongly.
"#;

/// Wire protocol of a provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// OpenAI chat completions (OpenAI, Ollama, llama.cpp, LM Studio, vLLM, …).
    #[default]
    Openai,
    /// Anthropic Messages API (Claude).
    Anthropic,
}

/// A `[providers.<name>]` section; unset fields keep the preset or default value.
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderConfig {
    pub kind: Option<ProviderKind>,
    /// Name shown in the UI (defaults to the section name).
    pub label: Option<String>,
    pub base_url: Option<String>,
    /// Model for new conversations.
    pub model: Option<String>,
    /// API key written in the file (prefer `api_key_env`).
    pub api_key: Option<String>,
    /// Environment variable holding the API key.
    pub api_key_env: Option<String>,
    /// Maximum reply length in tokens (required by the Anthropic API).
    pub max_output_tokens: Option<u32>,
    /// Context window in tokens, when the server cannot tell (overrides detection).
    pub context_window: Option<u64>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the key.
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("label", &self.label)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("api_key_env", &self.api_key_env)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("context_window", &self.context_window)
            .finish()
    }
}

impl ProviderConfig {
    /// Fields set in `other` replace those of `self`.
    fn merged_with(mut self, other: &ProviderConfig) -> Self {
        macro_rules! take {
            ($($field:ident),*) => {
                $(if other.$field.is_some() {
                    self.$field = other.$field.clone();
                })*
            };
        }
        take!(
            kind,
            label,
            base_url,
            model,
            api_key,
            api_key_env,
            max_output_tokens,
            context_window
        );
        self
    }
}

/// A usable provider, after merging presets, the file and the environment.
#[derive(Clone, PartialEq, Eq)]
pub struct Provider {
    /// Section name, e.g. `claude`.
    pub id: String,
    pub label: String,
    pub kind: ProviderKind,
    /// Without trailing slash.
    pub base_url: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    /// Where the key is expected, for error messages.
    pub api_key_env: Option<String>,
    pub max_output_tokens: u32,
    /// Context window set in the configuration.
    pub context_window: Option<u64>,
    /// Runs on this machine (loopback address): nothing leaves the computer.
    pub local: bool,
}

impl Provider {
    /// `true` when the provider needs a key that was not found.
    pub fn missing_key(&self) -> bool {
        self.api_key.is_none() && self.api_key_env.is_some()
    }
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the key.
        f.debug_struct("Provider")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("local", &self.local)
            .finish_non_exhaustive()
    }
}

/// Default reply length limit for providers that require one.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 8192;

/// Built-in providers, in display order.
fn presets() -> Vec<(&'static str, ProviderConfig)> {
    vec![
        (
            "ollama",
            ProviderConfig {
                label: Some("Ollama".into()),
                base_url: Some("http://localhost:11434/v1".into()),
                model: Some("llama3.2".into()),
                ..ProviderConfig::default()
            },
        ),
        (
            "openai",
            ProviderConfig {
                label: Some("OpenAI".into()),
                base_url: Some("https://api.openai.com/v1".into()),
                api_key_env: Some("OPENAI_API_KEY".into()),
                ..ProviderConfig::default()
            },
        ),
        (
            "claude",
            ProviderConfig {
                kind: Some(ProviderKind::Anthropic),
                label: Some("Claude".into()),
                base_url: Some("https://api.anthropic.com/v1".into()),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
                ..ProviderConfig::default()
            },
        ),
    ]
}

/// Application configuration. Missing keys fall back to their defaults.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub default_provider: String,
    pub system_prompt: String,
    pub connect_timeout_secs: u64,
    pub mouse_capture: bool,
    /// `[providers.*]` sections; absent means none (not the default `ollama` section), so
    /// that the legacy top-level keys below still apply.
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    /// Pre-providers layout (`base_url`, `model`, `api_key` at the top level): these
    /// configure the `ollama` provider when it has no section of its own.
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        let ollama = ProviderConfig {
            base_url: Some("http://localhost:11434/v1".into()),
            model: Some("llama3.2".into()),
            ..ProviderConfig::default()
        };
        Self {
            default_provider: "ollama".to_owned(),
            system_prompt:
                "You are a helpful assistant. Answer concisely and use Markdown when useful."
                    .to_owned(),
            connect_timeout_secs: 5,
            mouse_capture: true,
            providers: BTreeMap::from([("ollama".to_owned(), ollama)]),
            base_url: None,
            model: None,
            api_key: None,
        }
    }
}

impl Config {
    /// A configuration whose default provider is one OpenAI-compatible server (tests).
    pub fn single(base_url: &str, model: &str, api_key: Option<&str>) -> Self {
        let provider = ProviderConfig {
            base_url: Some(base_url.to_owned()),
            model: Some(model.to_owned()),
            api_key: api_key.map(str::to_owned),
            ..ProviderConfig::default()
        };
        Self {
            providers: BTreeMap::from([("ollama".to_owned(), provider)]),
            ..Self::default()
        }
    }

    /// All providers: presets, then custom sections; keys looked up with `env`.
    pub fn resolve_providers(&self, env: impl Fn(&str) -> Option<String>) -> Vec<Provider> {
        let mut sections: Vec<(String, ProviderConfig)> = presets()
            .into_iter()
            .map(|(id, preset)| (id.to_owned(), preset))
            .collect();
        for (id, section) in &self.providers {
            match sections.iter_mut().find(|(existing, _)| existing == id) {
                Some((_, preset)) => *preset = preset.clone().merged_with(section),
                None => sections.push((id.clone(), section.clone())),
            }
        }
        if !self.providers.contains_key("ollama") {
            let legacy = ProviderConfig {
                base_url: self.base_url.clone(),
                model: self.model.clone(),
                api_key: self.api_key.clone(),
                ..ProviderConfig::default()
            };
            if let Some((_, ollama)) = sections.iter_mut().find(|(id, _)| id == "ollama") {
                *ollama = ollama.clone().merged_with(&legacy);
            }
        }

        sections
            .into_iter()
            .map(|(id, section)| {
                let kind = section.kind.unwrap_or_default();
                let base_url = section
                    .base_url
                    .unwrap_or_else(|| match kind {
                        ProviderKind::Openai => "http://localhost:8080/v1".into(),
                        ProviderKind::Anthropic => "https://api.anthropic.com/v1".into(),
                    })
                    .trim_end_matches('/')
                    .to_owned();
                let api_key = section
                    .api_key
                    .filter(|k| !k.trim().is_empty())
                    .or_else(|| section.api_key_env.as_deref().and_then(&env))
                    .filter(|k| !k.trim().is_empty());
                Provider {
                    label: section.label.unwrap_or_else(|| id.clone()),
                    local: is_loopback(&base_url),
                    id,
                    kind,
                    base_url,
                    model: section.model.filter(|m| !m.is_empty()),
                    api_key,
                    api_key_env: section.api_key_env,
                    max_output_tokens: section
                        .max_output_tokens
                        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
                    context_window: section.context_window.filter(|n| *n > 0),
                }
            })
            .collect()
    }

    /// The provider for new conversations (falls back to the first one).
    pub fn default_provider<'a>(&self, providers: &'a [Provider]) -> Option<&'a Provider> {
        providers
            .iter()
            .find(|p| p.id == self.default_provider)
            .or_else(|| providers.first())
    }

    /// Parses a configuration from TOML text.
    pub fn from_toml(text: &str) -> Result<Self> {
        toml::from_str(text).wrap_err("invalid configuration file")
    }

    /// Path of the configuration file for the current platform.
    pub fn path() -> Result<PathBuf> {
        let dirs = project_dirs()?;
        Ok(dirs.config_dir().join("config.toml"))
    }

    /// Loads the configuration, writing a commented default file first if none exists.
    pub fn load_or_create() -> Result<Self> {
        let path = Self::path()?;
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .wrap_err_with(|| format!("cannot create {}", parent.display()))?;
            }
            fs::write(&path, DEFAULT_CONFIG_TOML)
                .wrap_err_with(|| format!("cannot write {}", path.display()))?;
        }
        let text = fs::read_to_string(&path)
            .wrap_err_with(|| format!("cannot read {}", path.display()))?;
        Self::from_toml(&text).wrap_err_with(|| format!("in {}", path.display()))
    }
}

/// `true` when `url` points to this machine (`localhost`, `127.0.0.1`, `::1`).
pub fn is_loopback(url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Platform directories for chatatui (config, data).
pub fn project_dirs() -> Result<ProjectDirs> {
    ProjectDirs::from("", "", "chatatui").ok_or_eyre("cannot determine the home directory")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn by_id<'a>(providers: &'a [Provider], id: &str) -> &'a Provider {
        providers
            .iter()
            .find(|p| p.id == id)
            .expect("provider exists")
    }

    #[test]
    fn default_file_matches_default_struct() {
        let parsed = Config::from_toml(DEFAULT_CONFIG_TOML).expect("default config parses");
        assert_eq!(parsed, Config::default());
    }

    #[test]
    fn presets_are_always_available() {
        let providers = Config::default().resolve_providers(no_env);
        let ids: Vec<&str> = providers.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["ollama", "openai", "claude"]);

        let ollama = by_id(&providers, "ollama");
        assert!(ollama.local);
        assert!(!ollama.missing_key());
        assert_eq!(ollama.model.as_deref(), Some("llama3.2"));

        let claude = by_id(&providers, "claude");
        assert_eq!(claude.kind, ProviderKind::Anthropic);
        assert!(!claude.local);
        assert!(claude.missing_key());
        assert_eq!(claude.max_output_tokens, 8192);
    }

    #[test]
    fn keys_come_from_the_environment_or_the_file() {
        let env = |name: &str| (name == "ANTHROPIC_API_KEY").then(|| "sk-ant-test".to_owned());
        let config =
            Config::from_toml("[providers.openai]\napi_key = \"sk-file\"\nmodel = \"gpt-test\"\n")
                .expect("parses");
        let providers = config.resolve_providers(env);
        assert_eq!(
            by_id(&providers, "claude").api_key.as_deref(),
            Some("sk-ant-test")
        );
        let openai = by_id(&providers, "openai");
        assert_eq!(openai.api_key.as_deref(), Some("sk-file"));
        assert_eq!(openai.model.as_deref(), Some("gpt-test"));
        assert_eq!(openai.base_url, "https://api.openai.com/v1", "preset kept");
    }

    #[test]
    fn custom_providers_are_added_after_presets() {
        let config = Config::from_toml(
            "[providers.lmstudio]\nlabel = \"LM Studio\"\nbase_url = \"http://localhost:1234/v1/\"\n",
        )
        .expect("parses");
        let providers = config.resolve_providers(no_env);
        let lmstudio = providers.last().expect("added");
        assert_eq!(lmstudio.id, "lmstudio");
        assert_eq!(lmstudio.label, "LM Studio");
        assert_eq!(lmstudio.base_url, "http://localhost:1234/v1");
        assert_eq!(lmstudio.kind, ProviderKind::Openai);
        assert!(lmstudio.local && !lmstudio.missing_key());
    }

    #[test]
    fn legacy_top_level_keys_configure_ollama() {
        let config =
            Config::from_toml("base_url = \"http://192.168.1.10:11434/v1\"\nmodel = \"qwen2.5\"\n")
                .expect("old layout still parses");
        let providers = config.resolve_providers(no_env);
        let ollama = by_id(&providers, "ollama");
        assert_eq!(ollama.base_url, "http://192.168.1.10:11434/v1");
        assert_eq!(ollama.model.as_deref(), Some("qwen2.5"));
        assert!(!ollama.local, "another machine");
    }

    #[test]
    fn default_provider_falls_back_to_the_first() {
        let config = Config {
            default_provider: "nope".into(),
            ..Config::default()
        };
        let providers = config.resolve_providers(no_env);
        assert_eq!(
            config.default_provider(&providers).map(|p| p.id.as_str()),
            Some("ollama")
        );
    }

    #[test]
    fn keys_are_never_printed() {
        let config = Config::single("http://x/v1", "m", Some("secret"));
        assert!(!format!("{config:?}").contains("secret"));
        let providers = config.resolve_providers(no_env);
        assert!(!format!("{providers:?}").contains("secret"));
    }

    #[test]
    fn multi_line_system_prompt() {
        let text = "system_prompt = '''\nLigne 1\nLigne 2'''\n";
        let parsed = Config::from_toml(text).expect("parses");
        assert_eq!(parsed.system_prompt, "Ligne 1\nLigne 2");
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(Config::from_toml("modle = \"typo\"").is_err());
        assert!(Config::from_toml("[providers.x]\nbase = \"typo\"").is_err());
    }

    #[test]
    fn loopback_detection() {
        for url in [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8080",
            "http://[::1]:1234",
        ] {
            assert!(is_loopback(url), "{url}");
        }
        assert!(!is_loopback("https://api.openai.com/v1"));
        assert!(!is_loopback("not a url"));
    }
}
