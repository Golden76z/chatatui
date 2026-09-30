//! The model selection popup: models of every provider, grouped and filterable.

use crate::llm::ProviderModels;

/// A selectable model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelChoice {
    /// Provider id.
    pub provider: String,
    /// Provider label, e.g. `Claude`.
    pub label: String,
    pub model: String,
    /// Context window in tokens, when the server says.
    pub context_window: Option<u64>,
}

impl ModelChoice {
    /// Text matched by the filter: `label model`.
    fn haystack(&self) -> String {
        format!("{} {}", self.label, self.model).to_lowercase()
    }
}

/// What the servers answered, as far as we know.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ModelList {
    /// The model lists are being fetched.
    #[default]
    Loading,
    /// Models of the providers that answered, and one error per provider that did not.
    Loaded {
        choices: Vec<ModelChoice>,
        /// `(provider label, message)`.
        errors: Vec<(String, String)>,
    },
}

/// State of the open popup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelPicker {
    pub models: ModelList,
    /// Case-insensitive words typed by the user; all must match.
    pub filter: String,
    /// Index into [`ModelPicker::visible`].
    pub selected: usize,
}

impl ModelPicker {
    /// Fills the list from each provider's answer (`label_of` maps ids to labels) and
    /// highlights the current model.
    pub fn set_models(
        &mut self,
        results: Vec<ProviderModels>,
        label_of: impl Fn(&str) -> String,
        current: (&str, &str),
    ) {
        let mut choices = Vec::new();
        let mut errors = Vec::new();
        for ProviderModels { provider, result } in results {
            let label = label_of(&provider);
            match result {
                Ok(models) => choices.extend(models.into_iter().map(|m| ModelChoice {
                    provider: provider.clone(),
                    label: label.clone(),
                    model: m.id,
                    context_window: m.context_window,
                })),
                Err(error) => errors.push((label, error)),
            }
        }
        self.models = ModelList::Loaded { choices, errors };
        self.selected = self
            .visible()
            .iter()
            .position(|c| c.provider == current.0 && c.model == current.1)
            .unwrap_or(0);
    }

    /// Models matching the filter, grouped by provider in the order they were listed.
    pub fn visible(&self) -> Vec<&ModelChoice> {
        let ModelList::Loaded { choices, .. } = &self.models else {
            return Vec::new();
        };
        let words: Vec<String> = self
            .filter
            .to_lowercase()
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        choices
            .iter()
            .filter(|c| {
                let haystack = c.haystack();
                words.iter().all(|w| haystack.contains(w.as_str()))
            })
            .collect()
    }

    /// Providers that could not be listed: `(label, message)`.
    pub fn errors(&self) -> &[(String, String)] {
        match &self.models {
            ModelList::Loaded { errors, .. } => errors,
            ModelList::Loading => &[],
        }
    }

    /// The highlighted model.
    pub fn selected_choice(&self) -> Option<&ModelChoice> {
        self.visible().get(self.selected).copied()
    }

    /// Moves the highlight up, wrapping around.
    pub fn select_previous(&mut self) {
        let len = self.visible().len();
        if len > 0 {
            self.selected = (self.selected + len - 1) % len;
        }
    }

    /// Moves the highlight down, wrapping around.
    pub fn select_next(&mut self) {
        let len = self.visible().len();
        if len > 0 {
            self.selected = (self.selected + 1) % len;
        }
    }

    /// Appends to the filter and highlights the first match.
    pub fn push_filter(&mut self, c: char) {
        self.filter.push(c);
        self.selected = 0;
    }

    /// Removes the last filter character.
    pub fn pop_filter(&mut self) {
        self.filter.pop();
        self.selected = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ModelInfo;

    fn label(id: &str) -> String {
        match id {
            "ollama" => "Ollama".into(),
            "claude" => "Claude".into(),
            other => other.into(),
        }
    }

    fn picker() -> ModelPicker {
        let mut picker = ModelPicker::default();
        picker.set_models(
            vec![
                ProviderModels {
                    provider: "ollama".into(),
                    result: Ok(vec![
                        ModelInfo::named("llama3.2"),
                        ModelInfo::named("qwen2.5:7b"),
                        ModelInfo::named("Qwen2.5-Coder"),
                    ]),
                },
                ProviderModels {
                    provider: "claude".into(),
                    result: Ok(vec![ModelInfo {
                        id: "claude-test".into(),
                        context_window: Some(200_000),
                    }]),
                },
                ProviderModels {
                    provider: "openai".into(),
                    result: Err("clé API absente".into()),
                },
            ],
            label,
            ("ollama", "qwen2.5:7b"),
        );
        picker
    }

    fn models(picker: &ModelPicker) -> Vec<&str> {
        picker.visible().iter().map(|c| c.model.as_str()).collect()
    }

    #[test]
    fn current_model_is_highlighted_and_errors_kept() {
        let picker = picker();
        assert_eq!(
            picker.selected_choice().map(|c| c.model.as_str()),
            Some("qwen2.5:7b")
        );
        assert_eq!(
            picker.errors(),
            [("openai".to_owned(), "clé API absente".to_owned())]
        );
        assert_eq!(picker.visible().len(), 4);
    }

    #[test]
    fn same_name_on_another_provider_is_not_the_current_one() {
        let mut picker = ModelPicker::default();
        picker.set_models(
            vec![
                ProviderModels {
                    provider: "a".into(),
                    result: Ok(vec![ModelInfo::named("m")]),
                },
                ProviderModels {
                    provider: "b".into(),
                    result: Ok(vec![ModelInfo::named("m")]),
                },
            ],
            label,
            ("b", "m"),
        );
        assert_eq!(picker.selected, 1);
    }

    #[test]
    fn filter_matches_provider_and_model_words() {
        let mut picker = picker();
        for c in "QWEN".chars() {
            picker.push_filter(c);
        }
        assert_eq!(models(&picker), vec!["qwen2.5:7b", "Qwen2.5-Coder"]);

        picker.filter.clear();
        for c in "claude test".chars() {
            picker.push_filter(c);
        }
        assert_eq!(models(&picker), vec!["claude-test"]);
        picker.pop_filter();
        assert_eq!(picker.selected, 0);
    }

    #[test]
    fn navigation_wraps_and_empty_lists_are_safe() {
        let mut picker = picker();
        picker.selected = 0;
        picker.select_previous();
        assert_eq!(
            picker.selected_choice().map(|c| c.model.as_str()),
            Some("claude-test")
        );

        picker.push_filter('#');
        picker.select_next();
        assert!(picker.selected_choice().is_none());
        assert!(ModelPicker::default().visible().is_empty(), "loading");
    }
}
