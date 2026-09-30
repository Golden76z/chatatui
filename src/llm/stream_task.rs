//! The background task that produces one assistant reply.
//!
//! Runs outside the UI loop: fetches context, builds the prompt, streams the reply and
//! forwards each fragment as an [`AppEvent::Llm`]. Cancelling the token drops the HTTP
//! stream, which closes the connection and lets the server stop generating.

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use super::{ChatRequest, Clients, LlmClient, LlmEvent, RequestId, StreamItem};
use crate::{
    context::ContextProvider,
    event::{AppEvent, Event},
    prompt,
    state::Message,
};

/// What a job produces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JobKind {
    /// An assistant reply to the conversation.
    #[default]
    Reply,
    /// A summary of the conversation (`/compact`); the context provider is not used.
    Summary,
}

/// Everything needed to generate one reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionJob {
    pub kind: JobKind,
    pub request_id: RequestId,
    /// Provider id, key of [`Backends::clients`].
    pub provider: String,
    pub model: String,
    pub system_prompt: String,
    /// Messages in the context, up to and including the new user message.
    pub history: Vec<Message>,
}

/// Backends used by the streaming task.
#[derive(Clone)]
pub struct Backends {
    /// One client per provider id.
    pub clients: Clients,
    pub context: Arc<dyn ContextProvider>,
}

impl Backends {
    /// Backends with a single provider (tests, simple setups).
    pub fn single(
        provider: &str,
        llm: Arc<dyn LlmClient>,
        context: Arc<dyn ContextProvider>,
    ) -> Self {
        Self {
            clients: Clients::from([(provider.to_owned(), llm)]),
            context,
        }
    }
}

impl std::fmt::Debug for Backends {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backends").finish_non_exhaustive()
    }
}

/// Runs a job until it completes, fails or is cancelled. Sends no event once cancelled.
pub async fn run(
    backends: Backends,
    job: CompletionJob,
    cancel: CancellationToken,
    events: UnboundedSender<Event>,
) {
    let request_id = job.request_id;
    let send = |event: LlmEvent| {
        // The receiver is gone only when the app is shutting down.
        let _ = events.send(Event::App(AppEvent::Llm { request_id, event }));
    };
    tokio::select! {
        // `biased` so that a cancellation observed together with a token wins.
        biased;
        () = cancel.cancelled() => {}
        () = generate(&backends, job, &send) => {}
    }
}

async fn generate(backends: &Backends, job: CompletionJob, send: &impl Fn(LlmEvent)) {
    let messages = match job.kind {
        JobKind::Reply => {
            let context = match backends.context.provide(&job.history).await {
                Ok(context) => context,
                Err(error) => return send(LlmEvent::Error(error.to_string())),
            };
            prompt::build_messages(&job.system_prompt, &context, &job.history)
        }
        JobKind::Summary => prompt::build_summary_request(&job.history),
    };
    let Some(llm) = backends.clients.get(&job.provider) else {
        return send(LlmEvent::Error(format!(
            "fournisseur inconnu : {}",
            job.provider
        )));
    };
    let request = ChatRequest {
        model: job.model,
        messages,
    };
    let mut stream = match llm.chat_stream(request).await {
        Ok(stream) => stream,
        Err(error) => return send(LlmEvent::Error(error.to_string())),
    };
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamItem::Text(token)) => send(LlmEvent::Token(token)),
            Ok(StreamItem::Usage(usage)) => send(LlmEvent::Usage(usage)),
            Err(error) => return send(LlmEvent::Error(error.to_string())),
        }
    }
    send(LlmEvent::Done);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        context::{Context, ContextChunk, ContextError, NoContext},
        llm::{
            ChatRole, LlmError,
            mock::{MockLlmClient, MockReply},
        },
        state::{Conversation, MessageStatus, Role},
    };

    fn job() -> CompletionJob {
        let mut conversation = Conversation::new();
        conversation.push(Role::User, "Bonjour", MessageStatus::Complete);
        CompletionJob {
            kind: JobKind::Reply,
            request_id: RequestId(7),
            provider: "ollama".into(),
            model: "test-model".into(),
            system_prompt: "Be brief.".into(),
            history: conversation.messages().to_vec(),
        }
    }

    fn backends(llm: Arc<MockLlmClient>) -> Backends {
        Backends::single("ollama", llm, Arc::new(NoContext))
    }

    /// Runs a job to completion and returns the LLM events it emitted.
    async fn run_to_end(backends: Backends) -> Vec<LlmEvent> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        run(backends, job(), CancellationToken::new(), tx).await;
        let mut events = Vec::new();
        while let Ok(Event::App(AppEvent::Llm { request_id, event })) = rx.try_recv() {
            assert_eq!(request_id, RequestId(7));
            events.push(event);
        }
        events
    }

    fn token(text: &str) -> LlmEvent {
        LlmEvent::Token(text.to_owned())
    }

    #[tokio::test]
    async fn streams_tokens_then_done() {
        let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&["Bon", "jour"])]));
        let events = run_to_end(backends(llm.clone())).await;
        assert_eq!(events, vec![token("Bon"), token("jour"), LlmEvent::Done]);

        let requests = llm.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "test-model");
        let roles: Vec<ChatRole> = requests[0].messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, vec![ChatRole::System, ChatRole::User]);
    }

    #[tokio::test]
    async fn usage_is_forwarded() {
        let usage = crate::llm::Usage {
            input_tokens: Some(40),
            output_tokens: Some(2),
        };
        let llm = Arc::new(MockLlmClient::new([MockReply::TokensWithUsage(
            vec!["ok".into()],
            usage,
        )]));
        let events = run_to_end(backends(llm)).await;
        assert_eq!(
            events,
            vec![token("ok"), LlmEvent::Usage(usage), LlmEvent::Done]
        );
    }

    #[tokio::test]
    async fn unknown_provider_is_reported() {
        let llm = Arc::new(MockLlmClient::new([]));
        let events = run_to_end(Backends::single("other", llm, Arc::new(NoContext))).await;
        assert_eq!(
            events,
            vec![LlmEvent::Error("fournisseur inconnu : ollama".into())]
        );
    }

    #[tokio::test]
    async fn connection_failure_is_reported() {
        let error = LlmError::Unreachable {
            server: "Ollama".into(),
            url: "http://localhost:11434/v1".into(),
        };
        let llm = Arc::new(MockLlmClient::new([MockReply::Fail(error)]));
        let events = run_to_end(backends(llm)).await;
        assert_eq!(
            events,
            vec![LlmEvent::Error(
                "Ollama injoignable sur http://localhost:11434/v1".into()
            )]
        );
    }

    #[tokio::test]
    async fn mid_stream_failure_keeps_previous_tokens() {
        let llm = Arc::new(MockLlmClient::new([MockReply::TokensThenError(
            vec!["a".into()],
            LlmError::Server("overloaded".into()),
        )]));
        let events = run_to_end(backends(llm)).await;
        assert_eq!(
            events,
            vec![
                token("a"),
                LlmEvent::Error("erreur du serveur : overloaded".into())
            ]
        );
    }

    #[tokio::test]
    async fn cancellation_stops_the_task_and_drops_the_stream() {
        let llm = Arc::new(MockLlmClient::new([MockReply::TokensThenHang(vec![
            "a".into(),
        ])]));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run(backends(llm.clone()), job(), cancel.clone(), tx));

        let first = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("first token arrives");
        assert!(matches!(
            first,
            Some(Event::App(AppEvent::Llm {
                event: LlmEvent::Token(_),
                ..
            }))
        ));

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("task stops after cancellation")
            .expect("task does not panic");
        assert_eq!(llm.dropped_streams(), 1, "HTTP stream is dropped");
        assert!(rx.try_recv().is_err(), "no event after cancellation");
    }

    struct FixedContext(Result<Context, ContextError>);

    #[async_trait]
    impl ContextProvider for FixedContext {
        async fn provide(&self, _conversation: &[Message]) -> Result<Context, ContextError> {
            self.0.clone()
        }
    }

    #[tokio::test]
    async fn context_is_injected_into_the_prompt() {
        let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&["ok"])]));
        let context = Context {
            chunks: vec![ContextChunk {
                source: "doc.md".into(),
                text: "secret fact".into(),
            }],
        };
        let backends = Backends::single("ollama", llm.clone(), Arc::new(FixedContext(Ok(context))));
        run_to_end(backends).await;
        let system = &llm.requests()[0].messages[0];
        assert_eq!(system.role, ChatRole::System);
        assert!(system.content.contains("secret fact"));
    }

    #[tokio::test]
    async fn context_failure_is_reported_without_calling_the_llm() {
        let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&["ok"])]));
        let backends = Backends::single(
            "ollama",
            llm.clone(),
            Arc::new(FixedContext(Err(ContextError("index missing".into())))),
        );
        let events = run_to_end(backends).await;
        assert_eq!(
            events,
            vec![LlmEvent::Error(
                "contexte indisponible : index missing".into()
            )]
        );
        assert!(llm.requests().is_empty());
    }

    #[tokio::test]
    async fn summary_jobs_use_the_summary_prompt_without_context() {
        let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&["- résumé"])]));
        let backends = Backends::single(
            "ollama",
            llm.clone(),
            Arc::new(FixedContext(Err(ContextError("must not be called".into())))),
        );
        let (tx, mut rx) = mpsc::unbounded_channel();
        let job = CompletionJob {
            kind: JobKind::Summary,
            ..job()
        };
        run(backends, job, CancellationToken::new(), tx).await;
        let mut events = Vec::new();
        while let Ok(Event::App(AppEvent::Llm { event, .. })) = rx.try_recv() {
            events.push(event);
        }
        assert_eq!(events, vec![token("- résumé"), LlmEvent::Done]);
        let request = &llm.requests()[0];
        assert!(
            request.messages[1]
                .content
                .starts_with("Summarize this conversation")
        );
    }
}
