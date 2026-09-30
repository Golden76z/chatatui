//! The background task that produces one assistant reply.
//!
//! Runs outside the UI loop: fetches context, builds the prompt, streams the reply and
//! forwards each fragment as an [`AppEvent::Llm`]. Cancelling the token drops the HTTP
//! stream, which closes the connection and lets the server stop generating.

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_util::sync::CancellationToken;

use super::{
    ChatMessage, ChatRequest, Clients, LlmClient, LlmEvent, RequestId, StreamItem, ToolCall,
};
use crate::{
    context::{ContextProvider, ContextQuery},
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
    /// Document collection searched for this reply (`/rag`), if any.
    pub rag_collection: Option<String>,
    /// Offer the tools (`/tools on`); each call waits for the user's decision.
    pub tools: bool,
}

/// Backends used by the streaming task.
#[derive(Clone)]
pub struct Backends {
    /// One client per provider id.
    pub clients: Clients,
    pub context: Arc<dyn ContextProvider>,
    /// MCP servers and their tools.
    pub mcp: Arc<crate::mcp::Registry>,
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
            mcp: Arc::default(),
        }
    }
}

impl std::fmt::Debug for Backends {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backends").finish_non_exhaustive()
    }
}

/// Tool rounds (reply → calls → results → reply) allowed in one answer.
const MAX_TOOL_ROUNDS: usize = 8;

/// Runs a job until it completes, fails or is cancelled. Sends no event once cancelled.
/// No tools are offered.
pub async fn run(
    backends: Backends,
    job: CompletionJob,
    cancel: CancellationToken,
    events: UnboundedSender<Event>,
) {
    run_with_tools(backends, job, cancel, events, None).await;
}

/// [`run`], with the user's answers to tool calls (`true`: allowed) arriving on
/// `decisions`, one per [`LlmEvent::ToolCall`]. Without it, tools are not offered.
pub async fn run_with_tools(
    backends: Backends,
    job: CompletionJob,
    cancel: CancellationToken,
    events: UnboundedSender<Event>,
    mut decisions: Option<UnboundedReceiver<bool>>,
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
        () = generate(&backends, job, &send, decisions.as_mut()) => {}
    }
}

/// A tool call being received in pieces.
#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

async fn generate(
    backends: &Backends,
    job: CompletionJob,
    send: &impl Fn(LlmEvent),
    mut decisions: Option<&mut UnboundedReceiver<bool>>,
) {
    let mut messages = match job.kind {
        JobKind::Reply => {
            let query = ContextQuery {
                collection: job.rag_collection.as_deref(),
                history: &job.history,
            };
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
        JobKind::Summary => prompt::build_summary_request(&job.history),
    };
    let Some(llm) = backends.clients.get(&job.provider) else {
        return send(LlmEvent::Error(format!(
            "fournisseur inconnu : {}",
            job.provider
        )));
    };
    let tools = if job.tools && job.kind == JobKind::Reply && decisions.is_some() {
        let mut tools = crate::tools::specs();
        tools.extend(backends.mcp.specs());
        tools
    } else {
        Vec::new()
    };
    let started = std::time::Instant::now();
    let millis =
        |since: std::time::Instant| u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX);
    let mut first_token_ms: Option<u64> = None;
    let mut last_timing = started;
    for round in 0..MAX_TOOL_ROUNDS {
        let request = ChatRequest {
            model: job.model.clone(),
            messages: messages.clone(),
            tools: tools.clone(),
        };
        let mut stream = match llm.chat_stream(request).await {
            Ok(stream) => stream,
            Err(error) => return send(LlmEvent::Error(error.to_string())),
        };
        let mut text = String::new();
        let mut calls: std::collections::BTreeMap<usize, PartialCall> = Default::default();
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamItem::Text(token)) => {
                    text.push_str(&token);
                    send(LlmEvent::Token(token));
                    first_token_ms.get_or_insert_with(|| millis(started));
                    if last_timing.elapsed() >= std::time::Duration::from_millis(500) {
                        last_timing = std::time::Instant::now();
                        send(LlmEvent::Timing {
                            first_token_ms,
                            elapsed_ms: millis(started),
                        });
                    }
                }
                Ok(StreamItem::Usage(usage)) => send(LlmEvent::Usage(usage)),
                Ok(StreamItem::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                }) => {
                    let call = calls.entry(index).or_default();
                    call.id.extend(id);
                    call.name.extend(name);
                    call.arguments.push_str(&arguments);
                }
                Err(error) => return send(LlmEvent::Error(error.to_string())),
            }
        }
        drop(stream);
        let Some(decisions) = decisions.as_deref_mut().filter(|_| !calls.is_empty()) else {
            send(LlmEvent::Timing {
                first_token_ms,
                elapsed_ms: millis(started),
            });
            return send(LlmEvent::Done);
        };
        let calls: Vec<ToolCall> = calls
            .into_values()
            .enumerate()
            .map(|(i, c)| ToolCall {
                // Some local servers leave the id out: make one up.
                id: if c.id.is_empty() {
                    format!("call_{round}_{i}")
                } else {
                    c.id
                },
                name: c.name,
                arguments: if c.arguments.trim().is_empty() {
                    "{}".into()
                } else {
                    c.arguments
                },
            })
            .collect();
        messages.push(ChatMessage::tool_request(text, calls.clone()));
        for call in calls {
            send(LlmEvent::ToolCall(call.clone()));
            // The app is gone (shutting down) when the channel closes.
            let Some(allowed) = decisions.recv().await else {
                return;
            };
            let output = if allowed {
                match backends.mcp.call(&call).await {
                    Some(output) => output,
                    None => {
                        crate::tools::run(
                            &call,
                            backends.context.as_ref(),
                            job.rag_collection.as_deref(),
                        )
                        .await
                    }
                }
            } else {
                crate::tools::ToolOutput {
                    ok: false,
                    text: "The user refused this tool call. Answer without it, or ask them.".into(),
                }
            };
            send(LlmEvent::ToolResult {
                call_id: call.id.clone(),
                ok: output.ok,
                output: output.text.clone(),
            });
            messages.push(ChatMessage::tool_result(call.id, output.text));
        }
    }
    send(LlmEvent::Error(format!(
        "trop d'appels d'outils d'affilée ({MAX_TOOL_ROUNDS})"
    )))
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
            rag_collection: None,
            tools: false,
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
            if !matches!(event, LlmEvent::Timing { .. }) {
                events.push(event);
            }
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
        async fn provide(&self, _query: ContextQuery<'_>) -> Result<Context, ContextError> {
            self.0.clone()
        }
    }

    #[tokio::test]
    async fn context_is_injected_into_the_prompt() {
        let llm = Arc::new(MockLlmClient::new([MockReply::tokens(&["ok"])]));
        let context = Context {
            chunks: vec![ContextChunk {
                source: "doc.md".into(),
                location: "p. 2".into(),
                text: "secret fact".into(),
            }],
        };
        let backends = Backends::single(
            "ollama",
            llm.clone(),
            Arc::new(FixedContext(Ok(context.clone()))),
        );
        let events = run_to_end(backends).await;
        assert_eq!(
            events[0],
            LlmEvent::Retrieved {
                first_number: 1,
                chunks: context.chunks
            },
            "retrieved passages are reported first"
        );
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
            if !matches!(event, LlmEvent::Timing { .. }) {
                events.push(event);
            }
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
