//! A provider that never adds anything.

use async_trait::async_trait;

use super::{Context, ContextError, ContextProvider};
use crate::state::Message;

/// Default provider: no additional context.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoContext;

#[async_trait]
impl ContextProvider for NoContext {
    async fn provide(&self, _conversation: &[Message]) -> Result<Context, ContextError> {
        Ok(Context::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_empty_context() {
        let context = NoContext.provide(&[]).await.expect("never fails");
        assert!(context.is_empty());
    }
}
