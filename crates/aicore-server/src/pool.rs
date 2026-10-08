//! A fixed pool of chat sessions shared by request handlers.

use std::sync::{Arc, Mutex};

use aicore::{CancelToken, ChatSession, Model, RequestCancel, SessionParams};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub struct ChatPool {
    sessions: Vec<Mutex<ChatSession>>,
    cancels: Vec<CancelToken>,
    /// Free session indices; the most recently released is reused first, which keeps a
    /// single client's conversation on the same session (KV-cache prefix reuse).
    free: Mutex<Vec<usize>>,
    permits: Arc<Semaphore>,
}

/// Exclusive use of one session; returned to the pool on drop.
pub struct Lease {
    pool: Arc<ChatPool>,
    index: usize,
    _permit: OwnedSemaphorePermit,
}

impl ChatPool {
    pub fn new(model: Arc<Model>, params: &SessionParams, size: usize) -> aicore::Result<Arc<Self>> {
        let size = size.max(1);
        let mut sessions = Vec::with_capacity(size);
        let mut cancels = Vec::with_capacity(size);
        for _ in 0..size {
            let s = ChatSession::new(model.clone(), params)?;
            cancels.push(s.cancel_token());
            sessions.push(Mutex::new(s));
        }
        Ok(Arc::new(Self {
            sessions,
            cancels,
            free: Mutex::new((0..size).rev().collect()),
            permits: Arc::new(Semaphore::new(size)),
        }))
    }

    /// Wait for a free session.
    pub async fn acquire(self: &Arc<Self>) -> Lease {
        let permit = self.permits.clone().acquire_owned().await.expect("semaphore never closed");
        let index = self.free.lock().unwrap_or_else(|e| e.into_inner()).pop().expect("permit implies a free session");
        Lease { pool: self.clone(), index, _permit: permit }
    }

    /// Cancel every running generation (shutdown).
    pub fn cancel_all(&self) {
        self.cancels.iter().for_each(CancelToken::cancel);
    }
}

impl Lease {
    pub fn session(&self) -> std::sync::MutexGuard<'_, ChatSession> {
        self.pool.sessions[self.index].lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Cancels the next request run on this lease (valid: the lease is exclusive).
    pub fn next_request_cancel(&self) -> RequestCancel {
        self.pool.cancels[self.index].next_request()
    }

}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.free.lock().unwrap_or_else(|e| e.into_inner()).push(self.index);
    }
}
