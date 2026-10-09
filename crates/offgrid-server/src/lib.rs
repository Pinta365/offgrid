//! OpenAI-compatible HTTP server for offgrid.
//!
//! Endpoints: `POST /v1/chat/completions` (incl. SSE streaming with `reasoning_content`
//! and `tool_calls` deltas), `POST /v1/embeddings`, `GET /v1/models`, `GET /health`.

mod openai;
mod pool;

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use offgrid::{
    ChatRequest, EmbedParams, Embedder, GpuLayers, Model, ModelParams, SessionParams, StreamEvent,
};
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub use openai::ApiError;
use pool::ChatPool;

/// Server configuration (CLI flags map 1:1).
#[derive(Debug, Clone, Default)]
pub struct ServerConfig {
    pub model: Option<PathBuf>,
    /// Multimodal projector (image/audio) for `model`.
    pub mmproj: Option<PathBuf>,
    pub embedding_model: Option<PathBuf>,
    /// Model id reported to clients (default: file stem).
    pub alias: Option<String>,
    pub embedding_alias: Option<String>,
    pub model_params: ModelParams,
    pub session: SessionParams,
    /// Concurrent chat sessions (each holds its own KV cache).
    pub parallel: usize,
    /// Require `Authorization: Bearer <key>` on /v1/*.
    pub api_key: Option<String>,
    /// Max request body in bytes (base64 images/audio are large). 0 = default (64 MiB).
    pub max_body_bytes: usize,
}

/// Default request size limit: room for base64-encoded photos and minutes of audio.
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 << 20;

pub struct AppState {
    max_body_bytes: usize,
    /// Set on shutdown: queued requests are refused instead of starting generation.
    shutting_down: std::sync::atomic::AtomicBool,
    chat: Option<(String, Arc<ChatPool>)>,
    embed: Option<(String, Arc<Mutex<Embedder>>)>,
    api_key: Option<String>,
    created: u64,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn model_id(path: &std::path::Path, alias: &Option<String>) -> String {
    alias.clone().unwrap_or_else(|| path.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned()))
}

impl AppState {
    /// Load models and create sessions (blocking; call before serving).
    pub fn load(config: &ServerConfig) -> offgrid::Result<Arc<Self>> {
        let chat = match &config.model {
            Some(path) => {
                let params = ModelParams { mmproj: config.mmproj.clone(), ..config.model_params.clone() };
                let model = Model::load(path, &params)?;
                let pool = ChatPool::new(model, &config.session, config.parallel)?;
                Some((model_id(path, &config.alias), pool))
            }
            None => None,
        };
        let embed = match &config.embedding_model {
            Some(path) => {
                let params = ModelParams { mmproj: None, ..config.model_params.clone() };
                let model = Model::load(path, &params)?;
                let emb = Embedder::new(model, &EmbedParams::default())?;
                Some((model_id(path, &config.embedding_alias), Arc::new(Mutex::new(emb))))
            }
            None => None,
        };
        if chat.is_none() && embed.is_none() {
            return Err(offgrid::Error::InvalidArgument("no model configured (--model / --embedding-model)".into()));
        }
        Ok(Arc::new(Self {
            shutting_down: std::sync::atomic::AtomicBool::new(false),
            max_body_bytes: if config.max_body_bytes == 0 { DEFAULT_MAX_BODY_BYTES } else { config.max_body_bytes },
            chat,
            embed,
            api_key: config.api_key.clone().filter(|k| !k.is_empty()),
            created: now(),
        }))
    }

    /// Begin shutdown: cancel running generations and refuse queued/new requests.
    pub fn shutdown(&self) {
        self.shutting_down.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some((_, pool)) = &self.chat {
            pool.cancel_all();
        }
    }

    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(std::sync::atomic::Ordering::SeqCst)
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/embeddings", post(embeddings))
        .route("/models", get(list_models))
        .route("/models/{id}", get(get_model))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        // axum's default is 2 MB, too small for base64 media.
        .layer(axum::extract::DefaultBodyLimit::max(state.max_body_bytes));
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .nest("/v1", v1)
        .fallback(|| async { ApiError::not_found("unknown endpoint") })
        .with_state(state)
}

async fn auth(State(state): State<Arc<AppState>>, headers: HeaderMap, req: Request, next: Next) -> Response {
    if let Some(key) = &state.api_key {
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|given| given == key);
        if !ok {
            return ApiError::new(StatusCode::UNAUTHORIZED, "invalid_api_key", "invalid or missing API key").into_response();
        }
    }
    next.run(req).await
}

fn model_entry(state: &AppState, id: &str) -> Value {
    json!({"id": id, "object": "model", "created": state.created, "owned_by": "offgrid"})
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = state
        .chat
        .iter()
        .map(|(id, _)| id)
        .chain(state.embed.iter().map(|(id, _)| id))
        .map(|id| model_entry(&state, id))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn get_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let known = state.chat.iter().map(|(i, _)| i).chain(state.embed.iter().map(|(i, _)| i)).any(|i| *i == id);
    if known { Json(model_entry(&state, &id)).into_response() } else { ApiError::not_found("model not found").into_response() }
}

/// Cancels its request when dropped: the client went away (handler future or SSE body
/// stream dropped). Request-scoped, so firing after the request finished is a no-op and
/// can't affect the session's next request. Aborts prompt processing too.
struct CancelOnDrop(offgrid::RequestCancel);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// SSE body stream that cancels its request when the client disconnects.
struct CancellingStream<S> {
    inner: S,
    _cancel: CancelOnDrop,
}

impl<S: futures_util::Stream + Unpin> futures_util::Stream for CancellingStream<S> {
    type Item = S::Item;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Body rejections (too large, unreadable) as OpenAI-style errors instead of plain text.
fn body_error(status: StatusCode, detail: String) -> Response {
    let kind = if status == StatusCode::PAYLOAD_TOO_LARGE { "request_too_large" } else { "invalid_request_error" };
    ApiError::new(status, kind, detail).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<axum::body::Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let body = match body {
        Ok(b) => b,
        Err(e) => return body_error(e.status(), e.body_text()),
    };
    let Some((model_id, pool)) = state.chat.clone() else {
        return ApiError::bad_request("no chat model loaded").into_response();
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return ApiError::bad_request(format!("invalid JSON body: {e}")).into_response(),
    };
    if body.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
        return ApiError::bad_request("only n=1 is supported").into_response();
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    let req: ChatRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return ApiError::bad_request(format!("invalid request: {e}")).into_response(),
    };
    if req.messages.is_empty() {
        return ApiError::bad_request("messages must not be empty").into_response();
    }
    let meta = openai::Meta::new(model_id);
    let lease = pool.acquire().await;
    // Requests that waited in the queue during shutdown must not start a generation.
    if state.is_shutting_down() {
        return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "server is shutting down").into_response();
    }

    let cancel = CancelOnDrop(lease.next_request_cancel());

    if stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
        tokio::task::spawn_blocking(move || {
            let mut tool_index = 0usize;
            let send = |v: Value| tx.blocking_send(Ok(Event::default().data(v.to_string()))).is_ok();
            if !send(meta.chunk(json!({"role": "assistant", "content": ""}), None)) {
                return;
            }
            let result = lease.session().complete_stream(&req, |event| {
                let delta = match event {
                    StreamEvent::Content(t) => json!({"content": t}),
                    StreamEvent::Reasoning(t) => json!({"reasoning_content": t}),
                    StreamEvent::ToolCall(c) => {
                        tool_index += 1;
                        json!({"tool_calls": [openai::tool_call_delta(tool_index - 1, c)]})
                    }
                };
                // A failed send means the client disconnected: cancel generation.
                send(meta.chunk(delta, None))
            });
            match result {
                Ok(resp) => {
                    send(meta.chunk(json!({}), Some(openai::finish_reason(resp.finish_reason))));
                    if include_usage {
                        send(meta.usage_chunk(&resp.usage));
                    }
                }
                Err(e) => {
                    send(json!({"error": {"message": e.to_string(), "type": "server_error"}}));
                }
            }
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        });
        let body = CancellingStream { inner: ReceiverStream::new(rx), _cancel: cancel };
        Sse::new(body).keep_alive(KeepAlive::default()).into_response()
    } else {
        let _cancel = cancel; // dropped with this future if the client disconnects
        let joined = tokio::task::spawn_blocking(move || lease.session().complete_stream(&req, |_| true)).await;
        match joined {
            Ok(Ok(resp)) => Json(meta.completion(&resp)).into_response(),
            Ok(Err(e)) => ApiError::from_core(e).into_response(),
            Err(e) => ApiError::internal(format!("worker failed: {e}")).into_response(),
        }
    }
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(b)) => b,
        Err(e) => return body_error(e.status(), e.body_text()),
    };
    let Some((model_id, embedder)) = state.embed.clone() else {
        return ApiError::bad_request("no embedding model loaded (--embedding-model)").into_response();
    };
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => {
            items.iter().filter_map(|v| v.as_str().map(String::from)).collect()
        }
        Some(Value::Array(_)) => return ApiError::bad_request("input must be a string or an array of strings (token arrays are not supported)").into_response(),
        _ => return ApiError::bad_request("missing input").into_response(),
    };
    if inputs.is_empty() || inputs.iter().any(String::is_empty) {
        return ApiError::bad_request("input must not be empty").into_response();
    }
    let base64 = match body.get("encoding_format").and_then(Value::as_str) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return ApiError::bad_request(format!("unsupported encoding_format '{other}'")).into_response(),
    };
    let joined = tokio::task::spawn_blocking(move || {
        let mut emb = embedder.lock().unwrap_or_else(|e| e.into_inner());
        let tokens: usize = inputs.iter().map(|s| emb.model().tokenize(s, true, false).map_or(0, |t| t.len())).sum();
        emb.embed(&inputs).map(|v| (v, tokens))
    })
    .await;
    match joined {
        Ok(Ok((vectors, tokens))) => Json(openai::embeddings(&model_id, &vectors, tokens, base64)).into_response(),
        Ok(Err(e)) => ApiError::from_core(e).into_response(),
        Err(e) => ApiError::internal(format!("worker failed: {e}")).into_response(),
    }
}

/// Parse a `--gpu-layers` value: `all`, `none` or a count.
pub fn parse_gpu_layers(s: &str) -> Result<GpuLayers, String> {
    match s {
        "all" => Ok(GpuLayers::All),
        "none" | "0" => Ok(GpuLayers::None),
        n => n.parse().map(GpuLayers::Count).map_err(|_| format!("expected all, none or a number, got '{n}'")),
    }
}
