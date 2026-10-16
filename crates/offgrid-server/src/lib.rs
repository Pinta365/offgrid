//! OpenAI-compatible HTTP server for offgrid.
//!
//! Endpoints: `POST /v1/chat/completions` (incl. SSE streaming with `reasoning_content`
//! and `tool_calls` deltas), `POST /v1/completions` (raw text, or fill-in-the-middle with
//! `suffix`), `POST /infill` (llama.cpp-compatible), `POST /v1/embeddings`,
//! `GET /v1/models`, `GET /health`.

mod completions;
#[cfg(feature = "image")]
mod images;
mod openai;
mod pool;

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use offgrid::{ChatRequest, EmbedParams, Embedder, GpuLayers, Model, ModelParams, SessionParams, StreamEvent};
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
    /// Code model for fill-in-the-middle (/infill, /v1/completions); default: `model`.
    pub fim_model: Option<PathBuf>,
    /// Model id reported to clients (default: file stem).
    pub alias: Option<String>,
    pub embedding_alias: Option<String>,
    pub fim_alias: Option<String>,
    pub model_params: ModelParams,
    pub session: SessionParams,
    /// Concurrent chat sessions (each holds its own KV cache).
    pub parallel: usize,
    /// Require `Authorization: Bearer <key>` on /v1/*.
    pub api_key: Option<String>,
    /// Max request body in bytes (base64 images/audio are large). 0 = default (64 MiB).
    pub max_body_bytes: usize,
    /// Image generation model for /v1/images/generations.
    #[cfg(feature = "image")]
    pub image: Option<offgrid::ImageModelParams>,
    #[cfg(feature = "image")]
    pub image_alias: Option<String>,
}

/// Default request size limit: room for base64-encoded photos and minutes of audio.
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 << 20;

pub struct AppState {
    max_body_bytes: usize,
    /// Set on shutdown: queued requests are refused instead of starting generation.
    shutting_down: std::sync::atomic::AtomicBool,
    chat: Option<(String, Arc<ChatPool>)>,
    fim: Option<(String, Arc<ChatPool>)>,
    embed: Option<(String, Arc<Mutex<Embedder>>)>,
    #[cfg(feature = "image")]
    image: Option<images::ImageSlot>,
    api_key: Option<String>,
    created: u64,
    supersede: completions::Supersede,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn model_id(path: &std::path::Path, alias: &Option<String>) -> String {
    alias
        .clone()
        .unwrap_or_else(|| path.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned()))
}

impl AppState {
    /// Load models and create sessions (blocking; call before serving).
    pub fn load(config: &ServerConfig) -> offgrid::Result<Arc<Self>> {
        let chat = match &config.model {
            Some(path) => {
                let params = ModelParams { mmproj: config.mmproj.clone(), ..config.model_params.clone() };
                let model = load_model(path, &params)?;
                let pool = ChatPool::new(model, &config.session, config.parallel)?;
                Some((model_id(path, &config.alias), pool))
            }
            None => None,
        };
        let fim = match &config.fim_model {
            Some(path) => {
                let params = ModelParams { mmproj: None, ..config.model_params.clone() };
                let model = load_model(path, &params)?;
                if !model.info().infill {
                    return Err(offgrid::Error::InvalidArgument(format!(
                        "{} has no fill-in-the-middle tokens (--fim-model needs a code model)",
                        path.display()
                    )));
                }
                let pool = ChatPool::new(model, &config.session, config.parallel)?;
                Some((model_id(path, &config.fim_alias), pool))
            }
            None => None,
        };
        let embed = match &config.embedding_model {
            Some(path) => {
                let params = ModelParams { mmproj: None, ..config.model_params.clone() };
                let model = load_model(path, &params)?;
                let emb = Embedder::new(model, &EmbedParams::default())?;
                Some((model_id(path, &config.embedding_alias), Arc::new(Mutex::new(emb))))
            }
            None => None,
        };
        #[cfg(feature = "image")]
        let image = match &config.image {
            Some(params) => {
                let path = params.diffusion_model.clone().or_else(|| params.model.clone()).unwrap_or_default();
                log::info!("loading image model {}", path.display());
                let start = std::time::Instant::now();
                let model = offgrid::ImageModel::load(params)?;
                log::info!("loaded image model in {:.1}s", start.elapsed().as_secs_f64());
                Some((model_id(&path, &config.image_alias), Arc::new(Mutex::new(model))))
            }
            None => None,
        };
        #[cfg(feature = "image")]
        let has_image = image.is_some();
        #[cfg(not(feature = "image"))]
        let has_image = false;
        if chat.is_none() && fim.is_none() && embed.is_none() && !has_image {
            return Err(offgrid::Error::InvalidArgument(
                "no model configured (--model / --fim-model / --embedding-model)".into(),
            ));
        }
        Ok(Arc::new(Self {
            shutting_down: std::sync::atomic::AtomicBool::new(false),
            max_body_bytes: if config.max_body_bytes == 0 { DEFAULT_MAX_BODY_BYTES } else { config.max_body_bytes },
            chat,
            fim,
            embed,
            #[cfg(feature = "image")]
            image,
            api_key: config.api_key.clone().filter(|k| !k.is_empty()),
            created: now(),
            supersede: Default::default(),
        }))
    }

    /// Begin shutdown: cancel running generations and refuse queued/new requests.
    pub fn shutdown(&self) {
        self.shutting_down.store(true, std::sync::atomic::Ordering::SeqCst);
        for (_, pool) in self.chat.iter().chain(&self.fim) {
            pool.cancel_all();
        }
    }

    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The pool for text completion and infill: the model named in the request, else the
    /// FIM model, else the chat model.
    fn completion_pool(&self, requested: Option<&str>) -> Option<(String, Arc<ChatPool>)> {
        let named =
            |slot: &Option<(String, Arc<ChatPool>)>| slot.clone().filter(|(id, _)| Some(id.as_str()) == requested);
        named(&self.fim).or_else(|| named(&self.chat)).or_else(|| self.fim.clone()).or_else(|| self.chat.clone())
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/completions", post(completions::completions))
        .route("/embeddings", post(embeddings))
        .route("/models", get(list_models))
        .route("/models/{id}", get(get_model));
    #[cfg(feature = "image")]
    let v1 = v1.route("/images/generations", post(images::generations));
    let v1 = v1
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        // axum's default is 2 MB, too small for base64 media.
        .layer(axum::extract::DefaultBodyLimit::max(state.max_body_bytes));
    // llama.cpp serves /infill outside /v1; clients such as llama.vscode expect it there.
    let infill = Router::new()
        .route("/infill", post(completions::infill))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .layer(axum::extract::DefaultBodyLimit::max(state.max_body_bytes));
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .merge(infill)
        .nest("/v1", v1)
        .fallback(|req: Request| async move {
            log::debug!("unknown endpoint: {} {}", req.method(), req.uri());
            ApiError::not_found("unknown endpoint")
        })
        .layer(middleware::map_response(json_charset))
        .with_state(state)
}

/// Clients such as Windows PowerShell 5 decode JSON without an explicit charset as Latin-1.
async fn json_charset(mut res: Response) -> Response {
    let ct = axum::http::header::CONTENT_TYPE;
    if res.headers().get(&ct).is_some_and(|v| v == "application/json") {
        res.headers_mut().insert(ct, axum::http::HeaderValue::from_static("application/json; charset=utf-8"));
    }
    res
}

async fn auth(State(state): State<Arc<AppState>>, headers: HeaderMap, req: Request, next: Next) -> Response {
    if let Some(key) = &state.api_key {
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|given| given == key);
        if !ok {
            return ApiError::new(StatusCode::UNAUTHORIZED, "invalid_api_key", "invalid or missing API key")
                .into_response();
        }
    }
    next.run(req).await
}

fn model_entry(state: &AppState, id: &str) -> Value {
    json!({"id": id, "object": "model", "created": state.created, "owned_by": "offgrid"})
}

/// Ids of the loaded models: chat, FIM, embedding, image.
fn model_ids(state: &AppState) -> Vec<&str> {
    let ids = state
        .chat
        .iter()
        .chain(&state.fim)
        .map(|(id, _)| id.as_str())
        .chain(state.embed.iter().map(|(id, _)| id.as_str()));
    #[cfg(feature = "image")]
    let ids = ids.chain(state.image.iter().map(|(id, _)| id.as_str()));
    ids.collect()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = model_ids(&state).into_iter().map(|id| model_entry(&state, id)).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn get_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if model_ids(&state).contains(&id.as_str()) {
        Json(model_entry(&state, &id)).into_response()
    } else {
        ApiError::not_found("model not found").into_response()
    }
}

/// Cancels its request when dropped (the client went away). No-op once the request is done.
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
fn body_error(status: StatusCode, detail: String) -> ApiError {
    let kind = if status == StatusCode::PAYLOAD_TOO_LARGE { "request_too_large" } else { "invalid_request_error" };
    ApiError::new(status, kind, detail)
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<axum::body::Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let body = match body {
        Ok(b) => b,
        Err(e) => return body_error(e.status(), e.body_text()).into_response(),
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
        return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "server is shutting down")
            .into_response();
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
        Err(e) => return body_error(e.status(), e.body_text()).into_response(),
    };
    let Some((model_id, embedder)) = state.embed.clone() else {
        return ApiError::bad_request("no embedding model loaded (--embedding-model)").into_response();
    };
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => {
            items.iter().filter_map(|v| v.as_str().map(String::from)).collect()
        }
        Some(Value::Array(_)) => {
            return ApiError::bad_request(
                "input must be a string or an array of strings (token arrays are not supported)",
            )
            .into_response();
        }
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

fn load_model(path: &std::path::Path, params: &ModelParams) -> offgrid::Result<Arc<Model>> {
    log::info!("loading {}", path.display());
    let start = std::time::Instant::now();
    let model = Model::load(path, params)?;
    log::info!("loaded {} in {:.1}s", path.display(), start.elapsed().as_secs_f64());
    Ok(model)
}

/// Parse a `--gpu-layers` value: `all`, `none` or a count.
pub fn parse_gpu_layers(s: &str) -> Result<GpuLayers, String> {
    match s {
        "all" => Ok(GpuLayers::All),
        "none" | "0" => Ok(GpuLayers::None),
        n => n.parse().map(GpuLayers::Count).map_err(|_| format!("expected all, none or a number, got '{n}'")),
    }
}
