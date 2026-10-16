//! Text completion: `POST /v1/completions` (OpenAI legacy; `suffix` makes it
//! fill-in-the-middle) and `POST /infill` (llama.cpp's format, used by llama.vscode).

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use offgrid::{ChatResponse, ChatSession, FinishReason, InfillRequest, RequestCancel, Usage};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::pool::ChatPool;
use crate::{ApiError, AppState, CancelOnDrop, CancellingStream, body_error, openai};

type Body = Result<axum::body::Bytes, axum::extract::rejection::BytesRejection>;

/// What to run on the leased session.
enum Job {
    Infill(InfillRequest),
    /// A raw prompt (special tokens parsed), with sampling, stop and limit from `req`.
    Raw {
        prompt: String,
        req: InfillRequest,
    },
}

impl Job {
    fn run(&self, session: &mut ChatSession, on_piece: impl FnMut(&str) -> bool) -> offgrid::Result<ChatResponse> {
        match self {
            Job::Infill(req) => session.infill(req, on_piece),
            Job::Raw { prompt, req } => session.generate(prompt, req.max_tokens, &req.stop, &req.sampling, on_piece),
        }
    }
}

/// Editors send a completion request per keystroke; only the latest one matters. A request
/// supersedes earlier ones with the same key (llama.cpp's `id_slot`, which llama.vscode
/// sends): a waiting one is answered empty, a running one is cancelled.
#[derive(Default)]
pub struct Supersede {
    next: AtomicU64,
    /// Per key: the latest request's number, and its canceller once it runs.
    latest: Mutex<HashMap<i64, (u64, Option<RequestCancel>)>>,
}

impl Supersede {
    /// Register a new request for `key`, cancelling the running one. Returns its ticket.
    fn arrive(&self, key: i64) -> (i64, u64) {
        let n = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let previous = self.latest.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (n, None));
        if let Some((_, Some(cancel))) = previous {
            log::debug!("request {n} supersedes a running one (id_slot {key})");
            cancel.cancel();
        }
        (key, n)
    }

    /// About to run: `false` if a newer request arrived meanwhile.
    fn start(&self, (key, n): (i64, u64), cancel: &RequestCancel) -> bool {
        match self.latest.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&key) {
            Some((latest, slot)) if *latest == n => {
                *slot = Some(cancel.clone());
                true
            }
            _ => false,
        }
    }

    /// The request ended: forget its canceller, so a later request doesn't report cancelling
    /// it (cancelling a finished request would be a no-op anyway).
    fn finish(&self, (key, n): (i64, u64)) {
        if let Some((latest, slot)) = self.latest.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&key)
            && *latest == n
        {
            *slot = None;
        }
    }
}

/// Response wire format.
enum Format {
    /// llama.cpp's `/infill` and `/completion` objects.
    Llama { model: String },
    /// OpenAI `text_completion`.
    OpenAi { meta: openai::Meta, include_usage: bool },
}

impl Format {
    fn chunk(&self, text: &str) -> Value {
        match self {
            Format::Llama { .. } => json!({"content": text, "stop": false}),
            Format::OpenAi { meta, .. } => meta.text_chunk(text, None),
        }
    }

    /// Events that end a stream.
    fn finish(&self, resp: &ChatResponse) -> Vec<Value> {
        match self {
            Format::Llama { model } => vec![llama_result(model, resp, "")],
            Format::OpenAi { meta, include_usage } => {
                let mut events = vec![meta.text_chunk("", Some(openai::finish_reason(resp.finish_reason)))];
                if *include_usage {
                    events.push(meta.usage_chunk(&resp.usage));
                }
                events
            }
        }
    }

    fn body(&self, resp: &ChatResponse) -> Value {
        match self {
            Format::Llama { model } => llama_result(model, resp, &resp.content),
            Format::OpenAi { meta, .. } => meta.text_completion(resp),
        }
    }
}

/// llama.cpp's final result object (streamed with empty `content`).
fn llama_result(model: &str, resp: &ChatResponse, content: &str) -> Value {
    let u = &resp.usage;
    let prompt_n = u.prompt_tokens - u.cached_tokens;
    let per_second = |n: u32, ms: f64| if ms > 0.0 { f64::from(n) * 1000.0 / ms } else { 0.0 };
    let stop_type = match resp.finish_reason {
        FinishReason::Stop => "eos",
        FinishReason::Length => "limit",
        FinishReason::Cancelled | FinishReason::ToolCalls => "none",
    };
    json!({
        "content": content,
        "model": model,
        "stop": true,
        "stop_type": stop_type,
        "tokens_predicted": u.completion_tokens,
        "tokens_evaluated": u.prompt_tokens,
        "tokens_cached": u.prompt_tokens + u.completion_tokens,
        "timings": {
            "cache_n": u.cached_tokens,
            "prompt_n": prompt_n,
            "prompt_ms": u.prompt_ms,
            "prompt_per_second": per_second(prompt_n, u.prompt_ms),
            "predicted_n": u.completion_tokens,
            "predicted_ms": u.generation_ms,
            "predicted_per_second": per_second(u.completion_tokens, u.generation_ms),
        }
    })
}

fn parse_body(body: Body) -> Result<serde_json::Map<String, Value>, ApiError> {
    let body = body.map_err(|e| body_error(e.status(), e.body_text()))?;
    match serde_json::from_slice(&body) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => Err(ApiError::bad_request("the body must be a JSON object")),
        Err(e) => Err(ApiError::bad_request(format!("invalid JSON body: {e}"))),
    }
}

fn parse_request(body: serde_json::Map<String, Value>) -> Result<InfillRequest, ApiError> {
    serde_json::from_value(Value::Object(body)).map_err(|e| ApiError::bad_request(format!("invalid request: {e}")))
}

fn pool_for(state: &AppState, body: &serde_json::Map<String, Value>) -> Result<(String, Arc<ChatPool>), ApiError> {
    state
        .completion_pool(body.get("model").and_then(Value::as_str))
        .ok_or_else(|| ApiError::bad_request("no text model loaded (--model / --fim-model)"))
}

fn require_infill(id: &str, pool: &ChatPool) -> Result<(), ApiError> {
    if pool.supports_infill() {
        Ok(())
    } else {
        let msg = format!("model '{id}' has no fill-in-the-middle tokens; load a code model with --fim-model");
        Err(ApiError::bad_request(msg))
    }
}

pub async fn infill(State(state): State<Arc<AppState>>, body: Body) -> Response {
    let run = async {
        let body = parse_body(body)?;
        log_request("/infill", &body);
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        if body.get("input_extra").is_some_and(|v| !v.is_array() && !v.is_null()) {
            return Err(ApiError::bad_request("input_extra must be an array of {filename, text}"));
        }
        let (model, pool) = pool_for(&state, &body)?;
        require_infill(&model, &pool)?;
        let id_slot = body.get("id_slot").and_then(Value::as_i64);
        // llama.cpp's sampler list: we only read whether it asks for the infill sampler.
        let infill_sampler = body.get("samplers").and_then(Value::as_array).map(|a| a.iter().any(|s| s == "infill"));
        let mut req = parse_request(body)?;
        if let Some(on) = infill_sampler {
            req.infill_sampler = on;
        }
        let ticket = id_slot.map(|key| state.supersede.arrive(key));
        Ok(serve(state.clone(), pool, Job::Infill(req), stream, Format::Llama { model }, ticket).await)
    };
    run.await.unwrap_or_else(IntoResponse::into_response)
}

pub async fn completions(State(state): State<Arc<AppState>>, body: Body) -> Response {
    let run = async {
        let mut body = parse_body(body)?;
        log_request("/v1/completions", &body);
        let bad = |msg: &str| Err(ApiError::bad_request(msg));
        if body.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
            return bad("only n=1 is supported");
        }
        if body.get("best_of").and_then(Value::as_u64).is_some_and(|n| n != 1) {
            return bad("best_of is not supported");
        }
        if body.get("echo").and_then(Value::as_bool) == Some(true) {
            return bad("echo is not supported");
        }
        if body.get("logprobs").is_some_and(|v| !v.is_null() && v.as_u64() != Some(0)) {
            return bad("logprobs is not supported");
        }
        let prompt = match body.remove("prompt") {
            Some(Value::String(s)) => s,
            Some(Value::Array(mut items)) if items.len() == 1 && items[0].is_string() => {
                items.remove(0).as_str().unwrap_or_default().to_string()
            }
            Some(Value::Array(items)) if items.len() > 1 => return bad("only one prompt per request is supported"),
            None => return bad("missing prompt"),
            Some(_) => return bad("prompt must be a string (token arrays are not supported)"),
        };
        let suffix = match body.remove("suffix") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s),
            Some(_) => return bad("suffix must be a string"),
        };
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let include_usage =
            body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false);
        let (model, pool) = pool_for(&state, &body)?;
        let mut req = parse_request(body)?;
        let job = match suffix {
            Some(suffix) => {
                require_infill(&model, &pool)?;
                req.input_prefix = prompt;
                req.input_suffix = suffix;
                Job::Infill(req)
            }
            None if prompt.is_empty() => return bad("prompt must not be empty"),
            None => Job::Raw { prompt, req },
        };
        let format = Format::OpenAi { meta: openai::Meta::text(model), include_usage };
        Ok(serve(state.clone(), pool, job, stream, format, None).await)
    };
    run.await.unwrap_or_else(IntoResponse::into_response)
}

/// Debug log of a request's fields (long text as its length), to see what editor clients send.
fn log_request(path: &str, body: &serde_json::Map<String, Value>) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    let fields: Vec<String> = body
        .iter()
        .map(|(k, v)| match v {
            Value::String(s) if s.len() > 40 => format!("{k}=<{} chars>", s.len()),
            Value::Array(a) if k == "input_extra" => {
                let chars: usize = a.iter().filter_map(|c| c["text"].as_str()).map(str::len).sum();
                format!("{k}=<{} chunks, {chars} chars>", a.len())
            }
            other => format!("{k}={other}"),
        })
        .collect();
    log::debug!("{path} {}", fields.join(" "));
}

/// Debug log of a finished completion: why it stopped, cache use and timings.
fn log_result(resp: &ChatResponse, elapsed: std::time::Duration) {
    let u = &resp.usage;
    let preview: String = resp.content.chars().take(60).collect();
    log::debug!(
        "  -> {:?} prompt {} (cached {}) in {:.0} ms, {} generated in {:.0} ms, total {:.0} ms: {preview:?}",
        resp.finish_reason,
        u.prompt_tokens,
        u.cached_tokens,
        u.prompt_ms,
        u.completion_tokens,
        u.generation_ms,
        elapsed.as_secs_f64() * 1000.0
    );
}

/// An empty, cancelled result for a request a newer one replaced while it waited.
fn superseded(stream: bool, format: &Format) -> Response {
    let resp = ChatResponse {
        content: String::new(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        finish_reason: FinishReason::Cancelled,
        usage: Usage::default(),
    };
    if !stream {
        return Json(format.body(&resp)).into_response();
    }
    let mut events: Vec<Result<Event, Infallible>> =
        format.finish(&resp).into_iter().map(|v| Ok(Event::default().data(v.to_string()))).collect();
    if matches!(format, Format::OpenAi { .. }) {
        events.push(Ok(Event::default().data("[DONE]")));
    }
    Sse::new(futures_util::stream::iter(events)).into_response()
}

/// Run `job` on a session from `pool`: one JSON body, or SSE events (OpenAI streams end
/// with `[DONE]`, llama.cpp's with the final result).
async fn serve(
    state: Arc<AppState>,
    pool: Arc<ChatPool>,
    job: Job,
    stream: bool,
    format: Format,
    ticket: Option<(i64, u64)>,
) -> Response {
    let t0 = std::time::Instant::now();
    let lease = pool.acquire().await;
    // Requests that waited in the queue during shutdown must not start a generation.
    if state.is_shutting_down() {
        return ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "server is shutting down")
            .into_response();
    }
    let next = lease.next_request_cancel();
    if ticket.is_some_and(|t| !state.supersede.start(t, &next)) {
        // Answer without starting a request: a canceller for a request that never starts
        // would cancel the session's next one.
        log::debug!("skipping a superseded request");
        drop(lease);
        return superseded(stream, &format);
    }
    let cancel = CancelOnDrop(next);
    let finish = move || {
        if let Some(t) = ticket {
            state.supersede.finish(t);
        }
    };

    if stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
        tokio::task::spawn_blocking(move || {
            let send = |v: Value| tx.blocking_send(Ok(Event::default().data(v.to_string()))).is_ok();
            // A failed send means the client disconnected: cancel generation.
            let result = job.run(&mut lease.session(), |piece| send(format.chunk(piece)));
            finish();
            match result {
                Ok(resp) => {
                    log_result(&resp, t0.elapsed());
                    format.finish(&resp).into_iter().for_each(|v| {
                        send(v);
                    });
                }
                Err(e) => {
                    send(json!({"error": {"message": e.to_string(), "type": "server_error"}}));
                }
            }
            if matches!(format, Format::OpenAi { .. }) {
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
        });
        let body = CancellingStream { inner: ReceiverStream::new(rx), _cancel: cancel };
        Sse::new(body).keep_alive(KeepAlive::default()).into_response()
    } else {
        let _cancel = cancel; // dropped with this future if the client disconnects
        let joined = tokio::task::spawn_blocking(move || {
            let result = job.run(&mut lease.session(), |_| true);
            finish();
            result.map(|r| {
                log_result(&r, t0.elapsed());
                format.body(&r)
            })
        })
        .await;
        match joined {
            Ok(Ok(body)) => Json(body).into_response(),
            Ok(Err(e)) => ApiError::from_core(e).into_response(),
            Err(e) => ApiError::internal(format!("worker failed: {e}")).into_response(),
        }
    }
}
