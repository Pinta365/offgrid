//! OpenAI wire-format responses and errors.

use aicore::{ChatResponse, FinishReason, ToolCall, Usage};
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde_json::{Value, json};

/// Per-request response metadata shared by all chunks.
pub struct Meta {
    id: String,
    created: u64,
    model: String,
}

impl Meta {
    pub fn new(model: String) -> Self {
        Self { id: format!("chatcmpl-{}", random_id()), created: super::now(), model }
    }

    fn base(&self, object: &str) -> serde_json::Map<String, Value> {
        let mut m = serde_json::Map::new();
        m.insert("id".into(), json!(self.id));
        m.insert("object".into(), json!(object));
        m.insert("created".into(), json!(self.created));
        m.insert("model".into(), json!(self.model));
        m
    }

    pub fn completion(&self, resp: &ChatResponse) -> Value {
        let mut message = json!({"role": "assistant", "content": resp.content});
        if let Some(r) = &resp.reasoning_content {
            message["reasoning_content"] = json!(r);
        }
        if !resp.tool_calls.is_empty() {
            message["tool_calls"] = json!(resp.tool_calls);
            if resp.content.is_empty() {
                message["content"] = Value::Null;
            }
        }
        let mut m = self.base("chat.completion");
        m.insert(
            "choices".into(),
            json!([{"index": 0, "message": message, "finish_reason": finish_reason(resp.finish_reason), "logprobs": null}]),
        );
        m.insert("usage".into(), usage(&resp.usage));
        Value::Object(m)
    }

    pub fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        let mut m = self.base("chat.completion.chunk");
        m.insert("choices".into(), json!([{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}]));
        Value::Object(m)
    }

    pub fn usage_chunk(&self, u: &Usage) -> Value {
        let mut m = self.base("chat.completion.chunk");
        m.insert("choices".into(), json!([]));
        m.insert("usage".into(), usage(u));
        Value::Object(m)
    }
}

pub fn finish_reason(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Stop | FinishReason::Cancelled => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

fn usage(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
        "timings": {"prompt_ms": u.prompt_ms, "generation_ms": u.generation_ms}
    })
}

/// A complete call sent as one streaming delta (OpenAI streams `tool_calls` by index).
pub fn tool_call_delta(index: usize, call: &ToolCall) -> Value {
    json!({
        "index": index,
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments}
    })
}

pub fn embeddings(model: &str, vectors: &[Vec<f32>], tokens: usize, base64: bool) -> Value {
    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let embedding = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            json!({"object": "embedding", "index": i, "embedding": embedding})
        })
        .collect();
    json!({"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})
}

fn random_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    format!("{:016x}", h.finish())
}

/// OpenAI-style error: `{"error":{"message","type","code"}}`.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into() }
    }
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", message)
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }
    /// Map core errors: caller mistakes → 400, the rest → 500.
    pub fn from_core(e: aicore::Error) -> Self {
        use aicore::Error::*;
        match e {
            InvalidArgument(_) | ContextOverflow { .. } | ChatTemplate(_) | Grammar | JsonSchema(_) | Tokenize => {
                Self::bad_request(e.to_string())
            }
            other => Self::internal(other.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"message": self.message, "type": self.kind, "code": null}});
        (self.status, Json(body)).into_response()
    }
}
