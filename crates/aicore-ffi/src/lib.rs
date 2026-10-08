//! C ABI for aicore.
//!
//! Conventions:
//! - Handles are opaque pointers; free each with its `_free` function (NULL is ignored).
//! - Parameters, requests and results are UTF-8 JSON strings. NULL/empty JSON means defaults.
//! - Returned `char*` strings are owned by the caller: release with `aicore_string_free`.
//! - Fallible functions take a trailing `char **error_out` (nullable). On failure they return
//!   NULL / a negative value and store a caller-owned message there; on success it is set to NULL.
//!   (No thread-local state, so calls may hop threads: Deno nonblocking, Go, thread pools.)
//! - A chat/embedder keeps its model alive; handles may be freed in any order.
//! - One handle must not be used by two threads at once, except `aicore_chat_cancel`,
//!   which is safe to call from any thread at any time.

#![allow(clippy::missing_safety_doc)]

mod logging;
mod util;

use std::ffi::{CString, c_char, c_void};
use std::sync::{Arc, Mutex};

use aicore::{
    CancelToken, ChatRequest, ChatSession, EmbedParams, Embedder, Model, ModelParams,
    SessionParams, StreamEvent,
};

pub use logging::AicoreLogCallback;
use util::{FfiResult, guard, handle, json_arg, json_out, str_arg};

/// Bumped on any incompatible change to the C ABI or the JSON shapes.
pub const AICORE_ABI_VERSION: u32 = 1;

/// Opaque model handle.
pub struct AicoreModel(Arc<Model>);

/// Opaque chat session handle.
pub struct AicoreChat {
    session: Mutex<ChatSession>,
    cancel: CancelToken,
}

/// Opaque embedder handle.
pub struct AicoreEmbedder(Mutex<Embedder>);

/// Streaming callback: receives `len` bytes of UTF-8 text (also NUL-terminated).
/// Return `false` to cancel generation.
pub type AicoreTokenCallback =
    Option<unsafe extern "C" fn(piece: *const c_char, len: usize, user_data: *mut c_void) -> bool>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------- library

#[unsafe(no_mangle)]
pub extern "C" fn aicore_abi_version() -> u32 {
    AICORE_ABI_VERSION
}

/// Library + bundled llama.cpp version as JSON: `{"aicore":"0.1.0","llama_cpp":"..."}`.
#[unsafe(no_mangle)]
pub extern "C" fn aicore_version_json() -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || {
        json_out(&serde_json::json!({
            "aicore": env!("CARGO_PKG_VERSION"),
            "abi": AICORE_ABI_VERSION,
            "llama_cpp": aicore::version(),
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Route logs to `callback` (NULL = stderr) at `level` (0=off, 1=error .. 5=trace).
/// Without this call the level comes from env `AICORE_LOG`, default off.
#[unsafe(no_mangle)]
pub extern "C" fn aicore_set_log_callback(callback: AicoreLogCallback, user_data: *mut c_void, level: i32) {
    logging::set_sink(callback, user_data, level);
}

/// Compute devices as a JSON array.
#[unsafe(no_mangle)]
pub extern "C" fn aicore_devices_json() -> *mut c_char {
    logging::install();
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || json_out(&aicore::devices()))
}

// ---------------------------------------------------------------- model

/// Load a GGUF model. `params_json` (nullable): `{"gpu_layers":"all"|"none"|{"count":N},"devices":["Vulkan0"],
/// "mmproj":"/path/mmproj.gguf"}` (`mmproj` = multimodal projector: enables image and/or audio
/// input, see `"vision"` / `"audio"` in [`aicore_model_info_json`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_model_load(
    path: *const c_char,
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut AicoreModel {
    logging::install();
    guard(error_out, std::ptr::null_mut(), || {
        let path = str_arg(path, "path")?;
        let params: ModelParams = json_arg(params_json, "params")?;
        let model = Model::load(path, &params).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(AicoreModel(model))))
    })
}

/// Model metadata as JSON (name, size, context, chat template, `vision` / `audio` support).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_model_info_json(model: *const AicoreModel) -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || json_out(&handle(model, "model")?.0.info()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_model_free(model: *mut AicoreModel) {
    if !model.is_null() {
        guard(std::ptr::null_mut(), (), || {
            drop(unsafe { Box::from_raw(model) });
            Ok(())
        });
    }
}

// ---------------------------------------------------------------- chat

/// Create a chat session. `params_json` (nullable): `{"n_ctx":8192,"n_batch":512,"n_threads":8}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_new(
    model: *const AicoreModel,
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut AicoreChat {
    guard(error_out, std::ptr::null_mut(), || {
        let model = handle(model, "model")?.0.clone();
        let params: SessionParams = json_arg(params_json, "params")?;
        let session = ChatSession::new(model, &params).map_err(|e| e.to_string())?;
        let cancel = session.cancel_token();
        Ok(Box::into_raw(Box::new(AicoreChat { session: Mutex::new(session), cancel })))
    })
}

/// Run a chat completion. `request_json` is
/// `{"messages":[{"role":"user","content":"..."}],"max_tokens":256,"temperature":0.7,"stop":[],...}`.
/// Tools (OpenAI format): `"tools":[{"type":"function","function":{"name","description","parameters"}}]`,
/// `"tool_choice":"auto"|"none"|"required"|{"type":"function","function":{"name":"..."}}`;
/// the response then has `"tool_calls":[{"id","type":"function","function":{"name","arguments"}}]`
/// and `"finish_reason":"tool_calls"`. Send results back as `{"role":"tool","tool_call_id":...,
/// "content":"..."}` after the assistant message (with its `tool_calls`). The callback streams
/// reply text only (no reasoning / tool-call markup); `reasoning_content` is in the response.
/// Images / audio: message `content` as OpenAI parts, `[{"type":"text","text":"..."},
/// {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
/// {"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"|"mp3"|"flac"}}]`
/// (images as data: URLs only; needs a model loaded with a matching `mmproj`).
/// Structured output: `"response_format":{"type":"json_object"}` or
/// `{"type":"json_schema","json_schema":{"schema":{...}}}`. Template: `"chat_template"`
/// (Jinja text or built-in name) and `"chat_template_kwargs":{...}`.
/// `callback` (nullable) streams text pieces. Blocks until done; returns response JSON
/// `{"content":"...","finish_reason":"stop"|"length"|"cancelled","usage":{...}}` or NULL on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_complete(
    chat: *const AicoreChat,
    request_json: *const c_char,
    callback: AicoreTokenCallback,
    user_data: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    guard(error_out, std::ptr::null_mut(), || {
        let chat = handle(chat, "chat")?;
        let req: ChatRequest = json_arg(request_json, "request")?;
        let mut session = lock(&chat.session);
        let resp = session
            .complete(&req, |piece| stream_piece(callback, user_data, piece))
            .map_err(|e| e.to_string())?;
        json_out(&resp)
    })
}

/// Event kinds for [`aicore_chat_complete_events`].
pub const AICORE_EVENT_CONTENT: i32 = 0;
pub const AICORE_EVENT_REASONING: i32 = 1;
/// `text` is the call as JSON: `{"id","type":"function","function":{"name","arguments"}}`.
pub const AICORE_EVENT_TOOL_CALL: i32 = 2;

/// Streaming event callback: `kind` is an `AICORE_EVENT_*` value; `text` holds `len`
/// bytes of UTF-8 (also NUL-terminated). Return `false` to cancel generation.
pub type AicoreEventCallback = Option<
    unsafe extern "C" fn(kind: i32, text: *const c_char, len: usize, user_data: *mut c_void) -> bool,
>;

/// Like [`aicore_chat_complete`] but streams reasoning and tool calls too:
/// reply text (`AICORE_EVENT_CONTENT`) and reasoning (`AICORE_EVENT_REASONING`) as they
/// are generated, then one `AICORE_EVENT_TOOL_CALL` per parsed call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_complete_events(
    chat: *const AicoreChat,
    request_json: *const c_char,
    callback: AicoreEventCallback,
    user_data: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    guard(error_out, std::ptr::null_mut(), || {
        let chat = handle(chat, "chat")?;
        let req: ChatRequest = json_arg(request_json, "request")?;
        let mut session = lock(&chat.session);
        let resp = session
            .complete_stream(&req, |event| {
                let Some(cb) = callback else { return true };
                let (kind, text) = match event {
                    StreamEvent::Content(t) => (AICORE_EVENT_CONTENT, t.to_string()),
                    StreamEvent::Reasoning(t) => (AICORE_EVENT_REASONING, t.to_string()),
                    StreamEvent::ToolCall(c) => (AICORE_EVENT_TOOL_CALL, serde_json::to_string(c).unwrap_or_default()),
                };
                let Ok(c) = CString::new(text.as_str()) else { return true };
                unsafe { cb(kind, c.as_ptr(), text.len(), user_data) }
            })
            .map_err(|e| e.to_string())?;
        json_out(&resp)
    })
}

/// Raw text completion without chat template. `request_json`:
/// `{"prompt":"...","max_tokens":N,"stop":[...],"temperature":...}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_generate(
    chat: *const AicoreChat,
    request_json: *const c_char,
    callback: AicoreTokenCallback,
    user_data: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    #[derive(serde::Deserialize, Default)]
    #[serde(default)]
    struct GenerateRequest {
        prompt: String,
        max_tokens: Option<u32>,
        stop: Vec<String>,
        #[serde(flatten)]
        sampling: aicore::SamplingParams,
    }
    guard(error_out, std::ptr::null_mut(), || {
        let chat = handle(chat, "chat")?;
        let req: GenerateRequest = json_arg(request_json, "request")?;
        let mut session = lock(&chat.session);
        let resp = session
            .generate(&req.prompt, req.max_tokens, &req.stop, &req.sampling, |piece| {
                stream_piece(callback, user_data, piece)
            })
            .map_err(|e| e.to_string())?;
        json_out(&resp)
    })
}

fn stream_piece(callback: AicoreTokenCallback, user_data: *mut c_void, piece: &str) -> bool {
    let Some(cb) = callback else { return true };
    let Ok(c) = CString::new(piece) else { return true };
    unsafe { cb(c.as_ptr(), piece.len(), user_data) }
}

/// Cancel the in-flight request on this session. Safe from any thread, any time.
/// A no-op when no request is running: it never affects a later request.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_cancel(chat: *const AicoreChat) {
    if let Some(chat) = unsafe { chat.as_ref() } {
        chat.cancel.cancel();
    }
}

/// Clear the session's KV cache.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_reset(chat: *const AicoreChat) {
    guard(std::ptr::null_mut(), (), || {
        lock(&handle(chat, "chat")?.session).reset();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_chat_free(chat: *mut AicoreChat) {
    if !chat.is_null() {
        guard(std::ptr::null_mut(), (), || {
            drop(unsafe { Box::from_raw(chat) });
            Ok(())
        });
    }
}

// ---------------------------------------------------------------- embeddings

/// Create an embedder. `params_json` (nullable): `{"n_ctx":0,"max_batch_inputs":16,"normalize":true}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_embedder_new(
    model: *const AicoreModel,
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut AicoreEmbedder {
    guard(error_out, std::ptr::null_mut(), || {
        let model = handle(model, "model")?.0.clone();
        let params: EmbedParams = json_arg(params_json, "params")?;
        let emb = Embedder::new(model, &params).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(AicoreEmbedder(Mutex::new(emb)))))
    })
}

/// Embedding vector length, or -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_embedder_dim(emb: *const AicoreEmbedder) -> i64 {
    guard(std::ptr::null_mut(), -1, || Ok(lock(&handle(emb, "embedder")?.0).dim() as i64))
}

/// Embed `n_inputs` UTF-8 strings into `out`, which must hold `n_inputs * dim` floats (row-major).
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_embed(
    emb: *const AicoreEmbedder,
    inputs: *const *const c_char,
    n_inputs: usize,
    out: *mut f32,
    out_len: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    guard(error_out, -1, || {
        let emb = handle(emb, "embedder")?;
        if n_inputs == 0 {
            return Ok(0);
        }
        if inputs.is_null() || out.is_null() {
            return Err("inputs/out is NULL".into());
        }
        let texts = (0..n_inputs)
            .map(|i| str_arg(unsafe { *inputs.add(i) }, "input"))
            .collect::<FfiResult<Vec<&str>>>()?;
        let mut emb = lock(&emb.0);
        let dim = emb.dim();
        if out_len < n_inputs * dim {
            return Err(format!("out_len {out_len} < n_inputs * dim ({})", n_inputs * dim));
        }
        let vectors = emb.embed(&texts).map_err(|e| e.to_string())?;
        let out = unsafe { std::slice::from_raw_parts_mut(out, n_inputs * dim) };
        for (row, v) in out.chunks_exact_mut(dim).zip(&vectors) {
            row.copy_from_slice(v);
        }
        Ok(0)
    })
}

/// JSON convenience: `inputs_json` is `["text", ...]`; returns `[[f32...], ...]` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_embed_json(
    emb: *const AicoreEmbedder,
    inputs_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    guard(error_out, std::ptr::null_mut(), || {
        let emb = handle(emb, "embedder")?;
        let inputs: Vec<String> = json_arg(inputs_json, "inputs")?;
        let vectors = lock(&emb.0).embed(&inputs).map_err(|e| e.to_string())?;
        json_out(&vectors)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aicore_embedder_free(emb: *mut AicoreEmbedder) {
    if !emb.is_null() {
        guard(std::ptr::null_mut(), (), || {
            drop(unsafe { Box::from_raw(emb) });
            Ok(())
        });
    }
}
