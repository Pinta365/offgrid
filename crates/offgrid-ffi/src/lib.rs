//! C ABI for offgrid.
//!
//! - Handles are opaque; free each with its `_free` function (NULL is ignored).
//! - Params, requests and results are UTF-8 JSON. NULL/empty JSON means defaults.
//! - Returned `char*` strings are caller-owned: free with `offgrid_string_free`.
//! - Fallible functions take a trailing nullable `char **error_out`; on failure they return
//!   NULL / negative and store a message there (no thread-local error state).
//! - A handle must not be used from two threads at once, except `offgrid_chat_cancel`.

#![allow(clippy::missing_safety_doc)]

mod logging;
mod util;

use std::ffi::{CString, c_char, c_void};
use std::sync::{Arc, Mutex};

use offgrid::{
    CancelToken, ChatRequest, ChatSession, EmbedParams, Embedder, Model, ModelParams, SessionParams, StreamEvent,
};

pub use logging::OffgridLogCallback;
use util::{FfiResult, guard, handle, json_arg, json_out, str_arg};

/// Bumped on any incompatible change to the C ABI or the JSON shapes.
pub const OFFGRID_ABI_VERSION: u32 = 1;

/// Opaque model handle.
pub struct OffgridModel(Arc<Model>);

/// Opaque chat session handle.
pub struct OffgridChat {
    session: Mutex<ChatSession>,
    cancel: CancelToken,
}

/// Opaque embedder handle.
pub struct OffgridEmbedder(Mutex<Embedder>);

/// Streaming callback: receives `len` bytes of UTF-8 text (also NUL-terminated).
/// Return `false` to cancel generation.
pub type OffgridTokenCallback =
    Option<unsafe extern "C" fn(piece: *const c_char, len: usize, user_data: *mut c_void) -> bool>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[unsafe(no_mangle)]
pub extern "C" fn offgrid_abi_version() -> u32 {
    OFFGRID_ABI_VERSION
}

/// Library + bundled llama.cpp version as JSON: `{"offgrid":"0.1.0","llama_cpp":"..."}`.
#[unsafe(no_mangle)]
pub extern "C" fn offgrid_version_json() -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || {
        json_out(&serde_json::json!({
            "offgrid": env!("CARGO_PKG_VERSION"),
            "abi": OFFGRID_ABI_VERSION,
            "llama_cpp": offgrid::version(),
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Route logs to `callback` (NULL = stderr) at `level` (0=off, 1=error .. 5=trace).
/// Without this call the level comes from env `OFFGRID_LOG`, default off.
#[unsafe(no_mangle)]
pub extern "C" fn offgrid_set_log_callback(callback: OffgridLogCallback, user_data: *mut c_void, level: i32) {
    logging::set_sink(callback, user_data, level);
}

/// Compute devices as a JSON array.
#[unsafe(no_mangle)]
pub extern "C" fn offgrid_devices_json() -> *mut c_char {
    logging::install();
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || json_out(&offgrid::devices()))
}

/// Load a GGUF model. `params_json` (nullable):
/// `{"gpu_layers":"all"|"none"|{"count":N},"devices":["Vulkan0"],"mmproj":"/path/mmproj.gguf"}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_model_load(
    path: *const c_char,
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut OffgridModel {
    logging::install();
    guard(error_out, std::ptr::null_mut(), || {
        let path = str_arg(path, "path")?;
        let params: ModelParams = json_arg(params_json, "params")?;
        let model = Model::load(path, &params).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(OffgridModel(model))))
    })
}

/// Model metadata as JSON (name, size, context, chat template, `vision` / `audio` support).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_model_info_json(model: *const OffgridModel) -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || json_out(&handle(model, "model")?.0.info()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_model_free(model: *mut OffgridModel) {
    if !model.is_null() {
        guard(std::ptr::null_mut(), (), || {
            drop(unsafe { Box::from_raw(model) });
            Ok(())
        });
    }
}

/// Create a chat session. `params_json` (nullable): `{"n_ctx":8192,"n_batch":512,"n_threads":8}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_new(
    model: *const OffgridModel,
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut OffgridChat {
    guard(error_out, std::ptr::null_mut(), || {
        let model = handle(model, "model")?.0.clone();
        let params: SessionParams = json_arg(params_json, "params")?;
        let session = ChatSession::new(model, &params).map_err(|e| e.to_string())?;
        let cancel = session.cancel_token();
        Ok(Box::into_raw(Box::new(OffgridChat { session: Mutex::new(session), cancel })))
    })
}

/// Run a chat completion. `request_json` is an OpenAI chat request (messages, tools,
/// response_format, image/audio parts as data). `callback` (nullable) streams reply text.
/// Blocks until done; returns the response JSON, or NULL on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_complete(
    chat: *const OffgridChat,
    request_json: *const c_char,
    callback: OffgridTokenCallback,
    user_data: *mut c_void,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    guard(error_out, std::ptr::null_mut(), || {
        let chat = handle(chat, "chat")?;
        let req: ChatRequest = json_arg(request_json, "request")?;
        let mut session = lock(&chat.session);
        let resp =
            session.complete(&req, |piece| stream_piece(callback, user_data, piece)).map_err(|e| e.to_string())?;
        json_out(&resp)
    })
}

/// Event kinds for [`offgrid_chat_complete_events`].
pub const OFFGRID_EVENT_CONTENT: i32 = 0;
pub const OFFGRID_EVENT_REASONING: i32 = 1;
/// `text` is the call as JSON: `{"id","type":"function","function":{"name","arguments"}}`.
pub const OFFGRID_EVENT_TOOL_CALL: i32 = 2;

/// Streaming event callback: `kind` is an `OFFGRID_EVENT_*` value; `text` holds `len`
/// bytes of UTF-8 (also NUL-terminated). Return `false` to cancel generation.
pub type OffgridEventCallback =
    Option<unsafe extern "C" fn(kind: i32, text: *const c_char, len: usize, user_data: *mut c_void) -> bool>;

/// Like [`offgrid_chat_complete`] but also streams reasoning and tool calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_complete_events(
    chat: *const OffgridChat,
    request_json: *const c_char,
    callback: OffgridEventCallback,
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
                    StreamEvent::Content(t) => (OFFGRID_EVENT_CONTENT, t.to_string()),
                    StreamEvent::Reasoning(t) => (OFFGRID_EVENT_REASONING, t.to_string()),
                    StreamEvent::ToolCall(c) => (OFFGRID_EVENT_TOOL_CALL, serde_json::to_string(c).unwrap_or_default()),
                };
                let Ok(c) = CString::new(text.as_str()) else { return true };
                unsafe { cb(kind, c.as_ptr(), text.len(), user_data) }
            })
            .map_err(|e| e.to_string())?;
        json_out(&resp)
    })
}

/// Raw text completion without a chat template. `request_json`: `{"prompt":"...",...}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_generate(
    chat: *const OffgridChat,
    request_json: *const c_char,
    callback: OffgridTokenCallback,
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
        sampling: offgrid::SamplingParams,
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

fn stream_piece(callback: OffgridTokenCallback, user_data: *mut c_void, piece: &str) -> bool {
    let Some(cb) = callback else { return true };
    let Ok(c) = CString::new(piece) else { return true };
    unsafe { cb(c.as_ptr(), piece.len(), user_data) }
}

/// Cancel the running request. Safe from any thread; no-op when idle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_cancel(chat: *const OffgridChat) {
    if let Some(chat) = unsafe { chat.as_ref() } {
        chat.cancel.cancel();
    }
}

/// Id of the next request (for [`offgrid_chat_cancel_request`]). Only meaningful while no
/// other thread can start a request on the session first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_next_request_id(chat: *const OffgridChat) -> u64 {
    unsafe { chat.as_ref() }.map_or(0, |c| c.cancel.next_request().id())
}

/// Cancel one request by id, even before it starts. Safe from any thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_cancel_request(chat: *const OffgridChat, request_id: u64) {
    if let Some(chat) = unsafe { chat.as_ref() } {
        chat.cancel.request(request_id).cancel();
    }
}

/// Clear the session's KV cache.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_reset(chat: *const OffgridChat) {
    guard(std::ptr::null_mut(), (), || {
        lock(&handle(chat, "chat")?.session).reset();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_chat_free(chat: *mut OffgridChat) {
    if !chat.is_null() {
        guard(std::ptr::null_mut(), (), || {
            drop(unsafe { Box::from_raw(chat) });
            Ok(())
        });
    }
}

/// Create an embedder. `params_json` (nullable): `{"n_ctx":0,"max_batch_inputs":16,"normalize":true}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_embedder_new(
    model: *const OffgridModel,
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut OffgridEmbedder {
    guard(error_out, std::ptr::null_mut(), || {
        let model = handle(model, "model")?.0.clone();
        let params: EmbedParams = json_arg(params_json, "params")?;
        let emb = Embedder::new(model, &params).map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(OffgridEmbedder(Mutex::new(emb)))))
    })
}

/// Embedding vector length, or -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_embedder_dim(emb: *const OffgridEmbedder) -> i64 {
    guard(std::ptr::null_mut(), -1, || Ok(lock(&handle(emb, "embedder")?.0).dim() as i64))
}

/// Embed `n_inputs` UTF-8 strings into `out`, which must hold `n_inputs * dim` floats (row-major).
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_embed(
    emb: *const OffgridEmbedder,
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
        let texts =
            (0..n_inputs).map(|i| str_arg(unsafe { *inputs.add(i) }, "input")).collect::<FfiResult<Vec<&str>>>()?;
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
pub unsafe extern "C" fn offgrid_embed_json(
    emb: *const OffgridEmbedder,
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
pub unsafe extern "C" fn offgrid_embedder_free(emb: *mut OffgridEmbedder) {
    if !emb.is_null() {
        guard(std::ptr::null_mut(), (), || {
            drop(unsafe { Box::from_raw(emb) });
            Ok(())
        });
    }
}

/// Opaque image model handle.
pub struct OffgridImageModel {
    #[cfg(feature = "image")]
    inner: Mutex<offgrid::ImageModel>,
}

/// Image generation phase passed to the progress callback: weights loading.
pub const OFFGRID_IMAGE_LOADING: i32 = 0;
/// Image generation phase: a sampling step (`done` of `total` steps).
pub const OFFGRID_IMAGE_SAMPLING: i32 = 1;
/// Image generation phase: VAE decoding.
pub const OFFGRID_IMAGE_DECODING: i32 = 2;

/// Image progress callback: `phase` is an `OFFGRID_IMAGE_*` value. Return `false` to cancel.
pub type OffgridImageProgressCallback =
    Option<unsafe extern "C" fn(phase: i32, done: u32, total: u32, user_data: *mut c_void) -> bool>;

/// Load an image model (needs a build with image generation). `params_json`:
/// `{"diffusion_model","text_encoder","vae"}` or `{"model"}` for single-file checkpoints, plus
/// optional `"gpu"`, `"max_vram_gib"`, `"flash_attn"`, `"n_threads"`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_image_model_load(
    params_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut OffgridImageModel {
    guard(error_out, std::ptr::null_mut(), || {
        #[cfg(feature = "image")]
        {
            let params: offgrid::ImageModelParams = json_arg(params_json, "params")?;
            let model = offgrid::ImageModel::load(&params).map_err(|e| e.to_string())?;
            Ok(Box::into_raw(Box::new(OffgridImageModel { inner: Mutex::new(model) })))
        }
        #[cfg(not(feature = "image"))]
        {
            let _ = params_json;
            Err("this build of offgrid has no image generation".into())
        }
    })
}

/// Generate an image; returns PNG bytes (free with [`offgrid_bytes_free`]) and their length in
/// `*len_out`, or NULL on error. `request_json`: `{"prompt","negative_prompt","width","height",
/// "steps","cfg_scale","seed","sampler","scheduler"}`, plus `"init_image"` (base64 PNG) and
/// `"strength"` for image-to-image. A callback returning `false` cancels; the error is then
/// "cancelled". One generation runs at a time per process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_image_generate(
    model: *const OffgridImageModel,
    request_json: *const c_char,
    callback: OffgridImageProgressCallback,
    user_data: *mut c_void,
    len_out: *mut usize,
    error_out: *mut *mut c_char,
) -> *mut u8 {
    guard(error_out, std::ptr::null_mut(), || {
        let model = handle(model, "model")?;
        if len_out.is_null() {
            return Err("len_out is NULL".into());
        }
        #[cfg(feature = "image")]
        {
            let req: offgrid::ImageRequest = json_arg(request_json, "request")?;
            let mut model = model.inner.lock().unwrap_or_else(|e| e.into_inner());
            let image = model
                .generate(&req, |p| {
                    let Some(cb) = callback else { return true };
                    let (phase, done, total) = match p {
                        offgrid::ImageProgress::Loading { done, total } => (OFFGRID_IMAGE_LOADING, done, total),
                        offgrid::ImageProgress::Sampling { step, steps } => (OFFGRID_IMAGE_SAMPLING, step, steps),
                        offgrid::ImageProgress::Decoding { done, total } => (OFFGRID_IMAGE_DECODING, done, total),
                    };
                    unsafe { cb(phase, done, total, user_data) }
                })
                .map_err(|e| e.to_string())?;
            let png = image.to_png().into_boxed_slice();
            unsafe { *len_out = png.len() };
            Ok(Box::into_raw(png) as *mut u8)
        }
        #[cfg(not(feature = "image"))]
        {
            let _ = (model, request_json, callback, user_data);
            Err("this build of offgrid has no image generation".into())
        }
    })
}

/// Free bytes returned by offgrid (e.g. [`offgrid_image_generate`]); `len` is the returned length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_bytes_free(bytes: *mut u8, len: usize) {
    if !bytes.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(bytes, len)) });
    }
}

/// Free an image model and its memory (NULL is ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn offgrid_image_model_free(model: *mut OffgridImageModel) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}
