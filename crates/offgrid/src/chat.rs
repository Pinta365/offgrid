use std::borrow::Cow;
use std::ffi::{CString, c_char, c_void};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use offgrid_sys as sys;
use serde::{Deserialize, Serialize};

use crate::batch::Batch;
use crate::media::{Chunks, MediaMemo, Segment, media_marker};
use crate::sampling::Sampler;
use crate::template::{ChatTemplate, RenderArgs, is_jinja};
use crate::text::{PieceKind, Push, StreamFilter, TextStream};
use crate::tool_parse::{self, ToolFormat};
use crate::tools::{GrammarSpec, POLYFILL_STOP, Tool, ToolCall, ToolChoice, polyfill_messages, tool_grammar};
use crate::{Error, Media, MediaKind, Model, Result, SamplingParams};

const DEFAULT_MAX_CTX: u32 = 8192;

/// Settings fixed for the lifetime of a session's inference context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionParams {
    /// Context size in tokens. 0 = the model's training context, capped at 8192.
    pub n_ctx: u32,
    /// Max tokens submitted per decode call during prompt processing.
    pub n_batch: u32,
    /// CPU threads. `None` = half the available hardware threads.
    pub n_threads: Option<i32>,
    /// Reuse cached chunks of at least this many tokens that moved to an earlier position
    /// (llama.cpp's `n_cache_reuse`), e.g. when an editor's code window slides down a file:
    /// their KV entries are shifted instead of evaluated again. 0 = off. Text-only prompts on
    /// models whose cache can shift.
    pub cache_reuse: u32,
}

impl Default for SessionParams {
    fn default() -> Self {
        Self { n_ctx: 0, n_batch: 512, n_threads: None, cache_reuse: 0 }
    }
}

/// An OpenAI-style chat message. `content` holds one [`media_marker`] per item in `media`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "WireMessage", into = "WireMessage")]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool` (templates may accept others).
    pub role: String,
    /// Message text (with a media marker where each image / audio clip goes).
    pub content: String,
    /// Images and audio referenced by the markers in `content`, in order.
    pub media: Vec<Media>,
    /// Model reasoning (`<think>` content) of an earlier assistant turn.
    pub reasoning_content: Option<String>,
    /// Tool calls made by an assistant turn.
    pub tool_calls: Vec<ToolCall>,
    /// For `role: "tool"`: the id of the call this message answers.
    pub tool_call_id: Option<String>,
    /// For `role: "tool"`: the function name (some templates use it).
    pub name: Option<String>,
}

/// Serde shape of [`ChatMessage`] (OpenAI wire format).
#[derive(Serialize, Deserialize)]
struct WireMessage {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl TryFrom<WireMessage> for ChatMessage {
    type Error = String;

    fn try_from(w: WireMessage) -> std::result::Result<Self, String> {
        let (content, media) = match w.content {
            serde_json::Value::Null => (String::new(), Vec::new()),
            serde_json::Value::String(s) => (s, Vec::new()),
            serde_json::Value::Array(parts) => {
                let (mut texts, mut media) = (Vec::new(), Vec::new());
                for part in parts {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("text") => texts.push(part.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string()),
                        Some("image_url") => {
                            let url = part
                                .pointer("/image_url/url")
                                .or_else(|| part.get("image_url"))
                                .and_then(|u| u.as_str())
                                .ok_or("image_url part without url")?;
                            media.push(Media::from_data_url(url).map_err(|e| e.to_string())?);
                            texts.push(media_marker().to_string());
                        }
                        Some("input_audio") => {
                            let data = part
                                .pointer("/input_audio/data")
                                .and_then(|d| d.as_str())
                                .ok_or("input_audio part without data")?;
                            let format = part
                                .pointer("/input_audio/format")
                                .and_then(|f| f.as_str())
                                .ok_or("input_audio part without format (wav, mp3 or flac)")?;
                            media.push(Media::from_input_audio(data, format).map_err(|e| e.to_string())?);
                            texts.push(media_marker().to_string());
                        }
                        other => return Err(format!("unsupported content part type {other:?}")),
                    }
                }
                (texts.join("\n"), media)
            }
            other => return Err(format!("content must be a string, null or an array of parts, got {other}")),
        };
        Ok(Self {
            role: w.role,
            content,
            media,
            reasoning_content: w.reasoning_content,
            tool_calls: w.tool_calls,
            tool_call_id: w.tool_call_id,
            name: w.name,
        })
    }
}

impl From<ChatMessage> for WireMessage {
    fn from(m: ChatMessage) -> Self {
        let content = if m.media.is_empty() {
            serde_json::Value::String(m.content)
        } else {
            // Back to OpenAI parts: text segments around each marker, media in between.
            let mut parts = Vec::new();
            let mut media = m.media.iter();
            for (i, segment) in m.content.split(media_marker()).enumerate() {
                if i > 0
                    && let Some(item) = media.next()
                {
                    parts.push(match item.kind {
                        MediaKind::Image => {
                            serde_json::json!({"type": "image_url", "image_url": {"url": item.to_data_url()}})
                        }
                        MediaKind::Audio => {
                            let (data, format) = item.to_input_audio();
                            serde_json::json!({"type": "input_audio", "input_audio": {"data": data, "format": format}})
                        }
                    });
                }
                let text = segment.trim_matches('\n');
                if !text.is_empty() {
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
            }
            serde_json::Value::Array(parts)
        };
        Self {
            role: m.role,
            content,
            reasoning_content: m.reasoning_content,
            tool_calls: m.tool_calls,
            tool_call_id: m.tool_call_id,
            name: m.name,
        }
    }
}

pub(crate) fn string_or_seq<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stop {
        One(String),
        Many(Vec<String>),
    }
    Ok(match Option::<Stop>::deserialize(d)? {
        None => Vec::new(),
        Some(Stop::One(s)) => vec![s],
        Some(Stop::Many(v)) => v,
    })
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), ..Default::default() }
    }
    /// A user message with images and/or audio (each placed before the text).
    pub fn user_with_media(text: impl Into<String>, media: Vec<Media>) -> Self {
        let markers: String = media.iter().map(|_| format!("{}\n", media_marker())).collect();
        Self { media, ..Self::new("user", format!("{markers}{}", text.into())) }
    }
    /// [`Self::user_with_media`] for images.
    pub fn user_with_images(text: impl Into<String>, images: Vec<Media>) -> Self {
        debug_assert!(images.iter().all(|m| m.kind == MediaKind::Image), "user_with_images given non-image media");
        Self::user_with_media(text, images)
    }
    /// [`Self::user_with_media`] for audio.
    pub fn user_with_audio(text: impl Into<String>, audio: Vec<Media>) -> Self {
        debug_assert!(audio.iter().all(|m| m.kind == MediaKind::Audio), "user_with_audio given non-audio media");
        Self::user_with_media(text, audio)
    }
    /// Result of a tool call, answering `call`.
    pub fn tool_result(call: &ToolCall, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(call.id.clone()),
            name: Some(call.function.name.clone()),
            ..Self::new("tool", content)
        }
    }
    /// Assistant turn that called tools (to append to the history before tool results).
    pub fn assistant_tool_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self { tool_calls: calls, ..Self::new("assistant", content) }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate. `None` = until end-of-generation or the context is full.
    #[serde(alias = "max_completion_tokens")]
    pub max_tokens: Option<u32>,
    /// Stop strings (OpenAI accepts a single string or an array).
    #[serde(deserialize_with = "string_or_seq")]
    pub stop: Vec<String>,
    /// Jinja template text or a llama.cpp built-in name (e.g. `chatml`). Default: the model's.
    pub chat_template: Option<String>,
    /// Extra variables for the chat template, e.g. `{"enable_thinking": false}`.
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Constrain the output to JSON (OpenAI-compatible).
    pub response_format: Option<ResponseFormat>,
    /// Functions the model may call (OpenAI `tools`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    /// `auto` (default), `none`, `required` or a specific function.
    pub tool_choice: Option<ToolChoice>,
    /// Allow several calls in one turn (default true; some formats only support one).
    pub parallel_tool_calls: Option<bool>,
    #[serde(flatten)]
    pub sampling: SamplingParams,
}

/// OpenAI-style `response_format`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Unconstrained text (same as omitting it).
    Text,
    /// Any JSON object; `schema` optionally constrains it (llama-server extension).
    JsonObject {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
    },
    /// `{"type":"json_schema","json_schema":{"name":"...","schema":{...}}}`
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    #[serde(default)]
    pub name: Option<String>,
    pub schema: serde_json::Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

impl ResponseFormat {
    /// GBNF grammar enforcing this format, if any.
    pub(crate) fn grammar(&self) -> Result<Option<String>> {
        let schema = match self {
            ResponseFormat::Text => return Ok(None),
            ResponseFormat::JsonObject { schema: None } => return Ok(Some(crate::json_schema::any_json_grammar())),
            ResponseFormat::JsonObject { schema: Some(s) } => s,
            ResponseFormat::JsonSchema { json_schema } => &json_schema.schema,
        };
        crate::json_schema::json_schema_to_grammar(schema).map(Some).map_err(Error::JsonSchema)
    }
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self { messages, ..Default::default() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// End-of-generation token or a stop string.
    Stop,
    /// `max_tokens` reached or the context is full.
    Length,
    Cancelled,
    /// The model called one or more tools (see `ChatResponse::tool_calls`).
    ToolCalls,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    /// Prompt tokens reused from the previous request's KV cache.
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub prompt_ms: f64,
    pub generation_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    /// Reply text, with tool-call markup and reasoning removed.
    pub content: String,
    /// `<think>` reasoning, if the model produced any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl ChatResponse {
    /// This response as an assistant message, for appending to the conversation.
    pub fn to_message(&self) -> ChatMessage {
        ChatMessage {
            reasoning_content: self.reasoning_content.clone(),
            tool_calls: self.tool_calls.clone(),
            ..ChatMessage::assistant(self.content.clone())
        }
    }
}

/// Streamed output of [`ChatSession::complete_stream`].
#[derive(Debug, Clone, Copy)]
pub enum StreamEvent<'a> {
    /// Reply text.
    Content(&'a str),
    /// Reasoning (`<think>`) text, streamed as generated.
    Reasoning(&'a str),
    /// A complete, parsed tool call (emitted after generation ends).
    ToolCall(&'a ToolCall),
}

/// Each request gets the next `epoch` and is cancelled when `cancelled >= epoch`, so a
/// cancel never leaks into a later request.
#[derive(Debug, Default)]
struct CancelState {
    /// Number of the current (or last finished) request; 0 = none yet.
    epoch: AtomicU64,
    /// Highest request number that has been cancelled.
    cancelled: AtomicU64,
}

impl CancelState {
    fn is_cancelled(&self) -> bool {
        let epoch = self.epoch.load(Ordering::SeqCst);
        epoch > 0 && self.cancelled.load(Ordering::SeqCst) >= epoch
    }
}

/// Thread-safe handle that cancels the running request of a session.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<CancelState>);

impl CancelToken {
    /// Cancel the running request. No-op when idle.
    pub fn cancel(&self) {
        self.0.cancelled.fetch_max(self.0.epoch.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    /// Whether the current request has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    /// A canceller for the session's *next* request. Only valid while the caller has
    /// exclusive use of the session until that request starts (e.g. holds its lock).
    pub fn next_request(&self) -> RequestCancel {
        RequestCancel { state: self.0.clone(), epoch: self.0.epoch.load(Ordering::SeqCst) + 1 }
    }

    /// A canceller for request number `id` (from [`RequestCancel::id`]).
    pub fn request(&self, id: u64) -> RequestCancel {
        RequestCancel { state: self.0.clone(), epoch: id }
    }

    /// Start a new request (called at the top of every public generation entry point).
    pub(crate) fn begin(&self) {
        self.0.epoch.fetch_add(1, Ordering::SeqCst);
    }
}

/// Cancels one specific request (see [`CancelToken::next_request`]).
#[derive(Debug, Clone)]
pub struct RequestCancel {
    state: Arc<CancelState>,
    epoch: u64,
}

impl RequestCancel {
    pub fn cancel(&self) {
        self.state.cancelled.fetch_max(self.epoch, Ordering::SeqCst);
    }

    /// The request number this cancels (see [`CancelToken::request`]).
    pub fn id(&self) -> u64 {
        self.epoch
    }
}

unsafe extern "C" fn abort_callback(data: *mut c_void) -> bool {
    unsafe { &*(data as *const CancelState) }.is_cancelled()
}

/// One inference context for chat. Requests are stateless; the KV cache is reused across
/// calls by longest common prefix.
pub struct ChatSession {
    model: Arc<Model>,
    ctx: NonNull<sys::llama_context>,
    batch: Batch,
    n_batch: usize,
    /// What the KV cache of sequence 0 holds, in order (for prefix reuse).
    cached: Vec<CacheItem>,
    cancel: CancelToken,
    /// The model's Jinja chat template, compiled on first use (`Err` = unusable, use built-in).
    model_template: Option<std::result::Result<ChatTemplate, String>>,
    /// Next position in sequence 0 (sum of the cached items' positions; M-RoPE images
    /// use a different number of positions than tokens).
    n_past: usize,
    /// Shapes of media seen before, so cached media can be re-tokenized without decoding.
    media_memo: MediaMemo,
    /// Minimum chunk for KV-cache reuse by shifting (0 = off or unsupported).
    cache_reuse: usize,
}

/// Error text used internally when a placeholder media chunk would have to be evaluated;
/// `complete_stream` then re-tokenizes with real data.
const PLACEHOLDER_EVAL: &str = "internal: placeholder media chunk needs real data";

/// KV-cache items for a tokenized prompt, in order.
fn segment_items(segments: &[Segment]) -> Vec<CacheItem> {
    segments
        .iter()
        .flat_map(|seg| match seg {
            Segment::Text(t) => t.iter().map(|&t| CacheItem::Token(t)).collect::<Vec<_>>(),
            Segment::Media(m) => vec![CacheItem::Media { id: m.id.clone(), n_tokens: m.n_tokens, n_pos: m.n_pos }],
        })
        .collect()
}

/// Whether any placeholder media chunk lies at or after item index `keep` (must be evaluated).
fn placeholder_after(segments: &[Segment], keep: usize) -> bool {
    let mut index = 0;
    for seg in segments {
        match seg {
            Segment::Text(t) => index += t.len(),
            Segment::Media(m) => {
                if m.placeholder && index >= keep {
                    return true;
                }
                index += 1;
            }
        }
    }
    false
}

/// One entry of the KV cache: a token, or a whole image identified by its content hash.
#[derive(Debug, Clone)]
enum CacheItem {
    Token(sys::llama_token),
    Media { id: String, n_tokens: usize, n_pos: usize },
}

/// Identity for prefix matching: token id, or image hash + token count. `n_pos` is
/// bookkeeping (measured when evaluated) and deliberately not compared.
impl PartialEq for CacheItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CacheItem::Token(a), CacheItem::Token(b)) => a == b,
            (CacheItem::Media { id: a, n_tokens: na, .. }, CacheItem::Media { id: b, n_tokens: nb, .. }) => {
                a == b && na == nb
            }
            _ => false,
        }
    }
}

impl CacheItem {
    fn n_pos(&self) -> usize {
        match self {
            CacheItem::Token(_) => 1,
            CacheItem::Media { n_pos, .. } => *n_pos,
        }
    }
    fn n_tokens(&self) -> usize {
        match self {
            CacheItem::Token(_) => 1,
            CacheItem::Media { n_tokens, .. } => *n_tokens,
        }
    }
}

/// What to evaluate before generating.
pub(crate) enum Prompt<'a> {
    Tokens(&'a [sys::llama_token]),
    /// Text + image chunks (mtmd), evaluated from position 0.
    Media(&'a Chunks),
}

// The context is used from one thread at a time (&mut self), but may move between threads.
unsafe impl Send for ChatSession {}

impl Drop for ChatSession {
    fn drop(&mut self) {
        let _guard = crate::backend::setup_lock();
        unsafe { sys::llama_free(self.ctx.as_ptr()) };
    }
}

pub(crate) fn default_threads() -> i32 {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (n / 2).max(1) as i32
}

impl ChatSession {
    pub fn new(model: Arc<Model>, params: &SessionParams) -> Result<Self> {
        let n_ctx = match params.n_ctx {
            0 => model.info().n_ctx_train.clamp(512, DEFAULT_MAX_CTX),
            n => n,
        };
        // At least 32 (or the whole context if smaller), at most the context; never panics.
        let n_ctx = n_ctx.max(1);
        let n_batch = params.n_batch.max(32.min(n_ctx)).min(n_ctx);
        let threads = params.n_threads.unwrap_or_else(default_threads);
        let cancel = CancelToken::default();

        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = n_ctx;
        cp.n_batch = n_batch;
        cp.n_ubatch = n_batch;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.embeddings = false;
        cp.no_perf = true;
        cp.abort_callback = Some(abort_callback);
        cp.abort_callback_data = Arc::as_ptr(&cancel.0) as *mut c_void;

        let ctx = {
            let _guard = crate::backend::setup_lock();
            unsafe { sys::llama_init_from_model(model.as_ptr(), cp) }
        };
        let ctx = NonNull::new(ctx).ok_or(Error::ContextCreate)?;
        let n_batch = unsafe { sys::llama_n_batch(ctx.as_ptr()) } as usize;
        let can_shift = unsafe { sys::llama_memory_can_shift(sys::llama_get_memory(ctx.as_ptr())) };
        if params.cache_reuse > 0 && !can_shift {
            log::debug!(target: "offgrid", "this model's KV cache can't shift: cache_reuse is off");
        }
        Ok(Self {
            model,
            ctx,
            batch: Batch::new(n_batch, 1),
            n_batch,
            cached: Vec::new(),
            cancel,
            model_template: None,
            n_past: 0,
            media_memo: MediaMemo::default(),
            cache_reuse: if can_shift { params.cache_reuse as usize } else { 0 },
        })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn n_ctx(&self) -> usize {
        unsafe { sys::llama_n_ctx(self.ctx.as_ptr()) as usize }
    }

    /// Handle for cancelling the in-flight request from another thread.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// Drop the KV cache.
    pub fn reset(&mut self) {
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ctx.as_ptr()), true) };
        self.cached.clear();
        self.n_past = 0;
    }

    /// Render the prompt: `req.chat_template`, else the model's Jinja template, else
    /// llama.cpp's built-in. See `Rendered` for templates that ignore `tools`.
    pub fn render_prompt(&mut self, req: &ChatRequest) -> Result<String> {
        self.render(req).map(|r| r.prompt)
    }

    fn render(&mut self, req: &ChatRequest) -> Result<Rendered> {
        let use_tools = tools_active(req);
        let tools = if use_tools {
            Some(serde_json::to_value(&req.tools).map_err(|e| Error::InvalidArgument(e.to_string()))?)
        } else {
            None
        };
        let bos = self.model.token_text(unsafe { sys::llama_vocab_bos(self.model.vocab()) });
        let eos = self.model.token_text(unsafe { sys::llama_vocab_eos(self.model.vocab()) });
        let base = RenderArgs {
            messages: &serde_json::Value::Null,
            tools: None,
            add_generation_prompt: true,
            bos_token: &bos,
            eos_token: &eos,
            extra: req.chat_template_kwargs.as_ref(),
        };

        let owned;
        let template = match req.chat_template.as_deref() {
            Some(t) if is_jinja(t) => {
                owned = ChatTemplate::new(t)?;
                Ok(&owned)
            }
            Some(_) if tools.is_some() => {
                return Err(Error::InvalidArgument(
                    "built-in chat templates can't show tools to the model; use the model's template or Jinja text"
                        .into(),
                ));
            }
            Some(name) => return Ok(Rendered::plain(self.apply_builtin_template(&req.messages, Some(name))?)),
            None => {
                if self.model_template.is_none() {
                    self.model_template = Some(match self.model.chat_template() {
                        Some(src) => ChatTemplate::new(&src).map_err(|e| e.to_string()),
                        None => Err("model has no chat template".into()),
                    });
                }
                self.model_template.as_ref().expect("set above").as_ref().map_err(String::clone)
            }
        };

        let tool_history = req.messages.iter().any(|m| m.role == "tool" || !m.tool_calls.is_empty());
        let polyfill = (use_tools || tool_history) && template.as_ref().is_ok_and(|t| ignores_tools(t, &base));
        let rendered = match template {
            Ok(t) if polyfill => render_polyfilled(t, &req.messages, use_tools.then_some(req.tools.as_slice()), &base),
            Ok(t) => render_jinja(t, &req.messages, &RenderArgs { tools: tools.as_ref(), ..base }),
            Err(e) => Err(Error::ChatTemplate(e)),
        };
        if req.chat_template.is_some() {
            // An explicit Jinja template has no fallback.
            return rendered.map(|prompt| Rendered { prompt, tools_polyfilled: polyfill });
        }
        match rendered {
            Ok(prompt) => Ok(Rendered { prompt, tools_polyfilled: polyfill }),
            Err(e) if tools.is_some() => {
                // Built-in templates can't show tools to the model.
                Err(Error::ChatTemplate(format!("tool calling needs the model's Jinja template: {e}")))
            }
            Err(e) => {
                log::warn!(target: "offgrid", "jinja chat template unusable ({e}); using llama.cpp built-in");
                Ok(Rendered::plain(self.apply_builtin_template(&req.messages, None)?))
            }
        }
    }

    /// Render with llama.cpp's built-in templates (no Jinja; fixed list of known formats).
    fn apply_builtin_template(&self, messages: &[ChatMessage], template: Option<&str>) -> Result<String> {
        let owned: Vec<(CString, CString)> = messages
            .iter()
            .map(|m| Ok((CString::new(m.role.as_str())?, CString::new(m.content.as_str())?)))
            .collect::<Result<_, std::ffi::NulError>>()
            .map_err(|_| Error::InvalidArgument("message contains NUL".into()))?;
        let msgs: Vec<sys::llama_chat_message> =
            owned.iter().map(|(r, c)| sys::llama_chat_message { role: r.as_ptr(), content: c.as_ptr() }).collect();
        let tmpl = match template {
            Some(t) => Some(CString::new(t).map_err(|_| Error::InvalidArgument("template contains NUL".into()))?),
            None => self.model.chat_template().and_then(|t| CString::new(t).ok()),
        };
        let tmpl_ptr = tmpl.as_ref().map_or(std::ptr::null(), |t| t.as_ptr());

        let mut buf = vec![0u8; messages.iter().map(|m| m.content.len() + m.role.len()).sum::<usize>() * 2 + 256];
        for _ in 0..2 {
            let n = unsafe {
                sys::llama_chat_apply_template(
                    tmpl_ptr,
                    msgs.as_ptr(),
                    msgs.len(),
                    true,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n < 0 {
                return Err(Error::ChatTemplate("template not supported by llama_chat_apply_template".into()));
            }
            if (n as usize) <= buf.len() {
                buf.truncate(n as usize);
                return String::from_utf8(buf).map_err(|e| Error::ChatTemplate(e.to_string()));
            }
            buf.resize(n as usize, 0);
        }
        Err(Error::ChatTemplate("template output size changed".into()))
    }

    /// Chat completion. `on_piece` receives streamed reply text; return `false` to cancel.
    pub fn complete(&mut self, req: &ChatRequest, mut on_piece: impl FnMut(&str) -> bool) -> Result<ChatResponse> {
        self.complete_stream(req, |event| match event {
            StreamEvent::Content(text) => on_piece(text),
            _ => true,
        })
    }

    /// Chat completion with streamed events: reply text and reasoning as they are
    /// generated, then each parsed tool call. Return `false` to cancel.
    pub fn complete_stream(
        &mut self,
        req: &ChatRequest,
        mut on_event: impl FnMut(StreamEvent<'_>) -> bool,
    ) -> Result<ChatResponse> {
        self.cancel.begin();
        if req.messages.is_empty() {
            return Err(Error::InvalidArgument("messages is empty".into()));
        }
        let use_tools = tools_active(req);
        let Rendered { prompt, tools_polyfilled } = self.render(req)?;
        let format = if use_tools && tools_polyfilled {
            ToolFormat::Polyfill
        } else if use_tools {
            tool_parse::detect_format(&self.template_source(req).unwrap_or_default())
        } else {
            ToolFormat::Generic
        };
        // Some templates open the reasoning block in the generation prompt.
        let think_open = prompt.trim_end().ends_with("<think>");

        let media: Vec<&Media> = req.messages.iter().flat_map(|m| &m.media).collect();
        let (tokens, chunks) = if media.is_empty() {
            let mut tokens = self.model.tokenize(&prompt, true, true)?;
            // Templates often include BOS text themselves; avoid a doubled BOS.
            let bos = unsafe { sys::llama_vocab_bos(self.model.vocab()) };
            if tokens.len() >= 2 && tokens[0] == bos && tokens[1] == bos {
                tokens.remove(0);
            }
            (tokens, None)
        } else {
            let model = self.model.clone();
            let projector = model.projector().ok_or_else(|| {
                Error::InvalidArgument(
                    "request has images/audio but the model has no multimodal projector (mmproj)".into(),
                )
            })?;
            let bos = model.token_text(unsafe { sys::llama_vocab_bos(model.vocab()) });
            let add_bos = bos.is_empty() || !prompt.starts_with(&bos);
            // Media still in the KV cache is tokenized from its remembered shape (no decode).
            let cached = self.cached_media_hashes();
            let mut chunks =
                projector.tokenize(&prompt, &media, add_bos, &mut self.media_memo, &|id| cached.contains(id))?;
            let segments = chunks.segments();
            if placeholder_after(&segments, self.reusable_prefix(&segment_items(&segments))) {
                // The cache diverges before some remembered media: it must be encoded again.
                chunks = projector.tokenize(&prompt, &media, add_bos, &mut self.media_memo, &|_| false)?;
            }
            (Vec::new(), Some(chunks))
        };
        let prompt_input = match &chunks {
            Some(c) => Prompt::Media(c),
            None => Prompt::Tokens(&tokens),
        };

        // At most one source of grammar: raw grammar, response_format, or tools.
        let mut grammars: Vec<GrammarSpec> = Vec::new();
        if let Some(g) = &req.sampling.grammar {
            grammars.push(GrammarSpec::strict(g.clone()));
        }
        if let Some(g) = req.response_format.as_ref().map(ResponseFormat::grammar).transpose()?.flatten() {
            grammars.push(GrammarSpec::strict(g));
        }
        if use_tools {
            let choice = req.tool_choice.clone().unwrap_or_default();
            if let Some(g) = tool_grammar(format, &req.tools, &choice, req.parallel_tool_calls.unwrap_or(true))? {
                grammars.push(g);
            }
        }
        if grammars.len() > 1 {
            return Err(Error::InvalidArgument(
                "only one of grammar, response_format and (constrained) tools can be used at once".into(),
            ));
        }

        // A strict grammar forces JSON / a tool call from the first token, so a reasoning
        // block opened by the prompt can never close: don't treat the output as reasoning.
        let think_open = think_open && !grammars.first().is_some_and(|g| g.lazy_patterns.is_empty());
        let marker = if use_tools { tool_parse::tool_call_start_marker(format) } else { None };
        // Hold back a reply starting with `{`: it may be a bare call object (any format).
        let mut filter = StreamFilter::new(marker, use_tools, think_open);
        let mut cancelled_by_callback = false;
        let mut streamed = String::new(); // content delivered so far
        let mut deliver = |pieces: Vec<(PieceKind, String)>, on_event: &mut dyn FnMut(StreamEvent<'_>) -> bool| {
            pieces.iter().all(|(kind, text)| match kind {
                PieceKind::Content => {
                    streamed.push_str(text);
                    on_event(StreamEvent::Content(text))
                }
                PieceKind::Reasoning => on_event(StreamEvent::Reasoning(text)),
            })
        };
        let mut on_piece = |piece: &str| {
            let ok = deliver(filter.push(piece), &mut on_event);
            cancelled_by_callback |= !ok;
            ok
        };
        let mut stop = Cow::Borrowed(&req.stop);
        if format == ToolFormat::Polyfill {
            stop.to_mut().push(POLYFILL_STOP.into());
        }
        let model = self.model.clone();
        let sampler = || Sampler::new(&model, &req.sampling, grammars.first(), false);
        let mut result = self.run(prompt_input, req.max_tokens, &stop, sampler()?, use_tools, &mut on_piece);
        if let (Err(Error::Media(msg)), Some(_)) = (&result, &chunks)
            && msg == PLACEHOLDER_EVAL
        {
            // The KV prefix shrank unexpectedly (e.g. a model that can't trim its cache):
            // nothing was generated yet, so tokenize with real data and run again.
            let model = self.model.clone();
            let projector = model.projector().expect("media implies a projector");
            let bos = model.token_text(unsafe { sys::llama_vocab_bos(model.vocab()) });
            let add_bos = bos.is_empty() || !prompt.starts_with(&bos);
            let real = projector.tokenize(&prompt, &media, add_bos, &mut self.media_memo, &|_| false)?;
            result = self.run(Prompt::Media(&real), req.max_tokens, &stop, sampler()?, use_tools, &mut on_piece);
        }
        let mut resp = result?;
        // Forget shapes of media no longer cached.
        let cached = self.cached_media_hashes();
        self.media_memo.retain(|id| cached.contains(id));
        if !cancelled_by_callback && resp.finish_reason != FinishReason::Cancelled {
            deliver(filter.flush(), &mut on_event);
        }

        let parsed = if use_tools {
            let parsed = tool_parse::parse_output(format, &resp.content, think_open);
            // Calls to functions that weren't offered (e.g. a plain `{"name": ...}` JSON reply
            // in the generic format) are not tool calls: keep the text as content.
            if parsed.calls.iter().all(|c| req.tools.iter().any(|t| t.function.name == c.name)) {
                parsed
            } else {
                let (reasoning, content) = tool_parse::split_reasoning(&resp.content, think_open);
                tool_parse::Parsed { content: content.trim().to_string(), reasoning, calls: Vec::new() }
            }
        } else {
            let (reasoning, content) = tool_parse::split_reasoning(&resp.content, think_open);
            tool_parse::Parsed { content: content.trim().to_string(), reasoning, calls: Vec::new() }
        };
        resp.content = parsed.content;
        resp.reasoning_content = parsed.reasoning;
        // Deliver text the stream filter held back (e.g. a `{...}` reply that wasn't a call).
        if !cancelled_by_callback && resp.finish_reason != FinishReason::Cancelled {
            let shown = streamed.trim();
            if let Some(missing) = resp.content.strip_prefix(shown)
                && !missing.trim().is_empty()
            {
                on_event(StreamEvent::Content(missing));
            }
        }
        resp.tool_calls = parsed.calls.into_iter().map(|c| ToolCall::new(c.name, c.arguments)).collect();
        if !resp.tool_calls.is_empty() && resp.finish_reason == FinishReason::Stop {
            resp.finish_reason = FinishReason::ToolCalls;
        }
        for call in &resp.tool_calls {
            on_event(StreamEvent::ToolCall(call));
        }
        Ok(resp)
    }

    /// Complete and run requested calls through `call_tool` until the model answers
    /// without tools (forced after `max_rounds`). Returns the answer and full conversation.
    pub fn run_tools(
        &mut self,
        req: &ChatRequest,
        max_rounds: usize,
        mut call_tool: impl FnMut(&ToolCall) -> String,
        mut on_piece: impl FnMut(&str) -> bool,
    ) -> Result<(ChatResponse, Vec<ChatMessage>)> {
        let mut req = req.clone();
        for round in 0..=max_rounds {
            if round == max_rounds {
                req.tool_choice = Some(ToolChoice::None);
            }
            let resp = self.complete(&req, &mut on_piece)?;
            req.messages.push(resp.to_message());
            if resp.tool_calls.is_empty() || resp.finish_reason != FinishReason::ToolCalls {
                return Ok((resp, req.messages));
            }
            for call in &resp.tool_calls {
                let result = call_tool(call);
                req.messages.push(ChatMessage::tool_result(call, result));
            }
        }
        unreachable!("the last round runs with tool_choice none")
    }

    /// The Jinja source used for this request (override text or the model's template).
    fn template_source(&self, req: &ChatRequest) -> Option<String> {
        match req.chat_template.as_deref() {
            Some(t) if is_jinja(t) => Some(t.to_string()),
            Some(_) => None,
            None => self.model.chat_template(),
        }
    }

    /// Raw text completion without a chat template.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: Option<u32>,
        stop: &[String],
        sampling: &SamplingParams,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        self.cancel.begin();
        let tokens = self.model.tokenize(prompt, true, true)?;
        let sampler = Sampler::new(&self.model, sampling, None, false)?;
        self.run(Prompt::Tokens(&tokens), max_tokens, stop, sampler, false, on_piece)
    }

    pub(crate) fn run(
        &mut self,
        prompt: Prompt<'_>,
        max_tokens: Option<u32>,
        stop: &[String],
        sampler: Sampler,
        special: bool,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let result = self.run_inner(prompt, max_tokens, stop, sampler, special, on_piece);
        if result.is_err() {
            // KV state may be partially written; start clean next time.
            self.reset();
        }
        result
    }

    fn run_inner(
        &mut self,
        prompt: Prompt<'_>,
        max_tokens: Option<u32>,
        stop: &[String],
        sampler: Sampler,
        special: bool,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let t_prompt = Instant::now();
        let segments = match prompt {
            Prompt::Tokens(t) => vec![Segment::Text(t.to_vec())],
            Prompt::Media(chunks) => chunks.segments(),
        };
        let mut usage = Usage::default();
        if !self.eval_prompt(&segments, &mut usage)? {
            self.reset();
            return Ok(self.finish(String::new(), FinishReason::Cancelled, usage));
        }
        usage.prompt_ms = t_prompt.elapsed().as_secs_f64() * 1000.0;
        self.generate_tokens(sampler, usage, max_tokens, stop, special, on_piece)
    }

    /// Number of leading `items` that can be kept from the KV cache: the common prefix,
    /// minus at least one item so the last one is evaluated (logits).
    fn reusable_prefix(&self, items: &[CacheItem]) -> usize {
        let common = self.cached.iter().zip(items).take_while(|(a, b)| a == b).count();
        common.min(items.len().saturating_sub(1))
    }

    /// After the common prefix (`keep` items), move later cached chunks that match the prompt
    /// to their new positions (llama.cpp's cache reuse). Chunks can only move to earlier
    /// positions: new tokens must follow the last cached position, so text inserted before a
    /// chunk means evaluating it again. Returns the number of leading items now in place; the
    /// caller drops the rest.
    fn reuse_chunks(&mut self, items: &[CacheItem], keep: usize) -> usize {
        // Positions equal item indices only for text.
        let text = |c: &CacheItem| matches!(c, CacheItem::Token(_));
        if !self.cached.iter().all(text) || !items.iter().all(text) {
            return keep;
        }
        let mem = unsafe { sys::llama_get_memory(self.ctx.as_ptr()) };
        let limit = items.len() - 1; // the last item is always evaluated (logits)
        let (mut head_c, mut head_p) = (keep, keep); // in the cache, in the prompt
        let mut reused = 0;
        while head_c < self.cached.len() && head_p < limit {
            let n = self.cached[head_c..].iter().zip(&items[head_p..limit]).take_while(|(a, b)| a == b).count();
            if n < self.cache_reuse {
                head_c += 1;
                continue;
            }
            // Drop what lies between the prompt head and the chunk, then shift the chunk.
            if !unsafe { sys::llama_memory_seq_rm(mem, 0, head_p as i32, head_c as i32) } {
                self.reset();
                return 0;
            }
            unsafe {
                sys::llama_memory_seq_add(mem, 0, head_c as i32, (head_c + n) as i32, head_p as i32 - head_c as i32)
            };
            for i in 0..n {
                self.cached[head_p + i] = self.cached[head_c + i].clone();
            }
            head_c += n;
            head_p += n;
            reused += n;
        }
        if reused > 0 {
            log::debug!(target: "offgrid", "cache reuse: {reused} tokens shifted, prefix {keep} -> {head_p}");
        }
        head_p
    }

    /// Content hashes (without `#k`) of media currently in the KV cache.
    fn cached_media_hashes(&self) -> std::collections::HashSet<String> {
        self.cached
            .iter()
            .filter_map(|c| match c {
                CacheItem::Media { id, .. } => Some(id.split('#').next().unwrap_or(id).to_string()),
                CacheItem::Token(_) => None,
            })
            .collect()
    }

    /// Bring the KV cache to `segments`, reusing the longest cached prefix. Returns `false`
    /// if cancelled.
    fn eval_prompt(&mut self, segments: &[Segment], usage: &mut Usage) -> Result<bool> {
        let n_ctx = self.n_ctx();
        let items = segment_items(segments);
        if items.is_empty() {
            return Err(Error::InvalidArgument("empty prompt".into()));
        }
        if matches!(items.last(), Some(CacheItem::Media { .. })) {
            // Media chunks produce no logits to sample from.
            return Err(Error::InvalidArgument(
                "the prompt ends with an image/audio item; the chat template must add text after it".into(),
            ));
        }
        // Capacity is in KV cells = tokens (M-RoPE images use fewer positions than tokens).
        let total_tokens: usize = items.iter().map(CacheItem::n_tokens).sum();
        if total_tokens >= n_ctx {
            return Err(Error::ContextOverflow { needed: total_tokens + 1, n_ctx });
        }
        usage.prompt_tokens = items.iter().map(CacheItem::n_tokens).sum::<usize>() as u32;

        let mut keep = self.reusable_prefix(&items);
        if self.cache_reuse > 0 && keep < self.cached.len() {
            keep = self.reuse_chunks(&items, keep);
        }
        if keep < self.cached.len() {
            let pos: usize = self.cached[..keep].iter().map(CacheItem::n_pos).sum();
            let mem = unsafe { sys::llama_get_memory(self.ctx.as_ptr()) };
            if unsafe { sys::llama_memory_seq_rm(mem, 0, pos as i32, -1) } {
                self.cached.truncate(keep);
            } else {
                self.reset();
                keep = 0;
            }
        }
        self.n_past = self.cached.iter().map(CacheItem::n_pos).sum();
        usage.cached_tokens = self.cached.iter().map(CacheItem::n_tokens).sum::<usize>() as u32;

        let last = items.len() - 1;
        let mut index = 0;
        for seg in segments {
            match seg {
                Segment::Text(tokens) => {
                    let start = index;
                    index += tokens.len();
                    if index <= keep {
                        continue;
                    }
                    let from = keep.saturating_sub(start);
                    for (ci, chunk) in tokens[from..].chunks(self.n_batch).enumerate() {
                        if self.cancel.is_cancelled() {
                            return Ok(false);
                        }
                        self.batch.clear();
                        for (j, &t) in chunk.iter().enumerate() {
                            let item = start + from + ci * self.n_batch + j;
                            self.batch.add(t, self.n_past as i32, &[0], item == last);
                            self.n_past += 1;
                            self.cached.push(CacheItem::Token(t));
                        }
                        if !self.decode()? {
                            return Ok(false);
                        }
                    }
                }
                Segment::Media(m) => {
                    let item = index;
                    index += 1;
                    if index <= keep {
                        continue;
                    }
                    if m.placeholder {
                        return Err(Error::Media(PLACEHOLDER_EVAL.into()));
                    }
                    if self.cancel.is_cancelled() {
                        return Ok(false);
                    }
                    let projector = self.model.projector().expect("media segments imply a projector");
                    let new_pos = match projector.eval_chunk(
                        self.ctx.as_ptr(),
                        m,
                        self.n_past as i32,
                        self.n_batch as i32,
                        item == last,
                    ) {
                        Ok(pos) => pos,
                        // Aborted by the cancel callback while decoding the media embeddings.
                        Err(Error::Decode(2)) if self.cancel.is_cancelled() => return Ok(false),
                        Err(e) => return Err(e),
                    };
                    let n_pos = (new_pos as usize).saturating_sub(self.n_past);
                    if n_pos != m.n_pos {
                        log::debug!(target: "offgrid", "media chunk used {n_pos} positions, expected {}", m.n_pos);
                    }
                    self.n_past = new_pos as usize;
                    self.cached.push(CacheItem::Media { id: m.id.clone(), n_tokens: m.n_tokens, n_pos });
                }
            }
        }
        Ok(true)
    }

    /// Sample and decode until EOG, a stop string, the budget, or cancellation.
    fn generate_tokens(
        &mut self,
        mut sampler: Sampler,
        mut usage: Usage,
        max_tokens: Option<u32>,
        stop: &[String],
        special: bool,
        mut on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let t_gen = Instant::now();
        let used: usize = self.cached.iter().map(CacheItem::n_tokens).sum();
        let budget = self.n_ctx().saturating_sub(used).min(max_tokens.map_or(usize::MAX, |m| m as usize));
        let mut stream = TextStream::new(stop);
        let mut content = String::new();
        let mut piece = Vec::new();
        let mut reason = FinishReason::Length;

        for _ in 0..budget {
            if self.cancel.is_cancelled() {
                reason = FinishReason::Cancelled;
                break;
            }
            let tok = sampler.sample(self.ctx.as_ptr(), -1);
            if self.model.is_eog(tok) {
                reason = FinishReason::Stop;
                break;
            }
            usage.completion_tokens += 1;
            self.model.token_bytes(tok, &mut piece, special);
            let (text, hit_stop) = match stream.push(&piece) {
                Push::Text(t) => (t, false),
                Push::Stop(t) => (t, true),
            };
            if !text.is_empty() {
                content.push_str(&text);
                if !on_piece(&text) {
                    reason = FinishReason::Cancelled;
                    break;
                }
            }
            if hit_stop {
                reason = FinishReason::Stop;
                break;
            }
            self.batch.clear();
            self.batch.add(tok, self.n_past as i32, &[0], true);
            match self.decode() {
                Ok(true) => {}
                Ok(false) => {
                    self.reset();
                    reason = FinishReason::Cancelled;
                    break;
                }
                // No KV slot left (the memory state is unchanged): end like max_tokens.
                Err(Error::Decode(1)) => {
                    reason = FinishReason::Length;
                    break;
                }
                Err(e) => return Err(e),
            }
            self.n_past += 1;
            self.cached.push(CacheItem::Token(tok));
        }
        let tail = stream.finish();
        if !tail.is_empty() && reason != FinishReason::Cancelled {
            on_piece(&tail);
            content.push_str(&tail);
        }
        usage.generation_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        Ok(self.finish(content, reason, usage))
    }

    fn finish(&self, content: String, finish_reason: FinishReason, usage: Usage) -> ChatResponse {
        ChatResponse { content, reasoning_content: None, tool_calls: Vec::new(), finish_reason, usage }
    }

    /// Returns `Ok(false)` when the decode was aborted by the cancel token.
    fn decode(&mut self) -> Result<bool> {
        match unsafe { sys::llama_decode(self.ctx.as_ptr(), self.batch.raw()) } {
            0 => Ok(true),
            2 if self.cancel.is_cancelled() => Ok(false),
            code => Err(Error::Decode(code)),
        }
    }
}

/// A rendered prompt.
struct Rendered {
    prompt: String,
    /// The template ignores `tools`, so the tools and any tool-call history were written
    /// into the messages in the generic `<tool_call>` format.
    tools_polyfilled: bool,
}

impl Rendered {
    fn plain(prompt: String) -> Self {
        Self { prompt, tools_polyfilled: false }
    }
}

/// Whether `template` leaves `tools` out of the prompt. Render errors count as "shows
/// tools" so the template's own error surfaces.
fn ignores_tools(template: &ChatTemplate, base: &RenderArgs) -> bool {
    const PROBE: &str = "offgrid_probe_function";
    let messages = serde_json::json!([{"role": "user", "content": "Hi"}]);
    let tools = serde_json::json!([{"type": "function", "function": {"name": PROBE, "description": "Probe.", "parameters": {"type": "object", "properties": {}}}}]);
    template
        .render(&RenderArgs { messages: &messages, tools: Some(&tools), ..*base })
        .is_ok_and(|prompt| !prompt.contains(PROBE))
}

fn render_polyfilled(
    template: &ChatTemplate,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    base: &RenderArgs,
) -> Result<String> {
    let render = |system_role| {
        let messages = template_messages(&polyfill_messages(messages, tools, system_role))?;
        template.render(&RenderArgs { messages: &messages, ..*base })
    };
    // Templates without a system role (Gemma 2, ...) raise; fold it into the user turn.
    render(true).or_else(|e| render(false).map_err(|_| e))
}

/// Render with the model's own template. Some templates only accept one call per assistant
/// turn (Llama 3.x raises); on failure, retry with multi-call turns split into one per call.
fn render_jinja(template: &ChatTemplate, messages: &[ChatMessage], base: &RenderArgs) -> Result<String> {
    let render = |messages: &[ChatMessage]| {
        let messages = template_messages(messages)?;
        template.render(&RenderArgs { messages: &messages, ..*base })
    };
    render(messages).or_else(|e| match split_multi_calls(messages) {
        Some(split) => render(&split).map_err(|_| e),
        None => Err(e),
    })
}

/// Each assistant turn with several calls becomes one turn per call, each followed by its
/// result. `None` when no turn has more than one call.
fn split_multi_calls(messages: &[ChatMessage]) -> Option<Vec<ChatMessage>> {
    if !messages.iter().any(|m| m.tool_calls.len() > 1) {
        return None;
    }
    let mut out = Vec::with_capacity(messages.len() * 2);
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        i += 1;
        if msg.tool_calls.len() < 2 {
            out.push(msg.clone());
            continue;
        }
        let mut results: Vec<ChatMessage> = Vec::new();
        while i < messages.len() && messages[i].role == "tool" {
            results.push(messages[i].clone());
            i += 1;
        }
        for (n, call) in msg.tool_calls.iter().enumerate() {
            out.push(ChatMessage {
                tool_calls: vec![call.clone()],
                content: if n == 0 { msg.content.clone() } else { String::new() },
                reasoning_content: if n == 0 { msg.reasoning_content.clone() } else { None },
                media: if n == 0 { msg.media.clone() } else { Vec::new() },
                ..msg.clone()
            });
            if let Some(pos) = results.iter().position(|r| r.tool_call_id.as_deref() == Some(call.id.as_str())) {
                out.push(results.remove(pos));
            }
        }
        out.extend(results);
    }
    Some(out)
}

fn tools_active(req: &ChatRequest) -> bool {
    !req.tools.is_empty() && req.tool_choice != Some(ToolChoice::None)
}

/// Messages as templates expect them: tool-call arguments as JSON objects (HF convention;
/// OpenAI sends them as strings), assistant tool-call turns with string content.
fn template_messages(messages: &[ChatMessage]) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(messages).map_err(|e| Error::InvalidArgument(e.to_string()))?;
    for (msg, orig) in value.as_array_mut().into_iter().flatten().zip(messages) {
        // Templates get plain text; images are the media markers mtmd expands later.
        msg["content"] = serde_json::Value::String(orig.content.clone());
        for call in msg.get_mut("tool_calls").and_then(|c| c.as_array_mut()).into_iter().flatten() {
            if let Some(args) = call.pointer_mut("/function/arguments")
                && let Some(text) = args.as_str()
                && let Ok(parsed @ serde_json::Value::Object(_)) = serde_json::from_str::<serde_json::Value>(text)
            {
                *args = parsed;
            }
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::FunctionCall;

    /// Over llama.cpp's template corpus: templates that mention `tools` show them; the rest
    /// get the polyfill and render a whole tool conversation with it.
    /// Llama 3.2's template raises on an assistant turn with several calls; those turns are
    /// rendered split, one call per turn, each followed by its result.
    #[test]
    fn single_call_templates_get_split_turns() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../offgrid-sys/llama.cpp/models/templates");
        let src = std::fs::read_to_string(dir.join("meta-llama-Llama-3.2-3B-Instruct.jinja")).unwrap();
        let t = ChatTemplate::new(&src).unwrap();
        let call = |id: &str, city: &str| ToolCall {
            id: id.into(),
            kind: "function".into(),
            function: FunctionCall { name: "get_weather".into(), arguments: format!(r#"{{"city":"{city}"}}"#) },
        };
        let (a, b) = (call("call_a", "Oslo"), call("call_b", "Paris"));
        let messages = vec![
            ChatMessage::user("Weather in Oslo and Paris?"),
            ChatMessage::assistant_tool_calls("", vec![a.clone(), b.clone()]),
            ChatMessage::tool_result(&a, r#"{"temp_c": 4}"#),
            ChatMessage::tool_result(&b, r#"{"temp_c": 15}"#),
        ];
        let tools =
            serde_json::to_value([Tool::function("get_weather", "Weather", serde_json::json!({"type": "object"}))])
                .unwrap();
        let base = RenderArgs {
            messages: &serde_json::Value::Null,
            tools: Some(&tools),
            add_generation_prompt: true,
            bos_token: "<|begin_of_text|>",
            eos_token: "<|eot_id|>",
            extra: None,
        };
        let direct = t.render(&RenderArgs { messages: &template_messages(&messages).unwrap(), ..base });
        assert!(direct.unwrap_err().to_string().contains("single tool-calls"));
        let prompt = render_jinja(&t, &messages, &base).unwrap();
        let (oslo, temp4, paris, temp15) = (
            prompt.find("Oslo\"").unwrap(),
            prompt.find("\"temp_c\": 4").unwrap(),
            prompt.find("Paris\"").unwrap(),
            prompt.find("15").unwrap(),
        );
        assert!(oslo < temp4 && temp4 < paris && paris < temp15, "each call followed by its result:\n{prompt}");
    }

    #[test]
    fn tool_polyfill_over_template_corpus() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../offgrid-sys/llama.cpp/models/templates");
        let base = RenderArgs {
            messages: &serde_json::Value::Null,
            tools: None,
            add_generation_prompt: true,
            bos_token: "<s>",
            eos_token: "</s>",
            extra: None,
        };
        let call = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: FunctionCall { name: "get_weather".into(), arguments: r#"{"city":"Oslo"}"#.into() },
        };
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Weather in Oslo?"),
            ChatMessage::assistant_tool_calls("", vec![call.clone()]),
            ChatMessage::tool_result(&call, r#"{"temp_c": 4}"#),
        ];
        let tools = [Tool::function("get_weather", "Weather for a city", serde_json::json!({"type": "object"}))];
        let mut polyfilled = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("llama.cpp submodule checked out") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "jinja") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(&path).unwrap();
            let Ok(t) = ChatTemplate::new(&src) else { continue };
            if !ignores_tools(&t, &base) {
                continue;
            }
            assert!(!src.contains("tools"), "{name} mentions tools but was judged to ignore them");
            let prompt =
                render_polyfilled(&t, &messages, Some(&tools), &base).unwrap_or_else(|e| panic!("{name}: {e}"));
            for part in ["# Tools", "\"name\": \"get_weather\"", "<tool_call>", "\"temp_c\": 4", "Weather in Oslo?"] {
                assert!(prompt.contains(part), "{name}: missing {part:?} in {prompt}");
            }
            polyfilled.push(name);
        }
        polyfilled.sort();
        // Gemma 2 has no system role: covers folding the instructions into the user turn.
        assert!(polyfilled.iter().any(|n| n.contains("gemma-2")), "{polyfilled:?}");
        assert!(polyfilled.len() >= 5, "{polyfilled:?}");
    }
}
