use std::ffi::{CString, c_char, c_void};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use aicore_sys as sys;
use serde::{Deserialize, Serialize};

use crate::batch::Batch;
use crate::sampling::Sampler;
use crate::template::{ChatTemplate, RenderArgs, is_jinja};
use crate::text::{PieceKind, Push, StreamFilter, TextStream};
use crate::tool_parse::{self, ToolFormat};
use crate::tools::{GrammarSpec, Tool, ToolCall, ToolChoice, tool_grammar};
use crate::vision::{Chunks, media_marker};
use crate::{Error, Image, Model, Result, SamplingParams};

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
}

impl Default for SessionParams {
    fn default() -> Self {
        Self { n_ctx: 0, n_batch: 512, n_threads: None }
    }
}

/// An OpenAI-style chat message.
///
/// On the wire `content` may be a string, `null`, or an array of parts
/// (`{"type":"text","text":...}` / `{"type":"image_url","image_url":{"url":"data:..."}}`).
/// Internally the text holds one [`media_marker`] per image, in order, and the images are
/// in `images`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "WireMessage", into = "WireMessage")]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool` (templates may accept others).
    pub role: String,
    /// Message text (with a media marker where each image goes).
    pub content: String,
    /// Images referenced by the markers in `content`, in order.
    pub images: Vec<Image>,
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
        let (content, images) = match w.content {
            serde_json::Value::Null => (String::new(), Vec::new()),
            serde_json::Value::String(s) => (s, Vec::new()),
            serde_json::Value::Array(parts) => {
                let (mut texts, mut images) = (Vec::new(), Vec::new());
                for part in parts {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("text") => texts.push(part.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string()),
                        Some("image_url") => {
                            let url = part
                                .pointer("/image_url/url")
                                .or_else(|| part.get("image_url"))
                                .and_then(|u| u.as_str())
                                .ok_or("image_url part without url")?;
                            images.push(Image::from_data_url(url).map_err(|e| e.to_string())?);
                            texts.push(media_marker().to_string());
                        }
                        other => return Err(format!("unsupported content part type {other:?}")),
                    }
                }
                (texts.join("\n"), images)
            }
            other => return Err(format!("content must be a string, null or an array of parts, got {other}")),
        };
        Ok(Self {
            role: w.role,
            content,
            images,
            reasoning_content: w.reasoning_content,
            tool_calls: w.tool_calls,
            tool_call_id: w.tool_call_id,
            name: w.name,
        })
    }
}

impl From<ChatMessage> for WireMessage {
    fn from(m: ChatMessage) -> Self {
        let content = if m.images.is_empty() {
            serde_json::Value::String(m.content)
        } else {
            // Back to OpenAI parts: text segments around each marker, images in between.
            let mut parts = Vec::new();
            let mut images = m.images.iter();
            for (i, segment) in m.content.split(media_marker()).enumerate() {
                if i > 0
                    && let Some(img) = images.next()
                {
                    parts.push(serde_json::json!({"type": "image_url", "image_url": {"url": img.to_data_url()}}));
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

fn string_or_seq<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<String>, D::Error> {
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
    /// A user message with images (each shown before the text).
    pub fn user_with_images(text: impl Into<String>, images: Vec<Image>) -> Self {
        let markers: String = images.iter().map(|_| format!("{}\n", media_marker())).collect();
        Self { images, ..Self::new("user", format!("{markers}{}", text.into())) }
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
    /// Override the model's chat template: Jinja template text, or a llama.cpp built-in
    /// name such as `chatml`. Default: the model's own (Jinja) template.
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

/// Thread-safe handle that cancels the running request of a session.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    fn reset(&self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

unsafe extern "C" fn abort_callback(data: *mut c_void) -> bool {
    unsafe { &*(data as *const AtomicBool) }.load(Ordering::Relaxed)
}

/// One inference context for chat / text generation.
///
/// Requests are stateless (send the full message list each time); the session
/// keeps the KV cache and reuses the longest common token prefix between calls,
/// so multi-turn chats only process the new tokens.
pub struct ChatSession {
    model: Arc<Model>,
    ctx: NonNull<sys::llama_context>,
    batch: Batch,
    n_batch: usize,
    /// Tokens currently in the KV cache for sequence 0.
    cached: Vec<sys::llama_token>,
    cancel: CancelToken,
    /// The model's Jinja chat template, compiled on first use (`Err` = unusable, use built-in).
    model_template: Option<std::result::Result<ChatTemplate, String>>,
    /// Next position in sequence 0 (== cached.len() unless media was evaluated; M-RoPE
    /// images use a different number of positions than tokens).
    n_past: usize,
    /// The KV cache holds media chunks that `cached` doesn't mirror: no prefix reuse.
    kv_untracked: bool,
}

/// What to evaluate before generating.
enum Prompt<'a> {
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
        let n_batch = params.n_batch.clamp(32, n_ctx);
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
        Ok(Self {
            model,
            ctx,
            batch: Batch::new(n_batch, 1),
            n_batch,
            cached: Vec::new(),
            cancel,
            model_template: None,
            n_past: 0,
            kv_untracked: false,
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
        self.kv_untracked = false;
    }

    /// Render the request's messages into the prompt text the model will see.
    ///
    /// Uses `req.chat_template` if set (Jinja text or a built-in name), otherwise the
    /// model's own Jinja template; if that fails to compile or render, falls back to
    /// llama.cpp's built-in template matching the model.
    pub fn render_prompt(&mut self, req: &ChatRequest) -> Result<String> {
        let messages = template_messages(&req.messages)?;
        let tools = if tools_active(req) {
            Some(serde_json::to_value(&req.tools).map_err(|e| Error::InvalidArgument(e.to_string()))?)
        } else {
            None
        };
        let bos = self.model.token_text(unsafe { sys::llama_vocab_bos(self.model.vocab()) });
        let eos = self.model.token_text(unsafe { sys::llama_vocab_eos(self.model.vocab()) });
        let args = RenderArgs {
            messages: &messages,
            tools: tools.as_ref(),
            add_generation_prompt: true,
            bos_token: &bos,
            eos_token: &eos,
            extra: req.chat_template_kwargs.as_ref(),
        };
        match req.chat_template.as_deref() {
            Some(t) if is_jinja(t) => ChatTemplate::new(t)?.render(&args),
            Some(name) => self.apply_builtin_template(&req.messages, Some(name)),
            None => {
                if self.model_template.is_none() {
                    self.model_template = Some(match self.model.chat_template() {
                        Some(src) => ChatTemplate::new(&src).map_err(|e| e.to_string()),
                        None => Err("model has no chat template".into()),
                    });
                }
                let rendered = match self.model_template.as_ref().expect("set above") {
                    Ok(t) => t.render(&args).map_err(|e| e.to_string()),
                    Err(e) => Err(e.clone()),
                };
                rendered.or_else(|e| {
                    if tools.is_some() {
                        // Built-in templates can't show tools to the model.
                        return Err(Error::ChatTemplate(format!("tool calling needs the model's Jinja template: {e}")));
                    }
                    log::warn!(target: "aicore", "jinja chat template unusable ({e}); using llama.cpp built-in");
                    self.apply_builtin_template(&req.messages, None)
                })
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
        let msgs: Vec<sys::llama_chat_message> = owned
            .iter()
            .map(|(r, c)| sys::llama_chat_message { role: r.as_ptr(), content: c.as_ptr() })
            .collect();
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

    /// Chat completion. `on_piece` receives streamed reply text (without reasoning or
    /// tool-call markup); return `false` to cancel. Tool calls and reasoning are returned
    /// parsed in the response. See [`Self::complete_stream`] to also stream reasoning.
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
        if req.messages.is_empty() {
            return Err(Error::InvalidArgument("messages is empty".into()));
        }
        let use_tools = tools_active(req);
        let prompt = self.render_prompt(req)?;
        let format = if use_tools {
            tool_parse::detect_format(&self.template_source(req).unwrap_or_default())
        } else {
            ToolFormat::Generic
        };
        // Some templates open the reasoning block in the generation prompt.
        let think_open = prompt.trim_end().ends_with("<think>");

        let images: Vec<&Image> = req.messages.iter().flat_map(|m| &m.images).collect();
        let (tokens, chunks) = if images.is_empty() {
            let mut tokens = self.model.tokenize(&prompt, true, true)?;
            // Templates often include BOS text themselves; avoid a doubled BOS.
            let bos = unsafe { sys::llama_vocab_bos(self.model.vocab()) };
            if tokens.len() >= 2 && tokens[0] == bos && tokens[1] == bos {
                tokens.remove(0);
            }
            (tokens, None)
        } else {
            let vision = self.model.vision().ok_or_else(|| {
                Error::InvalidArgument("request has images but the model has no vision projector (mmproj)".into())
            })?;
            let bos = self.model.token_text(unsafe { sys::llama_vocab_bos(self.model.vocab()) });
            let add_bos = bos.is_empty() || !prompt.starts_with(&bos);
            (Vec::new(), Some(vision.tokenize(&prompt, &images, add_bos)?))
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

        let marker = if use_tools { tool_parse::tool_call_start_marker(format) } else { None };
        let mut filter = StreamFilter::new(marker, use_tools && format == ToolFormat::Llama3Json, think_open);
        let mut cancelled_by_callback = false;
        let deliver = |pieces: Vec<(PieceKind, String)>, on_event: &mut dyn FnMut(StreamEvent<'_>) -> bool| {
            pieces.iter().all(|(kind, text)| match kind {
                PieceKind::Content => on_event(StreamEvent::Content(text)),
                PieceKind::Reasoning => on_event(StreamEvent::Reasoning(text)),
            })
        };
        let mut resp = self.run(prompt_input, req.max_tokens, &req.stop, &req.sampling, grammars.first(), use_tools, |piece| {
            let ok = deliver(filter.push(piece), &mut on_event);
            cancelled_by_callback |= !ok;
            ok
        })?;
        if !cancelled_by_callback && resp.finish_reason != FinishReason::Cancelled {
            deliver(filter.flush(), &mut on_event);
        }

        let parsed = if use_tools {
            tool_parse::parse_output(format, &resp.content, think_open)
        } else {
            let (reasoning, content) = tool_parse::split_reasoning(&resp.content, think_open);
            tool_parse::Parsed { content: content.trim().to_string(), reasoning, calls: Vec::new() }
        };
        resp.content = parsed.content;
        resp.reasoning_content = parsed.reasoning;
        resp.tool_calls = parsed.calls.into_iter().map(|c| ToolCall::new(c.name, c.arguments)).collect();
        if !resp.tool_calls.is_empty() && resp.finish_reason == FinishReason::Stop {
            resp.finish_reason = FinishReason::ToolCalls;
        }
        for call in &resp.tool_calls {
            on_event(StreamEvent::ToolCall(call));
        }
        Ok(resp)
    }

    /// Run a tool loop: complete, execute requested calls with `call_tool` (returning the
    /// result text, typically JSON), append calls + results, and repeat until the model
    /// answers without calling tools. After `max_rounds` tool rounds a final turn is run
    /// with `tool_choice: none` to force an answer.
    ///
    /// Returns the final response and the full conversation (request messages, assistant
    /// tool-call turns, tool results and the final answer) for continuing the chat.
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
        let tokens = self.model.tokenize(prompt, true, true)?;
        self.run(Prompt::Tokens(&tokens), max_tokens, stop, sampling, None, false, on_piece)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        prompt: Prompt<'_>,
        max_tokens: Option<u32>,
        stop: &[String],
        sampling: &SamplingParams,
        grammar: Option<&GrammarSpec>,
        special: bool,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let result = self.run_inner(prompt, max_tokens, stop, sampling, grammar, special, on_piece);
        if result.is_err() {
            // KV state may be partially written; start clean next time.
            self.reset();
        }
        self.cancel.reset();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_inner(
        &mut self,
        prompt: Prompt<'_>,
        max_tokens: Option<u32>,
        stop: &[String],
        sampling: &SamplingParams,
        grammar: Option<&GrammarSpec>,
        special: bool,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let n_ctx = self.n_ctx();
        let sampler = Sampler::new(&self.model, sampling, grammar)?;
        let t_prompt = Instant::now();
        let tokens = match prompt {
            Prompt::Tokens(t) => t,
            Prompt::Media(chunks) => {
                let n_pos = chunks.n_pos();
                if n_pos >= n_ctx {
                    return Err(Error::ContextOverflow { needed: n_pos + 1, n_ctx });
                }
                let mut usage = Usage { prompt_tokens: chunks.n_tokens() as u32, ..Default::default() };
                self.reset();
                self.kv_untracked = true;
                let vision = self.model.vision().expect("checked by caller");
                self.n_past = vision.eval(self.ctx.as_ptr(), chunks, 0, self.n_batch as i32)? as usize;
                usage.prompt_ms = t_prompt.elapsed().as_secs_f64() * 1000.0;
                return self.generate_tokens(sampler, usage, max_tokens, stop, special, on_piece);
            }
        };
        if tokens.is_empty() {
            return Err(Error::InvalidArgument("empty prompt".into()));
        }
        if tokens.len() >= n_ctx {
            return Err(Error::ContextOverflow { needed: tokens.len() + 1, n_ctx });
        }
        if self.kv_untracked {
            self.reset();
        }
        let mut usage = Usage { prompt_tokens: tokens.len() as u32, ..Default::default() };

        // Reuse the common prefix; always re-evaluate at least the last prompt token for logits.
        let mut keep = self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        keep = keep.min(tokens.len() - 1);
        if keep < self.cached.len() {
            let mem = unsafe { sys::llama_get_memory(self.ctx.as_ptr()) };
            if !unsafe { sys::llama_memory_seq_rm(mem, 0, keep as i32, -1) } {
                self.reset();
                keep = 0;
            }
            self.cached.truncate(keep);
        }
        usage.cached_tokens = keep as u32;

        let rest = &tokens[keep..];
        for (ci, chunk) in rest.chunks(self.n_batch).enumerate() {
            if self.cancel.is_cancelled() {
                return Ok(self.finish(String::new(), FinishReason::Cancelled, usage));
            }
            let last_chunk = (ci + 1) * self.n_batch >= rest.len();
            self.batch.clear();
            for (i, &t) in chunk.iter().enumerate() {
                let pos = self.cached.len() as i32;
                self.batch.add(t, pos, &[0], last_chunk && i == chunk.len() - 1);
                self.cached.push(t);
            }
            if !self.decode()? {
                self.reset();
                return Ok(self.finish(String::new(), FinishReason::Cancelled, usage));
            }
        }
        self.n_past = self.cached.len();
        usage.prompt_ms = t_prompt.elapsed().as_secs_f64() * 1000.0;
        self.generate_tokens(sampler, usage, max_tokens, stop, special, on_piece)
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
        let budget = self.n_ctx().saturating_sub(self.n_past).min(max_tokens.map_or(usize::MAX, |m| m as usize));
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
            if !self.decode()? {
                self.reset();
                reason = FinishReason::Cancelled;
                break;
            }
            self.n_past += 1;
            if !self.kv_untracked {
                self.cached.push(tok);
            }
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
