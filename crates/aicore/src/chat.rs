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
use crate::text::{Push, TextStream};
use crate::{Error, Model, Result, SamplingParams};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
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
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    /// Override the model's chat template: Jinja template text, or a llama.cpp built-in
    /// name such as `chatml`. Default: the model's own (Jinja) template.
    pub chat_template: Option<String>,
    /// Extra variables for the chat template, e.g. `{"enable_thinking": false}`.
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Constrain the output to JSON (OpenAI-compatible).
    pub response_format: Option<ResponseFormat>,
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
    pub content: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
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
    }

    /// Render the request's messages into the prompt text the model will see.
    ///
    /// Uses `req.chat_template` if set (Jinja text or a built-in name), otherwise the
    /// model's own Jinja template; if that fails to compile or render, falls back to
    /// llama.cpp's built-in template matching the model.
    pub fn render_prompt(&mut self, req: &ChatRequest) -> Result<String> {
        let messages = serde_json::to_value(&req.messages).map_err(|e| Error::InvalidArgument(e.to_string()))?;
        let bos = self.model.token_text(unsafe { sys::llama_vocab_bos(self.model.vocab()) });
        let eos = self.model.token_text(unsafe { sys::llama_vocab_eos(self.model.vocab()) });
        let args = RenderArgs {
            messages: &messages,
            tools: None,
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

    /// Chat completion. `on_piece` receives streamed text; return `false` to cancel.
    pub fn complete(&mut self, req: &ChatRequest, on_piece: impl FnMut(&str) -> bool) -> Result<ChatResponse> {
        if req.messages.is_empty() {
            return Err(Error::InvalidArgument("messages is empty".into()));
        }
        let prompt = self.render_prompt(req)?;
        let mut tokens = self.model.tokenize(&prompt, true, true)?;
        // Templates often include BOS text themselves; avoid a doubled BOS.
        let bos = unsafe { sys::llama_vocab_bos(self.model.vocab()) };
        if tokens.len() >= 2 && tokens[0] == bos && tokens[1] == bos {
            tokens.remove(0);
        }
        let sampling = match req.response_format.as_ref().map(ResponseFormat::grammar).transpose()?.flatten() {
            Some(_) if req.sampling.grammar.is_some() => {
                return Err(Error::InvalidArgument("use either grammar or response_format, not both".into()));
            }
            Some(grammar) => &SamplingParams { grammar: Some(grammar), ..req.sampling.clone() },
            None => &req.sampling,
        };
        self.run(&tokens, req.max_tokens, &req.stop, sampling, on_piece)
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
        self.run(&tokens, max_tokens, stop, sampling, on_piece)
    }

    fn run(
        &mut self,
        tokens: &[sys::llama_token],
        max_tokens: Option<u32>,
        stop: &[String],
        sampling: &SamplingParams,
        on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let result = self.run_inner(tokens, max_tokens, stop, sampling, on_piece);
        if result.is_err() {
            // KV state may be partially written; start clean next time.
            self.reset();
        }
        self.cancel.reset();
        result
    }

    fn run_inner(
        &mut self,
        tokens: &[sys::llama_token],
        max_tokens: Option<u32>,
        stop: &[String],
        sampling: &SamplingParams,
        mut on_piece: impl FnMut(&str) -> bool,
    ) -> Result<ChatResponse> {
        let n_ctx = self.n_ctx();
        if tokens.is_empty() {
            return Err(Error::InvalidArgument("empty prompt".into()));
        }
        if tokens.len() >= n_ctx {
            return Err(Error::ContextOverflow { needed: tokens.len() + 1, n_ctx });
        }
        let mut sampler = Sampler::new(&self.model, sampling)?;
        let mut usage = Usage { prompt_tokens: tokens.len() as u32, ..Default::default() };
        let t_prompt = Instant::now();

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
        usage.prompt_ms = t_prompt.elapsed().as_secs_f64() * 1000.0;

        let t_gen = Instant::now();
        let budget = (n_ctx - self.cached.len()).min(max_tokens.map_or(usize::MAX, |m| m as usize));
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
            self.model.token_bytes(tok, &mut piece);
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
            self.batch.add(tok, self.cached.len() as i32, &[0], true);
            if !self.decode()? {
                self.reset();
                reason = FinishReason::Cancelled;
                break;
            }
            self.cached.push(tok);
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
        ChatResponse { content, finish_reason, usage }
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
