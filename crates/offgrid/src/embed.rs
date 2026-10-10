use std::ptr::NonNull;
use std::sync::Arc;

use offgrid_sys as sys;
use serde::{Deserialize, Serialize};

use crate::batch::Batch;
use crate::chat::default_threads;
use crate::{Error, Model, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedParams {
    /// Max tokens per input (and per batch). 0 = the model's training context, capped at 8192.
    pub n_ctx: u32,
    /// Max inputs embedded together in one batch.
    pub max_batch_inputs: u32,
    /// L2-normalize output vectors (so dot product = cosine similarity).
    pub normalize: bool,
    pub n_threads: Option<i32>,
}

impl Default for EmbedParams {
    fn default() -> Self {
        Self { n_ctx: 0, max_batch_inputs: 16, normalize: true, n_threads: None }
    }
}

/// Embedding context for a model with a pooling head (e.g. Qwen3-Embedding, nomic-embed).
pub struct Embedder {
    model: Arc<Model>,
    ctx: NonNull<sys::llama_context>,
    batch: Batch,
    n_batch: usize,
    max_inputs: usize,
    normalize: bool,
    dim: usize,
}

unsafe impl Send for Embedder {}

impl Drop for Embedder {
    fn drop(&mut self) {
        let _guard = crate::backend::setup_lock();
        unsafe { sys::llama_free(self.ctx.as_ptr()) };
    }
}

impl Embedder {
    pub fn new(model: Arc<Model>, params: &EmbedParams) -> Result<Self> {
        let n_ctx = match params.n_ctx {
            0 => model.info().n_ctx_train.clamp(512, 8192),
            n => n,
        };
        let max_inputs = params.max_batch_inputs.max(1);
        let threads = params.n_threads.unwrap_or_else(default_threads);

        // The graph computes vocab-sized logits per ubatch token (~0.6 MB each for Qwen3), so use
        // small ubatches where inputs may span several (causal, last-token pooling).
        let arch = model.meta("general.architecture").unwrap_or_default();
        let causal = model.meta(&format!("{arch}.attention.causal")).is_none_or(|v| v != "false");
        let last_pooling = model.meta(&format!("{arch}.pooling_type")).as_deref() == Some("3");
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.n_ubatch = if causal && last_pooling { n_ctx.min(512) } else { n_ctx };
        cp.n_seq_max = max_inputs;
        // Shared KV pool: a per-sequence split (n_ctx / n_seq_max) makes long inputs fail.
        cp.kv_unified = true;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.embeddings = true;
        cp.no_perf = true;

        let ctx = {
            let _guard = crate::backend::setup_lock();
            unsafe { sys::llama_init_from_model(model.as_ptr(), cp) }
        };
        let ctx = NonNull::new(ctx).ok_or(Error::ContextCreate)?;
        let this_pooling = unsafe { sys::llama_pooling_type(ctx.as_ptr()) };
        // NONE gives per-token vectors and RANK (rerankers) gives n_cls_out scores per
        // sequence, not n_embd floats: reading `dim` floats would overrun the buffer.
        if this_pooling == sys::llama_pooling_type_LLAMA_POOLING_TYPE_NONE
            || this_pooling == sys::llama_pooling_type_LLAMA_POOLING_TYPE_RANK
        {
            let _guard = crate::backend::setup_lock();
            unsafe { sys::llama_free(ctx.as_ptr()) };
            return Err(Error::NoEmbeddings);
        }
        let n_batch = unsafe { sys::llama_n_batch(ctx.as_ptr()) } as usize;
        let max_inputs = unsafe { sys::llama_n_seq_max(ctx.as_ptr()) } as usize;
        let m = model.as_ptr();
        let dim = match unsafe { sys::llama_model_n_embd_out(m) } {
            d if d > 0 => d as usize,
            _ => (unsafe { sys::llama_model_n_embd(m) }) as usize,
        };
        Ok(Self { model, ctx, batch: Batch::new(n_batch, 1), n_batch, max_inputs, normalize: params.normalize, dim })
    }

    /// Embedding vector length.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed(&[text])?.pop().expect("one input"))
    }

    /// Embed several inputs; returns one vector per input, in order.
    pub fn embed<S: AsRef<str>>(&mut self, inputs: &[S]) -> Result<Vec<Vec<f32>>> {
        let tokenized = inputs
            .iter()
            .map(|s| {
                let t = self.model.tokenize(s.as_ref(), true, false)?;
                if t.is_empty() {
                    return Err(Error::InvalidArgument("empty input".into()));
                }
                if t.len() > self.n_batch {
                    return Err(Error::ContextOverflow { needed: t.len(), n_ctx: self.n_batch });
                }
                Ok(t)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut out = Vec::with_capacity(inputs.len());
        let mut start = 0;
        while start < tokenized.len() {
            // Pack as many inputs as fit the batch.
            let mut end = start;
            let mut total = 0;
            while end < tokenized.len() && end - start < self.max_inputs && total + tokenized[end].len() <= self.n_batch
            {
                total += tokenized[end].len();
                end += 1;
            }
            self.run_batch(&tokenized[start..end], &mut out)?;
            start = end;
        }
        Ok(out)
    }

    fn run_batch(&mut self, group: &[Vec<sys::llama_token>], out: &mut Vec<Vec<f32>>) -> Result<()> {
        let ctx = self.ctx.as_ptr();
        let mem = unsafe { sys::llama_get_memory(ctx) };
        if !mem.is_null() {
            unsafe { sys::llama_memory_clear(mem, true) };
        }
        self.batch.clear();
        for (seq, tokens) in group.iter().enumerate() {
            for (pos, &t) in tokens.iter().enumerate() {
                self.batch.add(t, pos as i32, &[seq as i32], true);
            }
        }
        let m = self.model.as_ptr();
        let rc = unsafe {
            if sys::llama_model_has_encoder(m) && !sys::llama_model_has_decoder(m) {
                sys::llama_encode(ctx, self.batch.raw())
            } else {
                sys::llama_decode(ctx, self.batch.raw())
            }
        };
        if rc != 0 {
            return Err(Error::Decode(rc));
        }
        for seq in 0..group.len() {
            let ptr = unsafe { sys::llama_get_embeddings_seq(ctx, seq as i32) };
            if ptr.is_null() {
                return Err(Error::NoEmbeddings);
            }
            let mut v = unsafe { std::slice::from_raw_parts(ptr, self.dim) }.to_vec();
            if self.normalize {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
            }
            out.push(v);
        }
        Ok(())
    }
}
