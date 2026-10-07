use std::ffi::CString;
use std::ptr::NonNull;

use aicore_sys as sys;
use serde::{Deserialize, Serialize};

use crate::tools::GrammarSpec;
use crate::{Error, Model, Result};

/// Sampling settings. Field names follow the OpenAI / llama.cpp server conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    /// `<= 0` means greedy decoding.
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// `None` picks a random seed.
    pub seed: Option<u32>,
    /// GBNF grammar constraining the output (root rule `root`).
    pub grammar: Option<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            grammar: None,
        }
    }
}

/// Owned llama sampler chain.
pub(crate) struct Sampler {
    ptr: NonNull<sys::llama_sampler>,
}

impl Sampler {
    /// `grammar` overrides `p.grammar` (callers resolve response_format / tools into it).
    pub fn new(model: &Model, p: &SamplingParams, grammar: Option<&GrammarSpec>) -> Result<Self> {
        unsafe {
            let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
            let chain = NonNull::new(chain).ok_or(Error::InvalidArgument("sampler chain".into()))?;
            let this = Self { ptr: chain };
            let add = |s: *mut sys::llama_sampler| sys::llama_sampler_chain_add(chain.as_ptr(), s);

            let from_params = p.grammar.clone().map(GrammarSpec::strict);
            if let Some(spec) = grammar.or(from_params.as_ref()) {
                let text = CString::new(spec.text.as_str()).map_err(|_| Error::Grammar)?;
                let root = CString::new(spec.root.as_str()).map_err(|_| Error::Grammar)?;
                let s = if spec.lazy_patterns.is_empty() {
                    sys::llama_sampler_init_grammar(model.vocab(), text.as_ptr(), root.as_ptr())
                } else {
                    let pats: Vec<CString> = spec
                        .lazy_patterns
                        .iter()
                        .map(|p| CString::new(p.as_str()).map_err(|_| Error::Grammar))
                        .collect::<Result<_>>()?;
                    let ptrs: Vec<*const std::ffi::c_char> = pats.iter().map(|p| p.as_ptr()).collect();
                    sys::llama_sampler_init_grammar_lazy_patterns(
                        model.vocab(),
                        text.as_ptr(),
                        root.as_ptr(),
                        ptrs.as_ptr() as *mut _,
                        ptrs.len(),
                        std::ptr::null(),
                        0,
                    )
                };
                if s.is_null() {
                    log::debug!(target: "aicore", "rejected grammar:\n{}", spec.text);
                    return Err(Error::Grammar);
                }
                add(s);
            }
            if p.repeat_penalty != 1.0 || p.frequency_penalty != 0.0 || p.presence_penalty != 0.0 {
                add(sys::llama_sampler_init_penalties(
                    model.n_vocab(),
                    p.repeat_last_n,
                    p.repeat_penalty,
                    p.frequency_penalty,
                    p.presence_penalty,
                ));
            }
            if p.temperature <= 0.0 {
                add(sys::llama_sampler_init_greedy());
            } else {
                if p.top_k > 0 {
                    add(sys::llama_sampler_init_top_k(p.top_k));
                }
                if p.top_p < 1.0 {
                    add(sys::llama_sampler_init_top_p(p.top_p, 1));
                }
                if p.min_p > 0.0 {
                    add(sys::llama_sampler_init_min_p(p.min_p, 1));
                }
                add(sys::llama_sampler_init_temp(p.temperature));
                let seed = p.seed.unwrap_or(sys::LLAMA_DEFAULT_SEED);
                add(sys::llama_sampler_init_dist(seed));
            }
            Ok(this)
        }
    }

    /// Sample from the logits of output `idx` of the last decode, and accept the token.
    pub fn sample(&mut self, ctx: *mut sys::llama_context, idx: i32) -> sys::llama_token {
        // llama_sampler_sample also calls accept on the chain.
        unsafe { sys::llama_sampler_sample(self.ptr.as_ptr(), ctx, idx) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { sys::llama_sampler_free(self.ptr.as_ptr()) };
    }
}
