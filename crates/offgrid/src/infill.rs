//! Fill-in-the-middle (code infill) for models with FIM tokens, e.g. Qwen2.5-Coder.

use std::time::{Duration, Instant};

use offgrid_sys as sys;
use serde::{Deserialize, Deserializer, Serialize};

use crate::chat::{ChatSession, FinishReason, Prompt, string_or_seq};
use crate::model::FimTokens;
use crate::sampling::Sampler;
use crate::{ChatResponse, Error, Result, SamplingParams};

/// Prompt tokens for the prefix and suffix around the cursor (llama.cpp's default batch).
/// Extra context fills the rest of the context window.
const FIM_WINDOW: usize = 2048;
/// Generation room kept free when `max_tokens` is unset.
const DEFAULT_RESERVE: usize = 256;

/// A code infill request: generate the text between `input_prefix` and `input_suffix`.
/// Field names follow llama.cpp's `/infill`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InfillRequest {
    /// Text before the cursor.
    pub input_prefix: String,
    /// Text after the cursor.
    pub input_suffix: String,
    /// Other code for context (open files, recent edits), least relevant first: when the
    /// context is short, the earliest chunks are dropped.
    pub input_extra: Vec<InfillChunk>,
    /// Text appended to `input_prefix` (llama.vscode: the cursor line typed so far). Special
    /// tokens in it are parsed.
    pub prompt: String,
    /// Max tokens to generate. `None` (or a negative `n_predict`) = until end-of-generation.
    #[serde(alias = "n_predict", deserialize_with = "non_negative")]
    pub max_tokens: Option<u32>,
    #[serde(deserialize_with = "string_or_seq")]
    pub stop: Vec<String>,
    /// Stop before a line indented by fewer than this many spaces or tabs. 0 = off.
    pub n_indent: u32,
    /// After this many milliseconds of generation, stop at the next newline.
    pub t_max_predict_ms: Option<u64>,
    /// End the completion once end-of-generation is about as likely as the remaining text
    /// (llama.cpp's infill sampler). Default true.
    pub infill_sampler: bool,
    #[serde(flatten)]
    pub sampling: SamplingParams,
}

impl Default for InfillRequest {
    fn default() -> Self {
        Self {
            input_prefix: String::new(),
            input_suffix: String::new(),
            input_extra: Vec::new(),
            prompt: String::new(),
            max_tokens: None,
            stop: Vec::new(),
            n_indent: 0,
            t_max_predict_ms: None,
            infill_sampler: true,
            sampling: SamplingParams::default(),
        }
    }
}

/// A piece of extra context: a file name and some of its text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InfillChunk {
    #[serde(default)]
    pub filename: Option<String>,
    pub text: String,
}

fn non_negative<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<u32>, D::Error> {
    Ok(Option::<i64>::deserialize(d)?.and_then(|n| u32::try_from(n).ok()))
}

impl ChatSession {
    /// Fill in the middle: `on_piece` receives streamed text; return `false` to cancel.
    /// The prompt follows llama.cpp's layout (repo-level format when the model has
    /// `FIM_REP`/`FIM_SEP` tokens), so requests that share their start reuse the KV cache.
    pub fn infill(&mut self, req: &InfillRequest, mut on_piece: impl FnMut(&str) -> bool) -> Result<ChatResponse> {
        self.cancel_token().begin();
        let model = self.model().clone();
        let fim = model.fim_tokens().ok_or_else(|| {
            Error::InvalidArgument("this model has no fill-in-the-middle tokens (use a code model)".into())
        })?;
        let tok = |text: &str| model.tokenize(text, false, false);
        let mut extra = Vec::new();
        if let Some(rep) = fim.rep {
            extra.push(rep);
            extra.extend(tok("myproject\n")?);
        }
        for chunk in &req.input_extra {
            match fim.sep {
                Some(sep) => {
                    extra.push(sep);
                    extra.extend(tok(&format!("{}\n", chunk.filename.as_deref().unwrap_or("tmp")))?);
                }
                None => extra.extend(tok("\n\n--- snippet ---\n\n")?),
            }
            extra.extend(tok(&chunk.text)?);
        }
        if let Some(sep) = fim.sep {
            extra.push(sep);
            extra.extend(tok("filename\n")?);
        }
        let parts = Parts {
            bos: model.add_bos().then(|| model.bos()),
            extra,
            prefix: tok(&req.input_prefix)?,
            prompt: model.tokenize(&req.prompt, false, true)?,
            suffix: tok(&req.input_suffix)?,
        };
        let reserve = req.max_tokens.map_or(DEFAULT_RESERVE, |m| m as usize);
        let tokens = layout(&fim, parts, self.n_ctx(), reserve);

        let mut limits = Limits::new(req.n_indent as usize, req.t_max_predict_ms.map(Duration::from_millis));
        let mut content = String::new();
        let mut limited = false;
        let mut cancelled = false;
        let sampler = Sampler::new(&model, &req.sampling, None, req.infill_sampler)?;
        let result = self.run(Prompt::Tokens(&tokens), req.max_tokens, &req.stop, sampler, false, |piece| {
            let (text, stop) = limits.push(piece);
            content.push_str(&text);
            if !text.is_empty() && !on_piece(&text) {
                cancelled = true;
                return false;
            }
            limited = stop;
            !stop
        });
        let mut resp = result?;
        if limited {
            resp.finish_reason = FinishReason::Length;
        } else if !cancelled && resp.finish_reason != FinishReason::Cancelled {
            // Whitespace held back at the end of the output belongs to it.
            let tail = limits.flush();
            if !tail.is_empty() {
                content.push_str(&tail);
                on_piece(&tail);
            }
        }
        resp.content = content;
        Ok(resp)
    }
}

/// Tokenized prompt parts, before truncation.
struct Parts {
    bos: Option<sys::llama_token>,
    extra: Vec<sys::llama_token>,
    prefix: Vec<sys::llama_token>,
    prompt: Vec<sys::llama_token>,
    suffix: Vec<sys::llama_token>,
}

/// `[BOS] extra [PRE] prefix prompt [SUF] suffix [MID]`, cut to fit `n_ctx` with `reserve`
/// tokens left for generation. The prefix (its end) gets 3/4 of the window around the
/// cursor and the suffix (its start) the rest; extra context (its end) fills what remains.
fn layout(fim: &FimTokens, mut p: Parts, n_ctx: usize, reserve: usize) -> Vec<sys::llama_token> {
    let reserve = reserve.min(n_ctx / 4);
    let window = FIM_WINDOW.min(n_ctx / 2);
    let n_prefix = p.prefix.len().min(3 * window / 4);
    let n_suffix = p.suffix.len().min((window / 4).saturating_sub(2 + p.prompt.len()));
    let n_extra = p.extra.len().min(n_ctx.saturating_sub(window + reserve + 4));
    p.prefix.drain(..p.prefix.len() - n_prefix);
    p.suffix.truncate(n_suffix);

    let mut out = Vec::with_capacity(n_extra + n_prefix + n_suffix + p.prompt.len() + 4);
    out.extend(p.bos);
    out.extend_from_slice(&p.extra[p.extra.len() - n_extra..]);
    out.push(fim.pre);
    out.extend(p.prefix);
    out.extend(p.prompt);
    out.push(fim.suf);
    out.extend(p.suffix);
    out.push(fim.mid);
    out
}

/// Stops generation at a line indented less than `n_indent`, or at a newline once
/// `t_max` has passed. Whitespace after a newline is held back until the line's
/// indentation is known, so a cut line never shows up in the streamed text.
struct Limits {
    n_indent: usize,
    t_max: Option<Duration>,
    start: Option<Instant>,
    /// A newline and the whitespace after it, held back (only with `n_indent`).
    pending: String,
}

impl Limits {
    fn new(n_indent: usize, t_max: Option<Duration>) -> Self {
        Self { n_indent, t_max, start: None, pending: String::new() }
    }

    /// Text to deliver, and whether to stop after it.
    fn push(&mut self, piece: &str) -> (String, bool) {
        let start = *self.start.get_or_insert_with(Instant::now);
        let mut out = String::new();
        for ch in piece.chars() {
            if ch == '\n' {
                if self.t_max.is_some_and(|t| start.elapsed() > t) {
                    return (out, true);
                }
                if self.n_indent > 0 {
                    out.push_str(&std::mem::take(&mut self.pending)); // a blank line is fine
                    self.pending.push('\n');
                } else {
                    out.push('\n');
                }
            } else if !self.pending.is_empty() && (ch == ' ' || ch == '\t') {
                self.pending.push(ch);
            } else if !self.pending.is_empty() {
                if self.pending.len() - 1 < self.n_indent {
                    self.pending.clear();
                    return (out, true);
                }
                out.push_str(&std::mem::take(&mut self.pending));
                out.push(ch);
            } else {
                out.push(ch);
            }
        }
        (out, false)
    }

    /// Held-back text, at the end of generation.
    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIM: FimTokens = FimTokens { pre: -2, suf: -3, mid: -4, rep: None, sep: None };

    fn parts(extra: usize, prefix: usize, suffix: usize) -> Parts {
        Parts {
            bos: Some(-1),
            extra: (1000..1000 + extra as i32).collect(),
            prefix: (2000..2000 + prefix as i32).collect(),
            prompt: vec![3000],
            suffix: (4000..4000 + suffix as i32).collect(),
        }
    }

    #[test]
    fn layout_keeps_everything_that_fits() {
        let out = layout(&FIM, parts(2, 3, 2), 4096, 128);
        assert_eq!(out, vec![-1, 1000, 1001, -2, 2000, 2001, 2002, 3000, -3, 4000, 4001, -4]);
    }

    #[test]
    fn layout_keeps_text_nearest_the_cursor() {
        // n_ctx 1024: window 512 → prefix 384, suffix 128 - 3; extra 1024 - 512 - 128 - 4.
        let out = layout(&FIM, parts(1000, 1000, 1000), 1024, 128);
        let pre = out.iter().position(|&t| t == -2).unwrap();
        let suf = out.iter().position(|&t| t == -3).unwrap();
        assert_eq!(pre - 1, 380, "extra tokens");
        assert_eq!(out[pre - 1], 1999, "the end of the extra context is kept");
        assert_eq!(out[pre + 1], 2000 + 1000 - 384, "the end of the prefix is kept");
        assert_eq!(suf - pre - 2, 384, "prefix tokens before the prompt");
        assert_eq!(out[suf + 1], 4000, "the start of the suffix is kept");
        assert_eq!(out.len() - suf - 2, 125, "suffix tokens");
        assert!(out.len() + 128 <= 1024);
    }

    fn run(n_indent: usize, pieces: &[&str]) -> (String, bool) {
        let mut l = Limits::new(n_indent, None);
        let mut out = String::new();
        for p in pieces {
            let (text, stop) = l.push(p);
            out.push_str(&text);
            if stop {
                return (out, true);
            }
        }
        out.push_str(&l.flush());
        (out, false)
    }

    #[test]
    fn indent_limit_cuts_before_a_dedented_line() {
        assert_eq!(
            run(4, &["x = 1\n", "    y = 2\n", "\n", "    z", " = 3\n", "w"]).0,
            "x = 1\n    y = 2\n\n    z = 3"
        );
        assert_eq!(run(4, &["a\n  ", "  b\n  c"]), ("a\n    b".into(), true));
        assert_eq!(run(4, &["a\n", "\t\t\t\tb"]), ("a\n\t\t\t\tb".into(), false));
        assert_eq!(run(0, &["a\nb\n"]), ("a\nb\n".into(), false));
        assert_eq!(run(2, &["a\n  "]), ("a\n  ".into(), false), "trailing whitespace is flushed");
    }

    #[test]
    fn time_limit_stops_at_a_newline() {
        let mut l = Limits::new(0, Some(Duration::ZERO));
        l.push("");
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(l.push("abc\ndef"), ("abc".into(), true));
    }

    #[test]
    fn request_accepts_llama_cpp_fields() {
        let req: InfillRequest = serde_json::from_value(serde_json::json!({
            "input_prefix": "def f():\n", "input_suffix": "\n", "prompt": "    ",
            "input_extra": [{"filename": "a.py", "text": "x = 1"}, {"text": "y"}],
            "n_predict": -1, "n_indent": 4, "t_max_predict_ms": 500, "top_k": 40, "samplers": ["top_k"]
        }))
        .unwrap();
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.input_extra[1], InfillChunk { filename: None, text: "y".into() });
        assert_eq!((req.n_indent, req.t_max_predict_ms, req.sampling.top_k), (4, Some(500), 40));
        assert!(req.infill_sampler, "on by default");
        let req: InfillRequest = serde_json::from_value(serde_json::json!({"max_tokens": 16, "stop": "\n"})).unwrap();
        assert_eq!((req.max_tokens, req.stop), (Some(16), vec!["\n".to_string()]));
    }
}
