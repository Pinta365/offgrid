//! Image input via llama.cpp's mtmd (multimodal projector, the model's `mmproj` file).

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use aicore_sys as sys;

use crate::backend::setup_lock;
use crate::{Error, Result};

/// An encoded image (PNG, JPEG, BMP, GIF, ... anything stb_image decodes).
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image({} bytes)", self.data.len())
    }
}

impl Image {
    pub fn from_bytes(data: impl Into<Vec<u8>>) -> Self {
        Self { data: data.into() }
    }

    pub fn from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self { data: std::fs::read(path)? })
    }

    /// Parse a `data:image/...;base64,...` URL (the OpenAI `image_url` form).
    /// Remote URLs are rejected: this is an offline core.
    pub fn from_data_url(url: &str) -> Result<Self> {
        let rest = url.strip_prefix("data:").ok_or_else(|| {
            Error::InvalidArgument("image_url must be a data: URL (remote images are not fetched)".into())
        })?;
        let (meta, payload) =
            rest.split_once(',').ok_or_else(|| Error::InvalidArgument("malformed data: URL".into()))?;
        if !meta.ends_with(";base64") {
            return Err(Error::InvalidArgument("data: URL must be base64-encoded".into()));
        }
        let data = base64_decode(payload).ok_or_else(|| Error::InvalidArgument("invalid base64 in data: URL".into()))?;
        Ok(Self { data })
    }

    /// `data:` URL for this image (mime type guessed from magic bytes).
    pub fn to_data_url(&self) -> String {
        let mime = match self.data.as_slice() {
            [0x89, b'P', b'N', b'G', ..] => "image/png",
            [0xFF, 0xD8, ..] => "image/jpeg",
            [b'G', b'I', b'F', ..] => "image/gif",
            [b'B', b'M', ..] => "image/bmp",
            _ => "application/octet-stream",
        };
        format!("data:{mime};base64,{}", base64_encode(&self.data))
    }
}

/// The text marker standing for one image in a prompt (mtmd's default `<__media__>`).
pub fn media_marker() -> &'static str {
    static MARKER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MARKER.get_or_init(|| unsafe { CStr::from_ptr(sys::mtmd_default_marker()) }.to_string_lossy().into_owned())
}

/// A loaded projector bound to its text model.
pub(crate) struct Vision {
    ctx: NonNull<sys::mtmd_context>,
    /// Encoding/eval is not thread-safe; tokenizing is.
    eval_lock: Mutex<()>,
}

// mtmd_context is used from several sessions; mutating calls are serialized by eval_lock.
unsafe impl Send for Vision {}
unsafe impl Sync for Vision {}

impl Drop for Vision {
    fn drop(&mut self) {
        let _guard = setup_lock();
        unsafe { sys::mtmd_free(self.ctx.as_ptr()) };
    }
}

impl Vision {
    pub fn load(path: &Path, model: *const sys::llama_model, use_gpu: bool, n_threads: i32) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::ModelNotFound(path.to_path_buf()));
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidArgument("mmproj path contains NUL".into()))?;
        let mut params = unsafe { sys::mtmd_context_params_default() };
        params.use_gpu = use_gpu;
        params.n_threads = n_threads;
        params.print_timings = false;
        let ctx = {
            let _guard = setup_lock();
            unsafe { sys::mtmd_init_from_file(c_path.as_ptr(), model, params) }
        };
        let ctx = NonNull::new(ctx).ok_or_else(|| Error::ModelLoad(path.to_path_buf()))?;
        if !unsafe { sys::mtmd_support_vision(ctx.as_ptr()) } {
            unsafe { sys::mtmd_free(ctx.as_ptr()) };
            return Err(Error::InvalidArgument(format!("{} is not a vision projector", path.display())));
        }
        Ok(Self { ctx, eval_lock: Mutex::new(()) })
    }

    /// Tokenize a prompt containing one media marker per image.
    pub fn tokenize(&self, prompt: &str, images: &[&Image], add_bos: bool) -> Result<Chunks> {
        let mut bitmaps = Vec::with_capacity(images.len());
        for (i, img) in images.iter().enumerate() {
            let opt = unsafe { sys::mtmd_helper_init_opt_default() };
            let w = unsafe {
                sys::mtmd_helper_bitmap_init_from_buf(self.ctx.as_ptr(), img.data.as_ptr(), img.data.len(), false, opt)
            };
            let bitmap = NonNull::new(w.bitmap).ok_or_else(|| Error::Image(format!("image {i} could not be decoded")))?;
            bitmaps.push(Bitmap(bitmap));
        }
        let text = CString::new(prompt).map_err(|_| Error::InvalidArgument("prompt contains NUL".into()))?;
        let input = sys::mtmd_input_text { text: text.as_ptr(), text_len: prompt.len(), add_special: add_bos, parse_special: true };
        let chunks = Chunks(NonNull::new(unsafe { sys::mtmd_input_chunks_init() }).ok_or(Error::Tokenize)?);
        let ptrs: Vec<*const sys::mtmd_bitmap> = bitmaps.iter().map(|b| b.0.as_ptr() as *const _).collect();
        match unsafe { sys::mtmd_tokenize(self.ctx.as_ptr(), chunks.0.as_ptr(), &input, ptrs.as_ptr() as *mut _, ptrs.len()) } {
            0 => Ok(chunks),
            1 => Err(Error::Image(format!(
                "{} image(s) but the prompt has a different number of image markers",
                images.len()
            ))),
            2 => Err(Error::Image("image preprocessing failed".into())),
            code => Err(Error::Image(format!("mtmd_tokenize failed ({code})"))),
        }
    }

    /// Encode images and decode all chunks into `lctx` starting at `n_past`; returns the
    /// new position. Logits are produced for the last token.
    pub fn eval(&self, lctx: *mut sys::llama_context, chunks: &Chunks, n_past: i32, n_batch: i32) -> Result<i32> {
        let _guard = self.eval_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut new_n_past = 0;
        let rc = unsafe {
            sys::mtmd_helper_eval_chunks(self.ctx.as_ptr(), lctx, chunks.0.as_ptr(), n_past, 0, n_batch, true, &mut new_n_past)
        };
        if rc != 0 {
            return Err(Error::Decode(rc));
        }
        Ok(new_n_past)
    }
}

struct Bitmap(NonNull<sys::mtmd_bitmap>);

impl Drop for Bitmap {
    fn drop(&mut self) {
        unsafe { sys::mtmd_bitmap_free(self.0.as_ptr()) };
    }
}

/// Tokenized multimodal prompt.
pub(crate) struct Chunks(NonNull<sys::mtmd_input_chunks>);

impl Chunks {
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_helper_get_n_tokens(self.0.as_ptr()) }
    }
    /// Positions consumed (differs from tokens for M-RoPE models).
    pub fn n_pos(&self) -> usize {
        unsafe { sys::mtmd_helper_get_n_pos(self.0.as_ptr()) }.max(0) as usize
    }
}

impl Drop for Chunks {
    fn drop(&mut self) {
        unsafe { sys::mtmd_input_chunks_free(self.0.as_ptr()) };
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        acc = acc << 6 | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        for data in [&b""[..], b"a", b"ab", b"abc", b"abcd", &[0xFF, 0x00, 0x10, 0x80, 0x7F]] {
            assert_eq!(base64_decode(&base64_encode(data)).unwrap(), data);
        }
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
    }

    #[test]
    fn data_urls() {
        let png = Image::from_bytes(vec![0x89, b'P', b'N', b'G', 1, 2, 3]);
        let url = png.to_data_url();
        assert!(url.starts_with("data:image/png;base64,"));
        assert_eq!(Image::from_data_url(&url).unwrap(), png);
        assert!(Image::from_data_url("https://example.com/cat.png").is_err());
        assert!(Image::from_data_url("data:image/png,rawbytes").is_err());
        assert!(Image::from_data_url("data:image/png;base64,***").is_err());
    }
}
