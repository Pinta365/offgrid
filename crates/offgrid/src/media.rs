//! Image and audio input via llama.cpp's mtmd (multimodal projector, the model's `mmproj` file).

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use offgrid_sys as sys;

use crate::backend::setup_lock;
use crate::{Error, Result};

/// What a [`Media`] item is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    /// PNG, JPEG, BMP, GIF, ... (anything stb_image decodes).
    Image,
    /// WAV, MP3 or FLAC (decoded by miniaudio, resampled for the model).
    Audio,
}

/// An encoded image or audio clip attached to a message.
#[derive(Clone, PartialEq, Eq)]
pub struct Media {
    pub kind: MediaKind,
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Media {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}({} bytes)", self.kind, self.data.len())
    }
}

impl Media {
    pub fn image(data: impl Into<Vec<u8>>) -> Self {
        Self { kind: MediaKind::Image, data: data.into() }
    }

    pub fn audio(data: impl Into<Vec<u8>>) -> Self {
        Self { kind: MediaKind::Audio, data: data.into() }
    }

    /// Read a file; the kind is detected from its magic bytes (WAV/MP3/FLAC = audio).
    pub fn from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let kind = if audio_format(&data).is_some() { MediaKind::Audio } else { MediaKind::Image };
        Ok(Self { kind, data })
    }

    /// Parse an image `data:image/...;base64,...` URL (the OpenAI `image_url` form).
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
        let data =
            base64_decode(payload).ok_or_else(|| Error::InvalidArgument("invalid base64 in data: URL".into()))?;
        Ok(Self::image(data))
    }

    /// Parse an OpenAI `input_audio` part: base64 `data` in `format` (`wav`, `mp3`, `flac`).
    /// The data must actually be in the declared format.
    pub fn from_input_audio(data: &str, format: &str) -> Result<Self> {
        if !matches!(format, "wav" | "mp3" | "flac") {
            return Err(Error::InvalidArgument(format!("unsupported audio format '{format}' (wav, mp3, flac)")));
        }
        let bytes =
            base64_decode(data).ok_or_else(|| Error::InvalidArgument("invalid base64 in input_audio".into()))?;
        match audio_format(&bytes) {
            Some(detected) if detected == format => Ok(Self::audio(bytes)),
            Some(detected) => {
                Err(Error::InvalidArgument(format!("input_audio declared '{format}' but the data is {detected}")))
            }
            None => Err(Error::InvalidArgument(format!("input_audio data is not {format}"))),
        }
    }

    /// Image as a `data:` URL (mime type guessed from magic bytes).
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

    /// Audio as OpenAI `input_audio` fields: (base64 data, format).
    pub fn to_input_audio(&self) -> (String, &'static str) {
        (base64_encode(&self.data), audio_format(&self.data).unwrap_or("wav"))
    }
}

/// Audio container from magic bytes (same rules as mtmd's `is_audio_file`, so our
/// declared kind agrees with how mtmd decodes the data).
fn audio_format(data: &[u8]) -> Option<&'static str> {
    match data {
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'A', b'V', b'E', ..] => Some("wav"),
        [b'f', b'L', b'a', b'C', ..] => Some("flac"),
        // ID3 tag, or an MPEG frame sync (11 set bits: any MPEG version / layer / CRC flag).
        [b'I', b'D', b'3', ..] => Some("mp3"),
        [0xFF, b1, _, ..] if b1 & 0xE0 == 0xE0 => Some("mp3"),
        _ => None,
    }
}

/// The text marker standing for one media item in a prompt (mtmd's default `<__media__>`).
pub fn media_marker() -> &'static str {
    static MARKER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MARKER.get_or_init(|| unsafe { CStr::from_ptr(sys::mtmd_default_marker()) }.to_string_lossy().into_owned())
}

/// Shape of a decoded media item: enough to tokenize it again as a placeholder (no
/// decoding) while its encoding is still in the KV cache.
#[derive(Debug, Clone)]
struct MediaShape {
    audio: bool,
    /// Image width / audio sample count (after resampling).
    nx: u32,
    ny: u32,
    /// mtmd's content hash (SHA-256), the chunk id used for KV-cache identity.
    id: String,
}

/// Per-session memo of media shapes, keyed by a randomly seeded hash of the bytes.
#[derive(Default)]
pub(crate) struct MediaMemo {
    hasher: std::collections::hash_map::RandomState,
    shapes: std::collections::HashMap<(u64, usize), MediaShape>,
}

impl MediaMemo {
    fn key(&self, data: &[u8]) -> (u64, usize) {
        use std::hash::BuildHasher;
        (self.hasher.hash_one(data), data.len())
    }

    /// Keep only shapes whose content hash satisfies `keep` (e.g. still in the KV cache).
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) {
        self.shapes.retain(|_, s| keep(&s.id));
    }
}

/// A loaded multimodal projector (`mmproj`) bound to its text model.
pub(crate) struct Projector {
    ctx: NonNull<sys::mtmd_context>,
    pub vision: bool,
    pub audio: bool,
    /// Encoding/eval is not thread-safe; tokenizing is.
    eval_lock: Mutex<()>,
}

// mtmd_context is used from several sessions; mutating calls are serialized by eval_lock.
unsafe impl Send for Projector {}
unsafe impl Sync for Projector {}

impl Drop for Projector {
    fn drop(&mut self) {
        let _guard = setup_lock();
        unsafe { sys::mtmd_free(self.ctx.as_ptr()) };
    }
}

/// Headroom for the projector's compute buffer on top of its weights.
const PROJECTOR_GPU_MARGIN: u64 = 512 << 20;

/// Whether the projector fits in free memory on the GPU clip.cpp will pick; clip.cpp aborts
/// instead of failing when it doesn't.
fn projector_fits_gpu(path: &Path, device: sys::ggml_backend_dev_t) -> bool {
    let Ok(size) = std::fs::metadata(path).map(|m| m.len()) else { return true };
    let dev = {
        let _guard = setup_lock();
        // Same choice as clip.cpp: the requested device, else the first GPU, else the first iGPU.
        let find = |ty| {
            (0..unsafe { sys::ggml_backend_dev_count() })
                .map(|i| unsafe { sys::ggml_backend_dev_get(i) })
                .find(|&d| !d.is_null() && unsafe { sys::ggml_backend_dev_type(d) } == ty)
        };
        if device.is_null() {
            find(sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU)
                .or_else(|| find(sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU))
        } else {
            Some(device)
        }
    };
    let Some(dev) = dev else { return true };
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { sys::ggml_backend_dev_memory(dev, &mut free, &mut total) };
    let need = size + PROJECTOR_GPU_MARGIN;
    if free == 0 || need <= free as u64 {
        return true;
    }
    log::warn!(
        target: "offgrid",
        "{} needs ~{} MiB but {} has {} MiB free; loading the projector on the CPU",
        path.display(),
        need >> 20,
        unsafe { CStr::from_ptr(sys::ggml_backend_dev_name(dev)) }.to_string_lossy(),
        free >> 20
    );
    false
}

impl Projector {
    pub fn load(path: &Path, model: *const sys::llama_model, use_gpu: bool, n_threads: i32) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::ModelNotFound(path.to_path_buf()));
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidArgument("mmproj path contains NUL".into()))?;
        let mut params = unsafe { sys::mtmd_context_params_default() };
        params.use_gpu = use_gpu && projector_fits_gpu(path, params.device);
        params.n_threads = n_threads;
        params.print_timings = false;
        let ctx = {
            let _guard = setup_lock();
            unsafe { sys::mtmd_init_from_file(c_path.as_ptr(), model, params) }
        };
        let ctx = NonNull::new(ctx).ok_or_else(|| Error::ModelLoad(path.to_path_buf()))?;
        let (vision, audio) =
            unsafe { (sys::mtmd_support_vision(ctx.as_ptr()), sys::mtmd_support_audio(ctx.as_ptr())) };
        if !vision && !audio {
            unsafe { sys::mtmd_free(ctx.as_ptr()) };
            return Err(Error::InvalidArgument(format!("{} supports neither images nor audio", path.display())));
        }
        Ok(Self { ctx, vision, audio, eval_lock: Mutex::new(()) })
    }

    /// Tokenize a prompt with one media marker per item. Items in `memo` that are still
    /// `reusable` (in the KV cache) become placeholders, skipping the decode.
    pub fn tokenize(
        &self,
        prompt: &str,
        media: &[&Media],
        add_bos: bool,
        memo: &mut MediaMemo,
        reusable: &dyn Fn(&str) -> bool,
    ) -> Result<Chunks> {
        let mut bitmaps = Vec::with_capacity(media.len());
        let mut placeholders = std::collections::HashSet::new();
        for (i, item) in media.iter().enumerate() {
            match item.kind {
                MediaKind::Image if !self.vision => {
                    return Err(Error::InvalidArgument("this model's projector does not accept images".into()));
                }
                MediaKind::Audio if !self.audio => {
                    return Err(Error::InvalidArgument("this model's projector does not accept audio".into()));
                }
                _ => {}
            }
            let key = memo.key(&item.data);
            if let Some(shape) = memo.shapes.get(&key)
                && shape.audio == (item.kind == MediaKind::Audio)
                && reusable(&shape.id)
            {
                let raw = unsafe {
                    if shape.audio {
                        sys::mtmd_bitmap_init_from_audio(shape.nx as usize, std::ptr::null())
                    } else {
                        sys::mtmd_bitmap_init(shape.nx, shape.ny, std::ptr::null())
                    }
                };
                let bitmap = Bitmap(NonNull::new(raw).ok_or_else(|| Error::Media("placeholder bitmap".into()))?);
                let id = CString::new(shape.id.as_str()).expect("hex id");
                unsafe { sys::mtmd_bitmap_set_id(bitmap.0.as_ptr(), id.as_ptr()) };
                placeholders.insert(shape.id.clone());
                bitmaps.push(bitmap);
                continue;
            }
            let opt = unsafe { sys::mtmd_helper_init_opt_default() };
            let w = unsafe {
                sys::mtmd_helper_bitmap_init_from_buf(
                    self.ctx.as_ptr(),
                    item.data.as_ptr(),
                    item.data.len(),
                    false,
                    opt,
                )
            };
            let what = match item.kind {
                MediaKind::Image => "image",
                MediaKind::Audio => "audio",
            };
            let bitmap =
                NonNull::new(w.bitmap).ok_or_else(|| Error::Media(format!("{what} {i} could not be decoded")))?;
            let bitmap = Bitmap(bitmap);
            // mtmd detects the real type from the bytes; it must match what was declared.
            if unsafe { sys::mtmd_bitmap_is_audio(bitmap.0.as_ptr()) } != (item.kind == MediaKind::Audio) {
                return Err(Error::Media(format!("media {i} was sent as {what} but its data is not {what}")));
            }
            let id_ptr = unsafe { sys::mtmd_bitmap_get_id(bitmap.0.as_ptr()) };
            if !id_ptr.is_null() {
                let shape = unsafe {
                    MediaShape {
                        audio: item.kind == MediaKind::Audio,
                        nx: sys::mtmd_bitmap_get_nx(bitmap.0.as_ptr()),
                        ny: sys::mtmd_bitmap_get_ny(bitmap.0.as_ptr()),
                        id: CStr::from_ptr(id_ptr).to_string_lossy().into_owned(),
                    }
                };
                memo.shapes.insert(key, shape);
            }
            bitmaps.push(bitmap);
        }
        let text = CString::new(prompt).map_err(|_| Error::InvalidArgument("prompt contains NUL".into()))?;
        let input = sys::mtmd_input_text {
            text: text.as_ptr(),
            text_len: prompt.len(),
            add_special: add_bos,
            parse_special: true,
        };
        let chunks =
            Chunks(NonNull::new(unsafe { sys::mtmd_input_chunks_init() }).ok_or(Error::Tokenize)?, placeholders);
        let ptrs: Vec<*const sys::mtmd_bitmap> = bitmaps.iter().map(|b| b.0.as_ptr() as *const _).collect();
        match unsafe {
            sys::mtmd_tokenize(self.ctx.as_ptr(), chunks.0.as_ptr(), &input, ptrs.as_ptr() as *mut _, ptrs.len())
        } {
            0 => Ok(chunks),
            1 => Err(Error::Media(format!(
                "{} media item(s) but the prompt has a different number of media markers",
                media.len()
            ))),
            2 => Err(Error::Media("media preprocessing failed".into())),
            code => Err(Error::Media(format!("mtmd_tokenize failed ({code})"))),
        }
    }

    /// Encode one media chunk (image or audio) and decode it into `lctx` at `n_past`; returns the new position.
    pub fn eval_chunk(
        &self,
        lctx: *mut sys::llama_context,
        chunk: &MediaChunk,
        n_past: i32,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<i32> {
        let _guard = self.eval_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut new_n_past = 0;
        let rc = unsafe {
            sys::mtmd_helper_eval_chunk_single(
                self.ctx.as_ptr(),
                lctx,
                chunk.ptr,
                n_past,
                0,
                n_batch,
                logits_last,
                &mut new_n_past,
            )
        };
        if rc != 0 {
            return Err(Error::Decode(rc));
        }
        Ok(new_n_past)
    }
}

/// One piece of a tokenized prompt: text tokens, or a media chunk (an image or audio clip).
pub(crate) enum Segment {
    Text(Vec<sys::llama_token>),
    Media(MediaChunk),
}

/// A media chunk inside a [`Chunks`] (valid while the `Chunks` lives).
pub(crate) struct MediaChunk {
    ptr: *const sys::mtmd_input_chunk,
    /// Content hash of the media (SHA-256, set by mtmd) + `#k` chunk index; used for
    /// KV-cache reuse.
    pub id: String,
    /// Tokenized from a placeholder (no data): reusable from the KV cache, not evaluable.
    pub placeholder: bool,
    pub n_tokens: usize,
    /// Positions consumed (differs from tokens for M-RoPE models).
    pub n_pos: usize,
}

struct Bitmap(NonNull<sys::mtmd_bitmap>);

impl Drop for Bitmap {
    fn drop(&mut self) {
        unsafe { sys::mtmd_bitmap_free(self.0.as_ptr()) };
    }
}

/// Tokenized multimodal prompt (+ the content hashes that were tokenized as placeholders).
pub(crate) struct Chunks(NonNull<sys::mtmd_input_chunks>, std::collections::HashSet<String>);

impl Chunks {
    pub fn segments(&self) -> Vec<Segment> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static UNHASHED: AtomicU64 = AtomicU64::new(0);
        // mtmd splits long audio into several chunks that share the clip's hash; number
        // repeated ids so each chunk has a distinct cache identity (deterministic per prompt).
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let n = unsafe { sys::mtmd_input_chunks_size(self.0.as_ptr()) };
        (0..n)
            .map(|i| unsafe {
                let chunk = sys::mtmd_input_chunks_get(self.0.as_ptr(), i);
                if sys::mtmd_input_chunk_get_type(chunk) == sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT {
                    let mut len = 0usize;
                    let toks = sys::mtmd_input_chunk_get_tokens_text(chunk, &mut len);
                    Segment::Text(if toks.is_null() {
                        Vec::new()
                    } else {
                        std::slice::from_raw_parts(toks, len).to_vec()
                    })
                } else {
                    let id_ptr = sys::mtmd_input_chunk_get_id(chunk);
                    let placeholder =
                        !id_ptr.is_null() && self.1.contains(CStr::from_ptr(id_ptr).to_string_lossy().as_ref());
                    let id = if id_ptr.is_null() {
                        // No content hash: never matches a cached entry.
                        format!("unhashed-{}", UNHASHED.fetch_add(1, Ordering::Relaxed))
                    } else {
                        let hash = CStr::from_ptr(id_ptr).to_string_lossy().into_owned();
                        let k = seen.entry(hash.clone()).or_insert(0);
                        *k += 1;
                        format!("{hash}#{k}")
                    };
                    Segment::Media(MediaChunk {
                        ptr: chunk,
                        id,
                        placeholder,
                        n_tokens: sys::mtmd_input_chunk_get_n_tokens(chunk),
                        n_pos: sys::mtmd_input_chunk_get_n_pos(chunk).max(0) as usize,
                    })
                }
            })
            .collect()
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
        let png = Media::image(vec![0x89, b'P', b'N', b'G', 1, 2, 3]);
        let url = png.to_data_url();
        assert!(url.starts_with("data:image/png;base64,"));
        assert_eq!(Media::from_data_url(&url).unwrap(), png);
        assert!(Media::from_data_url("https://example.com/cat.png").is_err());
        assert!(Media::from_data_url("data:image/png,rawbytes").is_err());
        assert!(Media::from_data_url("data:image/png;base64,***").is_err());
    }

    #[test]
    fn input_audio() {
        let wav = Media::audio(b"RIFF\0\0\0\0WAVEfmt ".to_vec());
        let (b64, format) = wav.to_input_audio();
        assert_eq!(format, "wav");
        assert_eq!(Media::from_input_audio(&b64, "wav").unwrap(), wav);
        assert!(Media::from_input_audio(&b64, "ogg").is_err());
        assert_eq!(audio_format(b"fLaC...."), Some("flac"));
        assert_eq!(audio_format(b"ID3\x04"), Some("mp3"));
        assert_eq!(audio_format(&[0x89, b'P', b'N', b'G']), None);
        // Frame syncs mtmd accepts: MPEG-1 L3 with/without CRC, MPEG-2, MPEG-2.5.
        for b1 in [0xFB, 0xFA, 0xF3, 0xF2, 0xE3] {
            assert_eq!(audio_format(&[0xFF, b1, 0x90, 0x00]), Some("mp3"), "{b1:#x}");
        }
        assert_eq!(audio_format(&[0xFF, 0xD8, 0xFF, 0xE0]), None, "JPEG is not mp3");
        // Declared format must match the data.
        assert!(Media::from_input_audio(&b64, "mp3").is_err());
        assert!(Media::from_input_audio(&base64_encode(b"not audio at all"), "wav").is_err());
    }
}
