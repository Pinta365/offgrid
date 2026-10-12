//! Image generation through stable-diffusion.cpp (Z-Image, FLUX, SD 1.x/2.x/XL, ...).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::{Mutex, Once};

use offgrid_sys::sd;
use serde::{Deserialize, Serialize};

use crate::backend::{ensure_init, setup_lock};
use crate::chat::default_threads;
use crate::{Error, Result};

/// Model files for an image model: a single-file checkpoint (`model`), or a `diffusion_model` with
/// the encoders and VAE its family needs (Z-Image: `text_encoder` = Qwen3-4B, `vae`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageModelParams {
    /// Single-file checkpoint (SD 1.x/2.x/XL).
    pub model: Option<PathBuf>,
    /// Diffusion model alone (Z-Image, FLUX, ...).
    pub diffusion_model: Option<PathBuf>,
    /// LLM text encoder (Z-Image and Qwen-Image use a Qwen GGUF).
    pub text_encoder: Option<PathBuf>,
    pub clip_l: Option<PathBuf>,
    pub clip_g: Option<PathBuf>,
    pub t5xxl: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    /// Use the GPU when one is available; false runs on the CPU.
    pub gpu: bool,
    /// GiB of GPU memory the model may use; weights beyond it stay in RAM. `None` = free memory.
    pub max_vram_gib: Option<f32>,
    /// Flash attention in the diffusion model.
    pub flash_attn: bool,
    /// CPU threads. `None` = half the available hardware threads.
    pub n_threads: Option<i32>,
}

impl Default for ImageModelParams {
    fn default() -> Self {
        Self {
            model: None,
            diffusion_model: None,
            text_encoder: None,
            clip_l: None,
            clip_g: None,
            t5xxl: None,
            vae: None,
            gpu: true,
            max_vram_gib: None,
            flash_attn: true,
            n_threads: None,
        }
    }
}

/// An image generation request. Unset options use the model's defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative_prompt: String,
    /// Pixels; a multiple of 16.
    pub width: u32,
    /// Pixels; a multiple of 16.
    pub height: u32,
    /// Sampling steps (Z-Image-Turbo: 8).
    pub steps: Option<u32>,
    /// Classifier-free guidance scale (Z-Image-Turbo: 1.0).
    pub cfg_scale: Option<f32>,
    /// `None` picks a random seed.
    pub seed: Option<u64>,
    /// Sampler, e.g. `euler`, `euler_a`, `dpm++2m`.
    pub sampler: Option<String>,
    /// Noise schedule, e.g. `discrete`, `karras`, `simple`.
    pub scheduler: Option<String>,
    /// Image-to-image: start from this image instead of noise. The output has its size (a
    /// multiple of 16); `width` and `height` are ignored. In JSON, a base64 PNG.
    #[serde(with = "png_base64")]
    pub init_image: Option<Image>,
    /// How far to move away from `init_image`: 0 keeps it, 1 ignores it. Default 0.75.
    pub strength: Option<f32>,
}

impl Default for ImageRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: None,
            cfg_scale: None,
            seed: None,
            sampler: None,
            scheduler: None,
            init_image: None,
            strength: None,
        }
    }
}

/// Progress of a generation. Weights load lazily, so `Loading` can also come after sampling
/// has started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum ImageProgress {
    Loading { done: u32, total: u32 },
    Sampling { step: u32, steps: u32 },
    Decoding { done: u32, total: u32 },
}

/// A generated image: 8-bit RGB, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl Image {
    /// Decode a PNG (8-bit RGB, RGBA, grey or palette).
    pub fn from_png(bytes: &[u8]) -> Result<Image> {
        let bad = |e: png::DecodingError| Error::InvalidArgument(format!("invalid PNG: {e}"));
        let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder.read_info().map_err(bad)?;
        let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
        let info = reader.next_frame(&mut buf).map_err(bad)?;
        let px = &buf[..info.buffer_size()];
        let rgb = match info.color_type {
            png::ColorType::Rgb => px.to_vec(),
            png::ColorType::Rgba => px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
            png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g]).collect(),
            png::ColorType::GrayscaleAlpha => px.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0]]).collect(),
            other => return Err(Error::InvalidArgument(format!("unsupported PNG colour type {other:?}"))),
        };
        Ok(Image { width: info.width, height: info.height, rgb })
    }

    /// Encode as PNG.
    pub fn to_png(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, self.width, self.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("PNG header to a Vec");
        writer.write_image_data(&self.rgb).expect("PNG data to a Vec");
        writer.finish().expect("PNG to a Vec");
        out
    }
}

/// A loaded image model. Load it when needed and drop it to free its memory; generation can run
/// on another thread while chat sessions are in use.
pub struct ImageModel {
    ctx: NonNull<sd::sd_ctx_t>,
}

// The context is used from one thread at a time (&mut self), but may move between threads.
unsafe impl Send for ImageModel {}

impl Drop for ImageModel {
    fn drop(&mut self) {
        let _guard = setup_lock();
        unsafe { sd::free_sd_ctx(self.ctx.as_ptr()) };
    }
}

/// sd.cpp's progress callback is process-wide, so generations run one at a time.
static GENERATE: Mutex<()> = Mutex::new(());

impl ImageModel {
    pub fn load(params: &ImageModelParams) -> Result<Self> {
        if params.model.is_none() && params.diffusion_model.is_none() {
            return Err(Error::InvalidArgument("image model needs `model` or `diffusion_model`".into()));
        }
        ensure_init();
        route_logs();
        let paths = [
            &params.model,
            &params.diffusion_model,
            &params.text_encoder,
            &params.clip_l,
            &params.clip_g,
            &params.t5xxl,
            &params.vae,
        ];
        let mut owned: Vec<CString> = Vec::new();
        let mut cpath = |p: &Option<PathBuf>| -> Result<*const c_char> {
            let Some(p) = p else { return Ok(std::ptr::null()) };
            if !p.is_file() {
                return Err(Error::ModelNotFound(p.clone()));
            }
            owned.push(c_string(&p.to_string_lossy())?);
            Ok(owned.last().unwrap().as_ptr())
        };
        let [model, diffusion, llm, clip_l, clip_g, t5xxl, vae] = paths.map(&mut cpath);

        let mut p: sd::sd_ctx_params_t = unsafe { std::mem::zeroed() };
        unsafe { sd::sd_ctx_params_init(&mut p) };
        p.model_path = model?;
        p.diffusion_model_path = diffusion?;
        p.llm_path = llm?;
        p.clip_l_path = clip_l?;
        p.clip_g_path = clip_g?;
        p.t5xxl_path = t5xxl?;
        p.vae_path = vae?;
        p.n_threads = params.n_threads.unwrap_or_else(default_threads);
        p.diffusion_flash_attn = params.flash_attn;
        let cpu = c_string("cpu")?;
        if !params.gpu {
            p.backend = cpu.as_ptr();
        }
        let max_vram = params.max_vram_gib.map(|g| c_string(&g.to_string())).transpose()?;
        if let Some(v) = &max_vram {
            p.max_vram = v.as_ptr();
        }

        let ctx = {
            let _guard = setup_lock();
            unsafe { sd::new_sd_ctx(&p) }
        };
        let ctx = NonNull::new(ctx).ok_or_else(|| {
            Error::ModelLoad(params.diffusion_model.clone().or_else(|| params.model.clone()).unwrap_or_default())
        })?;
        Ok(Self { ctx })
    }

    /// Generate an image. `progress` reports loading, sampling and decoding; return `false` to
    /// cancel, which ends with [`Error::Cancelled`].
    pub fn generate(&mut self, req: &ImageRequest, mut progress: impl FnMut(ImageProgress) -> bool) -> Result<Image> {
        let (width, height) = req.init_image.as_ref().map_or((req.width, req.height), |i| (i.width, i.height));
        if width == 0 || height == 0 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
            return Err(Error::InvalidArgument("width and height must be positive multiples of 16".into()));
        }
        if let Some(init) = &req.init_image
            && init.rgb.len() != init.width as usize * init.height as usize * 3
        {
            return Err(Error::InvalidArgument("init_image size doesn't match its RGB data".into()));
        }
        let prompt = c_string(&req.prompt)?;
        let negative = c_string(&req.negative_prompt)?;
        let mut g: sd::sd_img_gen_params_t = unsafe { std::mem::zeroed() };
        unsafe { sd::sd_img_gen_params_init(&mut g) };
        g.prompt = prompt.as_ptr();
        g.negative_prompt = negative.as_ptr();
        g.width = width as i32;
        g.height = height as i32;
        if let Some(init) = &req.init_image {
            // sd.cpp only reads the pixels.
            g.init_image = sd::sd_image_t { width, height, channel: 3, data: init.rgb.as_ptr() as *mut u8 };
            g.strength = req.strength.unwrap_or(0.75).clamp(0.0, 1.0);
        }
        g.batch_count = 1;
        g.seed = req.seed.map_or(-1, |s| (s & i64::MAX as u64) as i64);
        if let Some(steps) = req.steps {
            g.sample_params.sample_steps = steps as i32;
        }
        if let Some(cfg) = req.cfg_scale {
            g.sample_params.guidance.txt_cfg = cfg;
        }
        if let Some(name) = &req.sampler {
            let m = unsafe { sd::str_to_sample_method(c_string(name)?.as_ptr()) };
            if m == sd::sample_method_t_SAMPLE_METHOD_COUNT {
                return Err(Error::InvalidArgument(format!("unknown sampler '{name}'")));
            }
            g.sample_params.sample_method = m;
        }
        if let Some(name) = &req.scheduler {
            let s = unsafe { sd::str_to_scheduler(c_string(name)?.as_ptr()) };
            if s == sd::scheduler_t_SCHEDULER_COUNT {
                return Err(Error::InvalidArgument(format!("unknown scheduler '{name}'")));
            }
            g.sample_params.scheduler = s;
        }

        let _one_at_a_time = GENERATE.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = self.ctx.as_ptr();
        let steps = g.sample_params.sample_steps.max(1) as u32;
        let mut state = Progress { callback: &mut progress, ctx, steps, sampled: false, cancelled: false };
        let (mut images, mut count): (*mut sd::sd_image_t, c_int) = (std::ptr::null_mut(), 0);
        let ok = unsafe {
            sd::sd_cancel_generation(ctx, sd::sd_cancel_mode_t_SD_CANCEL_RESET);
            sd::sd_set_progress_callback(Some(on_progress), &mut state as *mut Progress as *mut c_void);
            let ok = sd::generate_image(ctx, &g, &mut images, &mut count);
            sd::sd_set_progress_callback(None, std::ptr::null_mut());
            ok
        };
        let first = (count > 0 && !images.is_null()).then(|| unsafe { &*images });
        let result = match first {
            _ if state.cancelled => Err(Error::Cancelled),
            Some(img) if ok && !img.data.is_null() => {
                let (w, h, ch) = (img.width as usize, img.height as usize, img.channel as usize);
                let src = unsafe { std::slice::from_raw_parts(img.data, w * h * ch) };
                let rgb = match ch {
                    3 => src.to_vec(),
                    4 => src.as_chunks::<4>().0.iter().flat_map(|px| [px[0], px[1], px[2]]).collect(),
                    n => return Err(Error::ImageGeneration(format!("unexpected {n} channels"))),
                };
                Ok(Image { width: img.width, height: img.height, rgb })
            }
            _ => Err(Error::ImageGeneration("generation failed (see the log)".into())),
        };
        if !images.is_null() {
            unsafe { sd::free_sd_images(images, count) };
        }
        result
    }
}

struct Progress<'a> {
    callback: &'a mut dyn FnMut(ImageProgress) -> bool,
    ctx: *mut sd::sd_ctx_t,
    /// Sampling steps; sd.cpp reports loading and decoding through the same callback.
    steps: u32,
    sampled: bool,
    cancelled: bool,
}

unsafe extern "C" fn on_progress(step: c_int, steps: c_int, _time: f32, data: *mut c_void) {
    let state = unsafe { &mut *(data as *mut Progress) };
    if state.cancelled {
        return;
    }
    let (done, total) = (step.max(0) as u32, steps.max(0) as u32);
    let event = if total == state.steps {
        state.sampled |= done == total;
        ImageProgress::Sampling { step: done, steps: total }
    } else if state.sampled {
        ImageProgress::Decoding { done, total }
    } else {
        ImageProgress::Loading { done, total }
    };
    let go = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (state.callback)(event))).unwrap_or(false);
    if !go {
        state.cancelled = true;
        unsafe { sd::sd_cancel_generation(state.ctx, sd::sd_cancel_mode_t_SD_CANCEL_ALL) };
    }
}

fn route_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { sd::sd_set_log_callback(Some(on_log), std::ptr::null_mut()) });
}

unsafe extern "C" fn on_log(level: sd::sd_log_level_t, text: *const c_char, _: *mut c_void) {
    if text.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let text = text.trim_end();
    if text.is_empty() {
        return;
    }
    let level = match level {
        sd::sd_log_level_t_SD_LOG_ERROR => log::Level::Error,
        sd::sd_log_level_t_SD_LOG_WARN => log::Level::Warn,
        sd::sd_log_level_t_SD_LOG_INFO => log::Level::Info,
        _ => log::Level::Debug,
    };
    log::log!(target: "stable-diffusion", level, "{text}");
}

fn c_string(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::InvalidArgument("string contains NUL".into()))
}

/// `Option<Image>` as a base64 PNG in JSON.
mod png_base64 {
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    use super::Image;
    use crate::media::{base64_decode, base64_encode};

    pub fn serialize<S: Serializer>(image: &Option<Image>, s: S) -> Result<S::Ok, S::Error> {
        match image {
            Some(img) => s.serialize_some(&base64_encode(&img.to_png())),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Image>, D::Error> {
        let Some(text) = Option::<String>::deserialize(d)? else { return Ok(None) };
        let text = text.split_once("base64,").map_or(text.as_str(), |(_, b)| b);
        let bytes = base64_decode(text.trim()).ok_or_else(|| D::Error::custom("init_image: invalid base64"))?;
        Image::from_png(&bytes).map(Some).map_err(D::Error::custom)
    }
}
