use std::ffi::{CString, c_char};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use offgrid_sys as sys;
use serde::{Deserialize, Serialize};

use crate::backend::{cstr_to_string, ensure_init, setup_lock};
use crate::media::Projector;
use crate::{Error, Result};

/// How many transformer layers to offload to GPU devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuLayers {
    /// Offload everything that fits the selected devices.
    #[default]
    All,
    /// CPU only.
    None,
    Count(u32),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelParams {
    pub gpu_layers: GpuLayers,
    /// Restrict offloading to these device names (see [`crate::devices`]), e.g. `["Vulkan0"]`.
    /// `None` lets llama.cpp choose.
    pub devices: Option<Vec<String>>,
    /// Multimodal projector (`mmproj-*.gguf`) shipped with a multimodal model; enables image
    /// and/or audio input, depending on the model.
    pub mmproj: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub description: String,
    pub n_params: u64,
    pub size_bytes: u64,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub has_encoder: bool,
    pub has_decoder: bool,
    pub chat_template: Option<String>,
    /// Image input available (projector loaded and supports vision).
    pub vision: bool,
    /// Audio input available (projector loaded and supports audio).
    pub audio: bool,
}

/// A loaded GGUF model. Cheap to share: sessions and embedders hold an `Arc<Model>`.
pub struct Model {
    /// Freed before the text model in `Drop`.
    projector: Option<Projector>,
    ptr: NonNull<sys::llama_model>,
}

// llama_model is immutable after load and safe to use from several contexts/threads.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        // The projector references the text model: free it first (it takes setup_lock itself).
        drop(self.projector.take());
        let _guard = setup_lock();
        unsafe { sys::llama_model_free(self.ptr.as_ptr()) };
    }
}

impl Model {
    pub fn load(path: impl AsRef<Path>, params: &ModelParams) -> Result<Arc<Self>> {
        ensure_init();
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error::ModelNotFound(path.to_path_buf()));
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidArgument("model path contains NUL".into()))?;

        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = match params.gpu_layers {
            GpuLayers::All => -1,
            GpuLayers::None => 0,
            GpuLayers::Count(n) => n.min(i32::MAX as u32) as i32,
        };

        // NULL-terminated device list; must outlive the load call.
        let mut device_list: Vec<sys::ggml_backend_dev_t> = Vec::new();
        if let Some(names) = &params.devices {
            for name in names {
                device_list.push(find_device(name)?);
            }
            device_list.push(std::ptr::null_mut());
            mp.devices = device_list.as_mut_ptr();
        }

        let raw = {
            let _guard = setup_lock();
            unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), mp) }
        };
        let ptr = NonNull::new(raw).ok_or_else(|| Error::ModelLoad(path.to_path_buf()))?;
        let mut model = Self { projector: None, ptr };
        if let Some(mmproj) = &params.mmproj {
            let threads = crate::chat::default_threads();
            model.projector = Some(Projector::load(mmproj, ptr.as_ptr(), params.gpu_layers != GpuLayers::None, threads)?);
        }
        Ok(Arc::new(model))
    }

    /// The multimodal projector, if loaded.
    pub(crate) fn projector(&self) -> Option<&Projector> {
        self.projector.as_ref()
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::llama_model {
        self.ptr.as_ptr()
    }

    pub(crate) fn vocab(&self) -> *const sys::llama_vocab {
        unsafe { sys::llama_model_get_vocab(self.ptr.as_ptr()) }
    }

    pub fn info(&self) -> ModelInfo {
        let m = self.ptr.as_ptr();
        let mut desc = vec![0u8; 256];
        let n = unsafe { sys::llama_model_desc(m, desc.as_mut_ptr() as *mut c_char, desc.len()) };
        desc.truncate(n.max(0) as usize);
        unsafe {
            ModelInfo {
                name: self.meta("general.name"),
                architecture: self.meta("general.architecture"),
                description: String::from_utf8_lossy(&desc).into_owned(),
                n_params: sys::llama_model_n_params(m),
                size_bytes: sys::llama_model_size(m),
                n_ctx_train: sys::llama_model_n_ctx_train(m).max(0) as u32,
                n_embd: sys::llama_model_n_embd(m).max(0) as u32,
                n_layer: sys::llama_model_n_layer(m).max(0) as u32,
                has_encoder: sys::llama_model_has_encoder(m),
                has_decoder: sys::llama_model_has_decoder(m),
                chat_template: self.chat_template(),
                vision: self.projector.as_ref().is_some_and(|p| p.vision),
                audio: self.projector.as_ref().is_some_and(|p| p.audio),
            }
        }
    }

    /// The model's default chat template, if the GGUF carries one.
    pub fn chat_template(&self) -> Option<String> {
        let ptr = unsafe { sys::llama_model_chat_template(self.ptr.as_ptr(), std::ptr::null()) };
        (!ptr.is_null()).then(|| unsafe { cstr_to_string(ptr) })
    }

    /// Read a scalar GGUF metadata value as a string.
    pub fn meta(&self, key: &str) -> Option<String> {
        let key = CString::new(key).ok()?;
        let mut buf = vec![0u8; 512];
        loop {
            let n = unsafe {
                sys::llama_model_meta_val_str(
                    self.ptr.as_ptr(),
                    key.as_ptr(),
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                )
            };
            if n < 0 {
                return None;
            }
            if (n as usize) < buf.len() {
                buf.truncate(n as usize);
                return Some(String::from_utf8_lossy(&buf).into_owned());
            }
            buf.resize(n as usize + 1, 0);
        }
    }

    /// Tokenize text. `add_special` adds BOS/EOS as the model is configured;
    /// `parse_special` turns special-token text (e.g. `<|im_start|>`) into control tokens.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let len = i32::try_from(text.len()).map_err(|_| Error::Tokenize)?;
        let mut tokens = vec![0 as sys::llama_token; text.len() / 2 + 8];
        for _ in 0..2 {
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab(),
                    text.as_ptr() as *const c_char,
                    len,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    parse_special,
                )
            };
            if n >= 0 {
                tokens.truncate(n as usize);
                return Ok(tokens);
            }
            if n == i32::MIN {
                break;
            }
            tokens.resize((-n) as usize, 0);
        }
        Err(Error::Tokenize)
    }

    /// Raw bytes of a token (may be a partial UTF-8 sequence). `special` renders control
    /// tokens (e.g. `[TOOL_CALLS]`) as text instead of dropping them.
    pub(crate) fn token_bytes(&self, token: sys::llama_token, buf: &mut Vec<u8>, special: bool) {
        buf.clear();
        buf.resize(64, 0);
        let mut n = unsafe {
            sys::llama_token_to_piece(self.vocab(), token, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, special)
        };
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe {
                sys::llama_token_to_piece(self.vocab(), token, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, special)
            };
        }
        buf.truncate(n.max(0) as usize);
    }

    /// Text of a (special) token such as BOS/EOS; empty for invalid ids.
    pub(crate) fn token_text(&self, token: sys::llama_token) -> String {
        if token < 0 {
            return String::new();
        }
        unsafe { crate::backend::cstr_to_string(sys::llama_vocab_get_text(self.vocab(), token)) }
    }

    pub(crate) fn is_eog(&self, token: sys::llama_token) -> bool {
        unsafe { sys::llama_vocab_is_eog(self.vocab(), token) }
    }

    pub(crate) fn n_vocab(&self) -> i32 {
        unsafe { sys::llama_vocab_n_tokens(self.vocab()) }
    }
}

fn find_device(name: &str) -> Result<sys::ggml_backend_dev_t> {
    let count = unsafe { sys::ggml_backend_dev_count() };
    for i in 0..count {
        let dev = unsafe { sys::ggml_backend_dev_get(i) };
        if !dev.is_null() && unsafe { cstr_to_string(sys::ggml_backend_dev_name(dev)) } == name {
            return Ok(dev);
        }
    }
    Err(Error::InvalidArgument(format!("unknown device '{name}'")))
}
