use std::ffi::{CStr, c_char, c_void};
use std::sync::{Mutex, MutexGuard, Once};

use offgrid_sys as sys;
use serde::Serialize;

static INIT: Once = Once::new();

/// llama.cpp backends (notably Vulkan) initialize device state lazily and without
/// locking, so model loading and context creation must not run concurrently.
/// Inference on separate contexts is fine in parallel.
static SETUP_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn setup_lock() -> MutexGuard<'static, ()> {
    ensure_init();
    SETUP_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Initialize llama.cpp once per process. Called implicitly by every entry point.
pub(crate) fn ensure_init() {
    INIT.call_once(|| unsafe {
        // AMD's switchable-graphics Vulkan layer (hybrid AMD iGPU + other GPU laptops) hangs
        // vkCreateInstance; llama.cpp picks the GPU itself, so the layer isn't needed.
        #[cfg(windows)]
        if std::env::var_os("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1").is_none() {
            std::env::set_var("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1", "1");
        }
        #[cfg(feature = "dynamic-backends")]
        crate::loader::preload_core_libs();
        sys::llama_log_set(Some(log_callback), std::ptr::null_mut());
        sys::mtmd_log_set(Some(log_callback), std::ptr::null_mut());
        sys::mtmd_helper_log_set(Some(log_callback), std::ptr::null_mut());
        #[cfg(feature = "dynamic-backends")]
        crate::loader::load_backends();
        sys::llama_backend_init();
    });
}

/// Route llama.cpp / ggml logging into the `log` crate.
unsafe extern "C" fn log_callback(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    if text.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let text = text.trim_end();
    if text.is_empty() {
        return;
    }
    let level = match level {
        sys::ggml_log_level_GGML_LOG_LEVEL_ERROR => log::Level::Error,
        sys::ggml_log_level_GGML_LOG_LEVEL_WARN => log::Level::Warn,
        sys::ggml_log_level_GGML_LOG_LEVEL_INFO => log::Level::Info,
        _ => log::Level::Debug,
    };
    log::log!(target: "llama.cpp", level, "{text}");
}

pub(crate) unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Cpu,
    Gpu,
    /// Integrated GPU sharing host memory.
    Igpu,
    Accel,
    Other,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    /// Backend device name, e.g. `Vulkan0`.
    pub name: String,
    pub description: String,
    pub kind: DeviceKind,
    pub memory_free: u64,
    pub memory_total: u64,
}

/// Compute devices llama.cpp can use, in backend order.
pub fn devices() -> Vec<DeviceInfo> {
    let _guard = setup_lock();
    let count = unsafe { sys::ggml_backend_dev_count() };
    (0..count)
        .filter_map(|i| {
            let dev = unsafe { sys::ggml_backend_dev_get(i) };
            if dev.is_null() {
                return None;
            }
            let (mut free, mut total) = (0usize, 0usize);
            unsafe { sys::ggml_backend_dev_memory(dev, &mut free, &mut total) };
            let kind = match unsafe { sys::ggml_backend_dev_type(dev) } {
                sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => DeviceKind::Cpu,
                sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => DeviceKind::Gpu,
                sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => DeviceKind::Igpu,
                sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL => DeviceKind::Accel,
                _ => DeviceKind::Other,
            };
            Some(DeviceInfo {
                name: unsafe { cstr_to_string(sys::ggml_backend_dev_name(dev)) },
                description: unsafe { cstr_to_string(sys::ggml_backend_dev_description(dev)) }.trim().to_string(),
                kind,
                memory_free: free as u64,
                memory_total: total as u64,
            })
        })
        .collect()
}

/// llama.cpp's CPU/backend feature summary.
pub fn system_info() -> String {
    ensure_init();
    unsafe { cstr_to_string(sys::llama_print_system_info()) }
}

/// Version of the bundled llama.cpp.
pub fn version() -> String {
    ensure_init();
    unsafe { cstr_to_string(sys::llama_version()) }
}
