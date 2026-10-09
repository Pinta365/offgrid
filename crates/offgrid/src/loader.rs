//! Runtime discovery of ggml backend modules (`dynamic-backends` builds).
//!
//! Backends (libggml-cpu-*.so, libggml-vulkan.so, ...) ship next to the offgrid
//! library. Because the host may be Deno, Python, etc., the executable's directory
//! is the wrong place to look; we locate the module that contains this code.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use offgrid_sys as sys;

/// Directories that may hold the shipped llama/ggml libraries and backend modules.
fn candidate_dirs() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("OFFGRID_BACKEND_DIR") {
        candidates.push(dir.into());
    }
    if let Some(dir) = module_dir() {
        // Executables (e.g. offgrid-server) live in bin/ next to the package's lib/.
        candidates.push(dir.join("../lib"));
        candidates.push(dir);
    }
    if !sys::BUILD_BACKEND_DIR.is_empty() {
        candidates.push(sys::BUILD_BACKEND_DIR.into());
    }
    candidates
}

/// Windows: offgrid.dll delay-loads llama/ggml/mtmd (see offgrid-ffi/build.rs). Load them by
/// full path from our folder first; the delay-load stubs then bind to these modules.
/// Must run before any other llama/ggml call. No-op elsewhere (RUNPATH handles it).
pub(crate) fn preload_core_libs() {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::System::LibraryLoader::{LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW};
        // Dependency order; the last one is used to locate the folder.
        const LIBS: [&str; 4] = ["ggml-base.dll", "ggml.dll", "libllama.dll", "libmtmd.dll"];
        let Some(dir) = candidate_dirs().into_iter().find(|d| d.join(LIBS[3]).is_file()) else {
            log::warn!(target: "offgrid", "llama/ggml DLLs not found next to offgrid; relying on the default search path");
            return;
        };
        for lib in LIBS {
            let wide: Vec<u16> = dir.join(lib).as_os_str().encode_wide().chain([0]).collect();
            let h = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
            if h.is_null() {
                log::error!(target: "offgrid", "failed to load {}", dir.join(lib).display());
            }
        }
    }
}

pub(crate) fn load_backends() {
    let candidates = candidate_dirs();
    match candidates.iter().find(|d| has_cpu_backend(d)) {
        Some(dir) => {
            log::info!(target: "offgrid", "loading backends from {}", dir.display());
            if let Ok(c) = CString::new(dir.to_string_lossy().as_bytes()) {
                unsafe { sys::ggml_backend_load_all_from_path(c.as_ptr()) };
            }
        }
        None => {
            log::warn!(target: "offgrid", "no backend directory found in {candidates:?}; trying ggml defaults");
            unsafe { sys::ggml_backend_load_all() };
        }
    }
    if unsafe { sys::ggml_backend_dev_count() } == 0 {
        log::error!(target: "offgrid", "no compute backends could be loaded");
    }
}

fn has_cpu_backend(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("libggml-cpu") || name.starts_with("ggml-cpu")
        })
    })
}

/// Directory of the shared object / DLL containing this function.
#[cfg(unix)]
fn module_dir() -> Option<PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let addr = module_dir as *const libc::c_void;
    if unsafe { libc::dladdr(addr, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned();
    std::fs::canonicalize(path).ok()?.parent().map(Path::to_path_buf)
}

#[cfg(windows)]
fn module_dir() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleFileNameW, GetModuleHandleExW,
    };
    let mut module: HMODULE = std::ptr::null_mut();
    let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if unsafe { GetModuleHandleExW(flags, module_dir as *const u16, &mut module) } == 0 {
        return None;
    }
    let mut buf = vec![0u16; 32768];
    let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if n == 0 {
        return None;
    }
    let path = PathBuf::from(std::ffi::OsString::from_wide(&buf[..n]));
    path.parent().map(Path::to_path_buf)
}

#[cfg(not(any(unix, windows)))]
fn module_dir() -> Option<PathBuf> {
    None
}
