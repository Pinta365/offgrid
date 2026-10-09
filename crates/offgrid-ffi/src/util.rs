use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::Serialize;
use serde::de::DeserializeOwned;

pub(crate) type FfiResult<T> = Result<T, String>;

/// Write `msg` to `error_out` (if non-NULL) as a caller-owned C string.
fn write_error(error_out: *mut *mut c_char, msg: impl std::fmt::Display) {
    log::debug!(target: "offgrid-ffi", "error: {msg}");
    if error_out.is_null() {
        return;
    }
    let msg = CString::new(msg.to_string().replace('\0', " ")).expect("NULs removed");
    unsafe { *error_out = msg.into_raw() };
}

/// Run `f`, converting errors and panics into `fallback` + `*error_out`.
/// `*error_out` is set to NULL on success.
pub(crate) fn guard<T>(error_out: *mut *mut c_char, fallback: T, f: impl FnOnce() -> FfiResult<T>) -> T {
    if !error_out.is_null() {
        unsafe { *error_out = std::ptr::null_mut() };
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            write_error(error_out, e);
            fallback
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            write_error(error_out, format!("internal panic: {msg}"));
            fallback
        }
    }
}
pub(crate) fn str_arg<'a>(ptr: *const c_char, name: &str) -> FfiResult<&'a str> {
    if ptr.is_null() {
        return Err(format!("{name} is NULL"));
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| format!("{name} is not valid UTF-8"))
}

/// Parse optional JSON; NULL or empty string gives `T::default()`.
pub(crate) fn json_arg<T: DeserializeOwned + Default>(ptr: *const c_char, name: &str) -> FfiResult<T> {
    if ptr.is_null() {
        return Ok(T::default());
    }
    let s = str_arg(ptr, name)?;
    if s.trim().is_empty() {
        return Ok(T::default());
    }
    serde_json::from_str(s).map_err(|e| format!("invalid {name} JSON: {e}"))
}

pub(crate) fn json_out<T: Serialize>(value: &T) -> FfiResult<*mut c_char> {
    let s = serde_json::to_string(value).map_err(|e| e.to_string())?;
    Ok(CString::new(s).map_err(|e| e.to_string())?.into_raw())
}

pub(crate) fn handle<'a, T>(ptr: *const T, name: &str) -> FfiResult<&'a T> {
    unsafe { ptr.as_ref() }.ok_or_else(|| format!("{name} is NULL"))
}
