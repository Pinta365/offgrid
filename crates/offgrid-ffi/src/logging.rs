use std::ffi::{CString, c_char, c_void};
use std::sync::{Once, RwLock};

use log::{Level, LevelFilter, Log, Metadata, Record};

/// Log callback: `level` is 1=error, 2=warn, 3=info, 4=debug, 5=trace.
pub type OffgridLogCallback = Option<unsafe extern "C" fn(level: i32, message: *const c_char, user_data: *mut c_void)>;

struct Sink {
    callback: OffgridLogCallback,
    user_data: *mut c_void,
}

// The C side promises the callback/user_data are usable from any thread.
unsafe impl Send for Sink {}
unsafe impl Sync for Sink {}

static SINK: RwLock<Option<Sink>> = RwLock::new(None);
static INSTALL: Once = Once::new();

struct FfiLogger;

impl Log for FfiLogger {
    fn enabled(&self, _: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let sink = SINK.read().unwrap_or_else(|e| e.into_inner());
        match sink.as_ref() {
            Some(Sink { callback: Some(cb), user_data }) => {
                let msg = format!("[{}] {}", record.target(), record.args()).replace('\0', " ");
                if let Ok(msg) = CString::new(msg) {
                    unsafe { cb(record.level() as i32, msg.as_ptr(), *user_data) };
                }
            }
            _ => eprintln!("offgrid {} [{}] {}", record.level(), record.target(), record.args()),
        }
    }

    fn flush(&self) {}
}

/// Install the logger once. Default level comes from `OFFGRID_LOG` (error|warn|info|debug|trace|off),
/// otherwise logging is off.
pub(crate) fn install() {
    INSTALL.call_once(|| {
        let level = std::env::var("OFFGRID_LOG")
            .ok()
            .and_then(|v| v.parse::<LevelFilter>().ok())
            .unwrap_or(LevelFilter::Off);
        if log::set_logger(&FfiLogger).is_ok() {
            log::set_max_level(level);
        }
    });
}

pub(crate) fn set_sink(callback: OffgridLogCallback, user_data: *mut c_void, level: i32) {
    install();
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = callback.map(|_| Sink { callback, user_data });
    log::set_max_level(match level {
        i32::MIN..=0 => LevelFilter::Off,
        1 => Level::Error.to_level_filter(),
        2 => Level::Warn.to_level_filter(),
        3 => Level::Info.to_level_filter(),
        4 => Level::Debug.to_level_filter(),
        _ => Level::Trace.to_level_filter(),
    });
}
