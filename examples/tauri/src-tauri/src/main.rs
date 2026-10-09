//! Desktop chat app that embeds offgrid directly (no sidecar, no FFI).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use offgrid::{ChatMessage, ChatRequest, ChatResponse, ChatSession, Model, ModelParams, RequestCancel, SessionParams, StreamEvent};
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum Status {
    Loading { path: String },
    Ready { model: String, device: String, n_ctx: usize },
    Failed { error: String },
}

/// Streamed to the frontend while a reply is generated.
#[derive(Clone, Serialize)]
#[serde(tag = "type", content = "text", rename_all = "snake_case")]
enum Delta {
    Content(String),
    Reasoning(String),
}

struct Engine {
    status: Mutex<Status>,
    session: Mutex<Option<ChatSession>>,
    /// Cancels the reply being generated, if any.
    cancel: Mutex<Option<RequestCancel>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            status: Mutex::new(Status::Loading { path: String::new() }),
            session: Mutex::new(None),
            cancel: Mutex::new(None),
        }
    }
}

/// Lock, ignoring poisoning: a panic in one request shouldn't wedge the app.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `$OFFGRID_MODEL`, or the first `.gguf` bundled in the app's `models/` resource folder.
fn find_model(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("OFFGRID_MODEL") {
        return Ok(path.into());
    }
    let dir = app.path().resource_dir().map_err(|e| e.to_string())?.join("models");
    let mut models: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            name.ends_with(".gguf") && !name.starts_with("mmproj")
        })
        .collect();
    models.sort();
    models
        .into_iter()
        .next()
        .ok_or_else(|| format!("no .gguf model in {} (or set OFFGRID_MODEL)", dir.display()))
}

fn load(app: &AppHandle, engine: &Engine) -> Result<Status, String> {
    let path = find_model(app)?;
    set_status(app, engine, Status::Loading { path: path.display().to_string() });
    let (session, status) = open_session(&path)?;
    *lock(&engine.session) = Some(session);
    Ok(status)
}

fn open_session(path: &Path) -> Result<(ChatSession, Status), String> {
    let model = Model::load(path, &ModelParams::default()).map_err(|e| e.to_string())?;
    let session = ChatSession::new(model.clone(), &SessionParams::default()).map_err(|e| e.to_string())?;
    let info = model.info();
    let device = offgrid::devices()
        .into_iter()
        .find(|d| matches!(d.kind, offgrid::DeviceKind::Gpu | offgrid::DeviceKind::Igpu))
        .map_or_else(|| "CPU".to_string(), |d| d.description);
    let status = Status::Ready {
        model: info.name.unwrap_or_else(|| info.description.clone()),
        device,
        n_ctx: session.n_ctx(),
    };
    Ok((session, status))
}

fn set_status(app: &AppHandle, engine: &Engine, status: Status) {
    *lock(&engine.status) = status.clone();
    let _ = app.emit("status", status);
}

#[tauri::command]
fn status(engine: State<'_, Arc<Engine>>) -> Status {
    lock(&engine.status).clone()
}

#[tauri::command]
async fn chat(
    engine: State<'_, Arc<Engine>>,
    messages: Vec<ChatMessage>,
    on_delta: Channel<Delta>,
) -> Result<ChatResponse, String> {
    let engine = Arc::clone(&engine);
    // Generation blocks for seconds; keep it off the async runtime. Sending fails when the
    // window is gone; generation stops then.
    tauri::async_runtime::spawn_blocking(move || generate(&engine, messages, |delta| on_delta.send(delta).is_ok()))
        .await
        .map_err(|e| e.to_string())?
}

/// Generate a reply to `messages`. `on_delta` returns `false` to stop.
fn generate(engine: &Engine, messages: Vec<ChatMessage>, mut on_delta: impl FnMut(Delta) -> bool) -> Result<ChatResponse, String> {
    let mut session = lock(&engine.session);
    let session = session.as_mut().ok_or("the model is not loaded")?;
    // Taken while holding the session, so Stop can't hit a different request.
    *lock(&engine.cancel) = Some(session.cancel_token().next_request());

    let result = session.complete_stream(&ChatRequest::new(messages), |event| match event {
        StreamEvent::Content(text) => on_delta(Delta::Content(text.to_string())),
        StreamEvent::Reasoning(text) => on_delta(Delta::Reasoning(text.to_string())),
        StreamEvent::ToolCall(_) => true,
    });
    *lock(&engine.cancel) = None;
    result.map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel(engine: State<'_, Arc<Engine>>) {
    if let Some(request) = lock(&engine.cancel).as_ref() {
        request.cancel();
    }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let engine = Arc::new(Engine::default());
            app.manage(Arc::clone(&engine));

            // Loading takes a few seconds; the window shows progress meanwhile.
            let app = app.handle().clone();
            std::thread::spawn(move || {
                let status = load(&app, &engine).unwrap_or_else(|error| Status::Failed { error });
                set_status(&app, &engine, status);
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![status, chat, cancel])
        .run(tauri::generate_context!())
        .expect("error while running the app");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frontend's request shape, end to end through `generate`. Needs `OFFGRID_TEST_MODEL`.
    #[test]
    fn chat_and_cancel() {
        let Some(path) = std::env::var_os("OFFGRID_TEST_MODEL") else {
            eprintln!("skipped: OFFGRID_TEST_MODEL not set");
            return;
        };
        let engine = Engine::default();
        let (session, status) = open_session(Path::new(&path)).unwrap();
        assert!(matches!(status, Status::Ready { .. }));
        *lock(&engine.session) = Some(session);

        // Messages as main.js sends them.
        let mut messages: Vec<ChatMessage> = serde_json::from_str(
            r#"[{"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"Reply with just the word: pong"}]"#,
        )
        .unwrap();
        let mut streamed = String::new();
        let res = generate(&engine, messages.clone(), |d| {
            if let Delta::Content(t) = d {
                streamed.push_str(&t);
            }
            true
        })
        .unwrap();
        assert!(res.content.to_lowercase().contains("pong"), "{res:?}");
        assert_eq!(streamed.trim(), res.content.trim());
        assert!(lock(&engine.cancel).is_none());

        // Second turn reuses the KV cache.
        messages.push(ChatMessage::assistant(res.content));
        messages.push(ChatMessage::user("Count from 1 to 200, one number per line."));
        let res = generate(&engine, messages.clone(), |_| true).unwrap();
        assert!(res.usage.cached_tokens > 0, "{:?}", res.usage);

        // Stop from another thread while generating, as the cancel command does.
        let engine = Arc::new(engine);
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                let mut first = true;
                generate(&engine, messages, |_| {
                    if std::mem::take(&mut first) {
                        tx.send(()).unwrap();
                    }
                    true
                })
            })
        };
        rx.recv().unwrap();
        lock(&engine.cancel).as_ref().unwrap().cancel();
        let res = worker.join().unwrap().unwrap();
        assert_eq!(res.finish_reason, offgrid::FinishReason::Cancelled);
        assert!(lock(&engine.cancel).is_none());
    }
}
