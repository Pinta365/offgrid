//! `POST /v1/images/generations` (feature `image`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use offgrid::{ImageModel, ImageRequest};
use serde_json::{Value, json};

use crate::openai::ApiError;
use crate::{AppState, body_error, now};

pub(crate) type ImageSlot = (String, Arc<Mutex<ImageModel>>);

/// Sets its flag when dropped: the client went away before the image was done.
struct Gone(Arc<AtomicBool>);

impl Drop for Gone {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// OpenAI's images API with `b64_json` output. Besides `prompt`, `size` and `n` (1 only), it
/// accepts offgrid's options: `negative_prompt`, `steps`, `cfg_scale`, `seed`, `sampler`,
/// `scheduler`, and `init_image` (base64 PNG) with `strength` for image-to-image.
pub(crate) async fn generations(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let mut body = match body {
        Ok(Json(b)) => b,
        Err(e) => return body_error(e.status(), e.body_text()).into_response(),
    };
    let Some((model_id, model)) = state.image.clone() else {
        return ApiError::bad_request("no image model loaded (--image-diffusion-model or --image-model)")
            .into_response();
    };
    let Some(obj) = body.as_object_mut() else {
        return ApiError::bad_request("expected a JSON object").into_response();
    };
    if obj.get("prompt").and_then(Value::as_str).is_none_or(str::is_empty) {
        return ApiError::bad_request("missing prompt").into_response();
    }
    if obj.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
        return ApiError::bad_request("only n=1 is supported").into_response();
    }
    match obj.get("response_format").and_then(Value::as_str) {
        None | Some("b64_json") => {}
        Some(other) => {
            return ApiError::bad_request(format!("unsupported response_format '{other}' (use b64_json)"))
                .into_response();
        }
    }
    if let Some(size) = obj.get("size").and_then(Value::as_str).filter(|s| *s != "auto") {
        let Some((w, h)) =
            size.split_once('x').and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        else {
            return ApiError::bad_request(format!("invalid size '{size}', expected WIDTHxHEIGHT")).into_response();
        };
        obj.insert("width".into(), w.into());
        obj.insert("height".into(), h.into());
    }
    for key in ["model", "n", "size", "response_format", "quality", "style", "user", "output_format", "background"] {
        obj.remove(key);
    }
    let req: ImageRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return ApiError::bad_request(e.to_string()).into_response(),
    };

    let gone = Arc::new(AtomicBool::new(false));
    let _guard = Gone(gone.clone());
    let worker_state = state.clone();
    let joined = tokio::task::spawn_blocking(move || {
        let mut model = model.lock().unwrap_or_else(|e| e.into_inner());
        model.generate(&req, |_| !gone.load(Ordering::SeqCst) && !worker_state.is_shutting_down())
    })
    .await;
    match joined {
        Ok(Ok(image)) => {
            let b64 = base64::engine::general_purpose::STANDARD.encode(image.to_png());
            Json(json!({"created": now(), "model": model_id, "data": [{"b64_json": b64}]})).into_response()
        }
        Ok(Err(e)) => ApiError::from_core(e).into_response(),
        Err(e) => ApiError::internal(format!("worker failed: {e}")).into_response(),
    }
}
