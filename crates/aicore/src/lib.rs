//! Embeddable offline AI core.
//!
//! Safe Rust API over llama.cpp: load a GGUF model, run chat completions with
//! streaming and cancellation, and compute embeddings.
//!
//! ```no_run
//! use aicore::{ChatMessage, ChatRequest, ChatSession, Model, ModelParams, SessionParams};
//!
//! let model = Model::load("model.gguf", &ModelParams::default())?;
//! let mut chat = ChatSession::new(model, &SessionParams::default())?;
//! let req = ChatRequest::new(vec![ChatMessage::user("Hello!")]);
//! let resp = chat.complete(&req, |piece| { print!("{piece}"); true })?;
//! println!("\n{:?}", resp.finish_reason);
//! # Ok::<(), aicore::Error>(())
//! ```

mod backend;
mod batch;
mod chat;
mod embed;
mod error;
mod json_schema;
#[cfg(feature = "dynamic-backends")]
mod loader;
mod model;
mod sampling;
mod template;
mod text;
mod tool_parse;
mod tools;
mod media;

pub use backend::{DeviceInfo, DeviceKind, devices, system_info, version};
pub use chat::{
    CancelToken, ChatMessage, ChatRequest, ChatResponse, ChatSession, FinishReason, JsonSchemaFormat,
    ResponseFormat, SessionParams, StreamEvent, Usage,
};
pub use embed::{EmbedParams, Embedder};
pub use error::{Error, Result};
pub use model::{GpuLayers, Model, ModelInfo, ModelParams};
pub use sampling::SamplingParams;
pub use tools::{FunctionCall, FunctionDef, Tool, ToolCall, ToolChoice};
pub use media::{Media, MediaKind, media_marker};
