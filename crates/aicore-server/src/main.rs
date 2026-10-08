//! aicore-server: OpenAI-compatible local inference sidecar.
//!
//!   aicore-server --model qwen3-4b.gguf --embedding-model embed.gguf --port 8080

use std::net::SocketAddr;
use std::path::PathBuf;

use aicore::{GpuLayers, ModelParams, SessionParams};
use aicore_server::{AppState, ServerConfig, parse_gpu_layers, router};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about = "OpenAI-compatible local inference server (aicore)")]
struct Args {
    /// Chat model (GGUF).
    #[arg(short, long, env = "AICORE_MODEL")]
    model: Option<PathBuf>,
    /// Multimodal projector for the chat model (mmproj-*.gguf); enables image and/or audio input.
    #[arg(long, env = "AICORE_MMPROJ")]
    mmproj: Option<PathBuf>,
    /// Embedding model (GGUF).
    #[arg(long, env = "AICORE_EMBEDDING_MODEL")]
    embedding_model: Option<PathBuf>,
    /// Model id reported to clients (default: file name).
    #[arg(long)]
    alias: Option<String>,
    /// Embedding model id reported to clients (default: file name).
    #[arg(long)]
    embedding_alias: Option<String>,
    /// Address to bind. Keep the default (localhost) unless you set --api-key.
    #[arg(long, default_value = "127.0.0.1", env = "AICORE_HOST")]
    host: String,
    #[arg(short, long, default_value_t = 8080, env = "AICORE_PORT")]
    port: u16,
    /// Context size per session (0 = model default, max 8192).
    #[arg(short = 'c', long, default_value_t = 0)]
    ctx: u32,
    /// Concurrent chat sessions; each holds its own KV cache (VRAM).
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// Layers to offload to GPU: all, none, or a number.
    #[arg(long, default_value = "all", value_parser = parse_gpu_layers)]
    gpu_layers: GpuLayers,
    /// Restrict to these devices (repeatable), e.g. --device Vulkan0.
    #[arg(long = "device")]
    devices: Vec<String>,
    /// CPU threads (default: half the hardware threads).
    #[arg(long)]
    threads: Option<i32>,
    /// Max request body size in MiB (base64 images/audio are large).
    #[arg(long, default_value_t = 64)]
    max_body_mb: usize,
    /// Require `Authorization: Bearer <key>`.
    #[arg(long, env = "AICORE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
}

fn main() {
    let args = Args::parse();
    init_logging();

    let config = ServerConfig {
        model: args.model,
        embedding_model: args.embedding_model,
        alias: args.alias,
        embedding_alias: args.embedding_alias,
        mmproj: args.mmproj,
        model_params: ModelParams {
            gpu_layers: args.gpu_layers,
            devices: (!args.devices.is_empty()).then_some(args.devices),
            mmproj: None,
        },
        session: SessionParams { n_ctx: args.ctx, n_threads: args.threads, ..Default::default() },
        parallel: args.parallel,
        api_key: args.api_key,
        max_body_bytes: args.max_body_mb << 20,
    };
    let addr: SocketAddr = match format!("{}:{}", args.host, args.port).parse() {
        Ok(a) => a,
        Err(e) => fail(format!("invalid --host/--port: {e}")),
    };
    if !addr.ip().is_loopback() && config.api_key.is_none() {
        log::warn!("listening on {addr} without --api-key: anyone on the network can use this server");
    }

    let state = AppState::load(&config).unwrap_or_else(|e| fail(format!("startup failed: {e}")));
    let runtime = tokio::runtime::Runtime::new().unwrap_or_else(|e| fail(e.to_string()));
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap_or_else(|e| fail(format!("bind {addr}: {e}")));
        log::info!("aicore-server listening on http://{addr}");
        let shutdown_state = state.clone();
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async move {
                shutdown_signal().await;
                log::info!("shutting down");
                shutdown_state.cancel_all();
            })
            .await
            .unwrap_or_else(|e| fail(e.to_string()));
    });
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}

/// Minimal stderr logger; level from AICORE_LOG (default: info for the server, warn for llama.cpp).
fn init_logging() {
    struct Logger(log::LevelFilter);
    impl log::Log for Logger {
        fn enabled(&self, m: &log::Metadata) -> bool {
            let level = if m.target() == "llama.cpp" { self.0.min(log::LevelFilter::Warn) } else { self.0 };
            m.level() <= level
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("[{} {}] {}", r.level(), r.target(), r.args());
            }
        }
        fn flush(&self) {}
    }
    let level = std::env::var("AICORE_LOG").ok().and_then(|v| v.parse().ok()).unwrap_or(log::LevelFilter::Info);
    if log::set_logger(Box::leak(Box::new(Logger(level)))).is_ok() {
        log::set_max_level(level);
    }
}

fn fail(msg: String) -> ! {
    eprintln!("aicore-server: {msg}");
    std::process::exit(1)
}
