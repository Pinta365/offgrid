//! offgrid-server: OpenAI-compatible local inference sidecar.
//!
//!   offgrid-server --model qwen3-4b.gguf --embedding-model embed.gguf --port 8080
//!   offgrid-server --fim-model qwen2.5-coder-1.5b.gguf   # code completion only

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;

use clap::Parser;
use offgrid::{GpuLayers, ModelParams, SessionParams};
use offgrid_server::{AppState, ServerConfig, parse_gpu_layers, router};

#[derive(Parser, Debug)]
#[command(version, about = "OpenAI-compatible local inference server (offgrid)")]
struct Args {
    /// Chat model (GGUF).
    #[arg(short, long, env = "OFFGRID_MODEL")]
    model: Option<PathBuf>,
    /// Multimodal projector for the chat model (mmproj-*.gguf); enables image and/or audio input.
    #[arg(long, env = "OFFGRID_MMPROJ")]
    mmproj: Option<PathBuf>,
    /// Embedding model (GGUF).
    #[arg(long, env = "OFFGRID_EMBEDDING_MODEL")]
    embedding_model: Option<PathBuf>,
    /// Code model (GGUF with FIM tokens, e.g. Qwen2.5-Coder) for /infill and /v1/completions.
    #[arg(long, env = "OFFGRID_FIM_MODEL")]
    fim_model: Option<PathBuf>,
    /// Model id reported to clients (default: file name).
    #[arg(long)]
    alias: Option<String>,
    /// Embedding model id reported to clients (default: file name).
    #[arg(long)]
    embedding_alias: Option<String>,
    /// FIM model id reported to clients (default: file name).
    #[arg(long)]
    fim_alias: Option<String>,
    /// Context size per FIM session (0 = same as --ctx). Code completion rarely needs more
    /// than 8192, so a smaller value saves memory next to a large chat context.
    #[arg(long, default_value_t = 0)]
    fim_ctx: u32,
    /// Shut down when stdin closes. A parent process (e.g. an editor extension) keeps a pipe
    /// open, so the server exits with it even if it crashes.
    #[arg(long)]
    exit_on_stdin_close: bool,
    /// Address to bind. Keep the default (localhost) unless you set --api-key.
    #[arg(long, default_value = "127.0.0.1", env = "OFFGRID_HOST")]
    host: String,
    #[arg(short, long, default_value_t = 8080, env = "OFFGRID_PORT")]
    port: u16,
    /// Context size per session (0 = model default, max 8192).
    #[arg(short = 'c', long, default_value_t = 0)]
    ctx: u32,
    /// Reuse cached chunks of at least this many tokens that moved to an earlier position
    /// (shift their KV entries instead of evaluating them again); 0 = off, as in llama-server.
    /// Editor plugins such as llama.vscode recommend 256.
    #[arg(long, default_value_t = 0)]
    cache_reuse: u32,
    /// KV cache type: f16, q8_0 (half the memory) or q4_0 (a quarter). Quantized types need a
    /// device with flash attention (Vulkan, CUDA, CPU).
    #[arg(long, default_value = "f16")]
    cache_type: offgrid::CacheType,
    /// Prompt tokens per decode call. Larger is faster for long prompts, especially with
    /// --n-cpu-moe (expert weights are copied to the GPU once per batch), and uses more memory.
    #[arg(long, default_value_t = 512)]
    batch: u32,
    /// Concurrent chat sessions; each holds its own KV cache (VRAM).
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// Layers to offload to GPU: all, none, or a number.
    #[arg(long, default_value = "all", value_parser = parse_gpu_layers)]
    gpu_layers: GpuLayers,
    /// Keep the MoE expert weights of the first N layers on the CPU, attention on the GPU
    /// (llama.cpp's --n-cpu-moe): for MoE models that don't fit the GPU. 0 = off.
    #[arg(long, default_value_t = 0)]
    n_cpu_moe: u32,
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
    #[arg(long, env = "OFFGRID_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Image diffusion model (Z-Image, FLUX, ...) for /v1/images/generations.
    #[cfg(feature = "image")]
    #[arg(long)]
    image_diffusion_model: Option<PathBuf>,
    /// Single-file image checkpoint (SD 1.x/2.x/XL), instead of --image-diffusion-model.
    #[cfg(feature = "image")]
    #[arg(long)]
    image_model: Option<PathBuf>,
    /// Text encoder for the image model (Z-Image: a Qwen3-4B GGUF).
    #[cfg(feature = "image")]
    #[arg(long)]
    image_text_encoder: Option<PathBuf>,
    /// VAE for the image model.
    #[cfg(feature = "image")]
    #[arg(long)]
    image_vae: Option<PathBuf>,
    /// GiB of GPU memory the image model may use; the rest of its weights stay in RAM.
    #[cfg(feature = "image")]
    #[arg(long)]
    image_max_vram: Option<f32>,
    /// Run the image model on the CPU.
    #[cfg(feature = "image")]
    #[arg(long)]
    image_cpu: bool,
    /// Run the image model's text encoder on the CPU (keeps its weights off the GPU).
    #[cfg(feature = "image")]
    #[arg(long)]
    image_text_encoder_cpu: bool,
    /// Decode images in tiles: a smaller GPU memory peak, so weights stay resident next to a chat model.
    #[cfg(feature = "image")]
    #[arg(long)]
    image_vae_tiling: bool,
    /// Image model id reported to clients (default: file name).
    #[cfg(feature = "image")]
    #[arg(long)]
    image_alias: Option<String>,
}

fn main() {
    let args = Args::parse();
    init_logging();

    let config = ServerConfig {
        model: args.model,
        embedding_model: args.embedding_model,
        fim_model: args.fim_model,
        alias: args.alias,
        embedding_alias: args.embedding_alias,
        fim_alias: args.fim_alias,
        fim_ctx: args.fim_ctx,
        mmproj: args.mmproj,
        model_params: ModelParams {
            gpu_layers: args.gpu_layers,
            devices: (!args.devices.is_empty()).then_some(args.devices),
            mmproj: None,
            n_cpu_moe: args.n_cpu_moe,
        },
        session: SessionParams {
            n_ctx: args.ctx,
            n_threads: args.threads,
            cache_reuse: args.cache_reuse,
            cache_type: args.cache_type,
            n_batch: args.batch,
        },
        parallel: args.parallel,
        api_key: args.api_key,
        max_body_bytes: args.max_body_mb << 20,
        #[cfg(feature = "image")]
        image: (args.image_diffusion_model.is_some() || args.image_model.is_some()).then(|| {
            offgrid::ImageModelParams {
                diffusion_model: args.image_diffusion_model,
                model: args.image_model,
                text_encoder: args.image_text_encoder,
                vae: args.image_vae,
                max_vram_gib: args.image_max_vram,
                gpu: !args.image_cpu,
                text_encoder_on_cpu: args.image_text_encoder_cpu,
                vae_tiling: args.image_vae_tiling,
                ..Default::default()
            }
        }),
        #[cfg(feature = "image")]
        image_alias: args.image_alias,
    };
    // IPv4/IPv6 literals ("::1" or "[::1]") and hostnames ("localhost").
    let host = args.host.trim_start_matches('[').trim_end_matches(']').to_string();
    // A hostname may resolve to both families; prefer IPv4 ("localhost" -> 127.0.0.1), which
    // is what most local clients connect to.
    let pick = |addrs: Vec<SocketAddr>| addrs.iter().find(|a| a.is_ipv4()).or(addrs.first()).copied();
    let addr: SocketAddr = match (host.as_str(), args.port).to_socket_addrs().map(|a| pick(a.collect())) {
        Ok(Some(a)) => a,
        Ok(None) => fail(format!("--host {} resolved to no address", args.host)),
        Err(e) => fail(format!("invalid --host/--port: {e}")),
    };
    if !addr.ip().is_loopback() && config.api_key.is_none() {
        log::warn!("listening on {addr} without --api-key: anyone on the network can use this server");
    }

    for d in offgrid::devices() {
        log::info!("device {} ({:?}): {}", d.name, d.kind, d.description);
    }
    let state = AppState::load(&config).unwrap_or_else(|e| fail(format!("startup failed: {e}")));
    let runtime = tokio::runtime::Runtime::new().unwrap_or_else(|e| fail(e.to_string()));
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap_or_else(|e| fail(format!("bind {addr}: {e}")));
        log::info!("offgrid-server listening on http://{addr}");
        let shutdown_state = state.clone();
        let stdin_closed = args.exit_on_stdin_close.then(stdin_closed);
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async move {
                match stdin_closed {
                    Some(closed) => tokio::select! {
                        _ = shutdown_signal() => {},
                        _ = closed => log::info!("stdin closed"),
                    },
                    None => shutdown_signal().await,
                }
                log::info!("shutting down (Ctrl-C again to exit immediately)");
                shutdown_state.shutdown();
                tokio::spawn(async {
                    shutdown_signal().await;
                    log::warn!("second signal: exiting now");
                    std::process::exit(130);
                });
            })
            .await
            .unwrap_or_else(|e| fail(e.to_string()));
    });
}

/// Resolves when stdin reaches end-of-file (or fails): the parent closed the pipe or exited.
fn stdin_closed() -> tokio::sync::oneshot::Receiver<()> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = [0u8; 256];
        let mut stdin = std::io::stdin();
        while matches!(stdin.read(&mut buf), Ok(n) if n > 0) {}
        let _ = tx.send(());
    });
    rx
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

/// Minimal stderr logger with seconds since start. OFFGRID_LOG sets the level for the server
/// and llama.cpp; unset, the server logs info and llama.cpp warnings.
fn init_logging() {
    struct Logger {
        level: log::LevelFilter,
        llama: log::LevelFilter,
        start: std::time::Instant,
    }
    impl log::Log for Logger {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= if m.target() == "llama.cpp" { self.llama } else { self.level }
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                let t = self.start.elapsed().as_secs_f64();
                eprintln!("[{t:7.2}s {} {}] {}", r.level(), r.target(), r.args());
            }
        }
        fn flush(&self) {}
    }
    let env = std::env::var("OFFGRID_LOG").ok().and_then(|v| v.parse().ok());
    let level = env.unwrap_or(log::LevelFilter::Info);
    let llama = env.unwrap_or(log::LevelFilter::Warn);
    let logger = Logger { level, llama, start: std::time::Instant::now() };
    if log::set_logger(Box::leak(Box::new(logger))).is_ok() {
        log::set_max_level(level.max(llama));
    }
}

fn fail(msg: String) -> ! {
    eprintln!("offgrid-server: {msg}");
    std::process::exit(1)
}
