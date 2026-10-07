# aicore — embeddable offline AI

A small native library that gives any application local, offline LLM chat and
embeddings. No daemon, no Docker, no network: link a library, ship a GGUF model.

Built in Rust on top of [llama.cpp](https://github.com/ggml-org/llama.cpp)
(statically linked, Vulkan GPU backend + CPU fallback).

```
   Rust/Tauri ──── aicore crate ─────┐
   C / C++ ───┐                      │
   Deno ──────┼── C ABI (libaicore) ─┤
   Python/Go/C# ┘   JSON in/out      │
                              aicore (safe Rust API)
                                     │
                         aicore-sys ─ llama.cpp (vendor/, pinned)
                                     └─ Vulkan · CPU · (CUDA)
```

## Layout

| Path | What |
|---|---|
| `crates/aicore-sys` | Raw bindgen bindings; builds `vendor/llama.cpp` statically with CMake |
| `crates/aicore` | Safe Rust API: `Model`, `ChatSession`, `Embedder`, `devices()` |
| `crates/aicore-ffi` | C ABI → `libaicore.so` / `aicore.dll` / `libaicore.a`, header in `include/aicore.h` |
| `bindings/deno` | Deno FFI bindings (async, streaming, AbortSignal) |
| `examples/c`, `examples/deno` | Runnable examples |
| `spike/` | Phase 0 notes and benchmarks |

## Quick start

Requirements (Linux): Rust, cmake, ninja, a C++ compiler, Vulkan headers + `glslc`
(`omarchy pkg add cmake ninja vulkan-headers spirv-headers shaderc`). Then:

```sh
git submodule update --init
cargo build --release -p aicore-ffi        # → target/release/libaicore.{so,a}

# Rust
cargo run --release -p aicore --example chat -- model.gguf
# C
make -C examples/c && examples/c/chat model.gguf "Hello"
# Deno
deno run --allow-ffi --allow-env --allow-read examples/deno/chat.ts model.gguf
```

CPU-only build: add `--no-default-features`. CUDA: `--features cuda`.

## API at a glance

- **Model**: `load(path, {gpu_layers, devices})`, `info()` (name, size, context, template).
- **ChatSession**: stateless OpenAI-style requests (`messages`, `max_tokens`, `stop`,
  `temperature`, `top_p`, …, `grammar`), streamed through a callback; reuses the KV-cache
  prefix between calls so follow-up turns only process new tokens. Cancel from any thread.
- **Embedder**: batched, L2-normalized vectors.

The C ABI passes parameters and results as JSON; see `crates/aicore-ffi/include/aicore.h`
for conventions (ownership, `error_out`, thread rules). ABI version: `aicore_abi_version()`.

## Tests

```sh
export AICORE_TEST_MODEL=/path/chat-model.gguf AICORE_TEST_EMBED_MODEL=/path/embed-model.gguf
cargo test --workspace
(cd bindings/deno && deno task test)
```

## Status

Phase 0 (spike) ✔ · Phase 1 (core crate) ✔ · Phase 2 (C ABI, C + Deno examples) ✔

Next: runtime-loaded GPU backends for portable binaries, Windows build, Jinja chat
templates (minijinja) + JSON-schema output, optional OpenAI-compatible sidecar server.
