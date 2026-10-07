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

## Portable / distributable build

```sh
scripts/package-linux.sh            # → dist/aicore-<ver>-linux-x86_64{/,.tar.gz}
scripts/check-portable.sh dist/aicore-<ver>-linux-x86_64 model.gguf
```

Built with `--features dynamic-backends`: llama.cpp as shared libs with backends loaded
at runtime from the library's own directory (`$AICORE_BACKEND_DIR` overrides):

- 14 CPU variants (x64 → sse42 → … → zen4/sapphirerapids); the best for the running CPU is
  picked at load time, so old CPUs work and new ones are fast.
- Vulkan is an optional module: no GPU, no Vulkan driver, or the module deleted → CPU fallback.
- libstdc++/libgcc are linked statically (symbols hidden); runtime deps are only glibc/libm
  (+ libvulkan for the Vulkan module). All libs use `$ORIGIN` RPATH, so the folder is relocatable.
- Ship the whole `lib/` folder; ~25 MB compressed (the Vulkan module is ~47 MB of shaders).

**glibc baseline:** with `cargo-zigbuild` + zig available (used automatically; zig may come
from `mise install zig@0.16.0`), the package targets **glibc 2.28** — Ubuntu 20.04+, Debian 10+,
RHEL 8+ — using zig's static libc++. Override with `AICORE_GLIBC=2.31`, or force the host
toolchain with `AICORE_TOOLCHAIN=host` (then the build host's glibc is required).
Measured: zig and host builds run at the same speed (GPU prompt ~5,000 tok/s, generation
~150 tok/s; CPU generation ~21 tok/s on a 9800X3D, Qwen3-4B Q4_K_M).

Backend modules are linked `-z nodelete`: ggml dlopens every module to probe it and closes the
unused ones, and unmapping them left dangling exit handlers (crash at process exit, CPU-only).

Note: the default (static) dev build and the dynamic build both write
`target/release/libaicore.so`; after packaging, point consumers at the dist folder
(e.g. `AICORE_LIB=dist/.../lib/libaicore.so`) or rebuild without the feature.

## Windows (cross-compiled from Linux)

```sh
# one-time: llvm-mingw (clang+lld+mingw-w64, UCRT) unpacked into ~/.local/share
#   https://github.com/mstorsjo/llvm-mingw/releases  (ucrt-ubuntu-…-x86_64.tar.xz)
rustup target add x86_64-pc-windows-gnullvm
scripts/package-windows.sh          # → dist/aicore-<ver>-windows-x86_64{/,.zip}
scripts/check-windows.sh dist/aicore-<ver>-windows-x86_64
```

- `bin/`: `aicore.dll` + llama/ggml DLLs, 14 CPU variants, `ggml-vulkan.dll`; keep them together.
  Only Windows system DLLs are needed (UCRT, kernel32; `vulkan-1.dll` from the GPU driver
  for the optional Vulkan module). libc++/libunwind are linked statically.
- Windows doesn't search a DLL's own folder for its dependencies, so `aicore.dll`
  **delay-loads** llama/ggml and preloads them by full path from its own folder. Hosts can
  `LoadLibrary`/`dlopen` it from anywhere (Deno, Python, C#, …).
- `lib/`: `libaicore.dll.a` (MinGW), `aicore.lib` (MSVC), `aicore.def`; `include/aicore.h`.
- Vulkan without an SDK: Vulkan/SPIR-V headers come from the Linux host, the `vulkan-1`
  import library is generated from `crates/aicore-sys/windows/vulkan-1.def`.
- zig is **not** used for Windows: ggml relies on MinGW auto-export, and zig's CRT object
  naming makes lld export `atexit` too (duplicate symbols).

Status: builds and passes static checks (imports, delay-loads, exports, examples link).
**Not yet run on real Windows.**

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

Phase 0 (spike) ✔ · Phase 1 (core crate) ✔ · Phase 2 (C ABI, C + Deno examples) ✔ ·
Phase 3a (portable Linux package, glibc 2.28 via zig) ✔ ·
Phase 3b (Windows cross-build, static-checked; untested on Windows)

Next: run on real Windows, Jinja chat
templates (minijinja) + JSON-schema output, optional OpenAI-compatible sidecar server.
