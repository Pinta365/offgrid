# offgrid-sys

Low-level, unsafe FFI bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp),
built from source and linked **statically** into the consuming crate.

- Generated with `bindgen` from `vendor/llama.cpp/include/llama.h`.
- Covers `llama_*`, `ggml_backend_*` and `ggml_log_*` functions, `llama_*`/`ggml_*`
  types and `LLAMA_*` constants.
- llama.cpp is always compiled with CMake in `Release`, even for cargo debug builds,
  so incremental development stays fast.
- Built against the commit pinned in the `vendor/llama.cpp` submodule.

This crate intentionally exposes raw FFI only; the safe wrapper lives in the
`offgrid` crate.

## Features

| Feature | Default | Effect |
|---|---|---|
| `vulkan` | yes | Enables the GGML Vulkan backend (`GGML_VULKAN=ON`) and links `vulkan`. |
| `cuda` | no | Enables the GGML CUDA backend (`GGML_CUDA=ON`). Requires a CUDA toolkit. |

CPU-only builds: `cargo build -p offgrid-sys --no-default-features`.

## Environment variables

| Variable | Effect |
|---|---|
| `OFFGRID_NATIVE=1` | Sets `GGML_NATIVE=ON` (tune the CPU backend for the host CPU). Off by default so artifacts are portable. |
| `OFFGRID_TEST_MODEL` | Path to a GGUF model. When set, the `tokenize_with_model` smoke test loads it (CPU, `n_gpu_layers=0`), tokenizes `"Hello world"` and asserts a non-empty result. When unset the test prints a skip message. |

## Linking

Static archives linked, in dependency order:
`llama`, `ggml`, `ggml-vulkan` (with `vulkan`), `ggml-cuda` (with `cuda`),
`ggml-cpu`, `ggml-base`, plus system libraries (`stdc++`, `m`, `pthread`, `dl`,
and `vulkan` on Linux when the `vulkan` feature is on).

## Tests

```bash
# Device enumeration only
cargo test -p offgrid-sys -- --nocapture

# With model tokenization
OFFGRID_TEST_MODEL=/path/to/model.gguf cargo test -p offgrid-sys -- --nocapture

# CPU-only build (no Vulkan)
cargo build -p offgrid-sys --no-default-features
```

## Bumping the pinned llama.cpp commit

```bash
cd vendor/llama.cpp
git fetch --depth 1 origin <new-commit-sha>
git checkout <new-commit-sha>
cd ../..
git add vendor/llama.cpp        # records the new gitlink in the index
cargo clean -p offgrid-sys && cargo build -p offgrid-sys
```

Because the bindings are generated at build time and the build script tracks
`include/llama.h`, a new commit is picked up automatically on the next build.
