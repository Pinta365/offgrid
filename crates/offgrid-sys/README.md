# offgrid-sys

Raw bindgen bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp) and mtmd, built
from the pinned `vendor/llama.cpp` submodule (always in CMake `Release`). The safe wrapper is
the `offgrid` crate.

## Features

| Feature | Default | Effect |
|---|---|---|
| `vulkan` | yes | Enables the GGML Vulkan backend (`GGML_VULKAN=ON`) and links `vulkan`. |
| `cuda` | no | Enables the GGML CUDA backend (`GGML_CUDA=ON`). Requires a CUDA toolkit. |
| `dynamic-backends` | no | Shared libs with runtime-loaded backends and all CPU variants (used by the packages). |

CPU-only builds: `cargo build -p offgrid-sys --no-default-features`.

## Environment variables

| Variable | Effect |
|---|---|
| `OFFGRID_NATIVE=1` | Sets `GGML_NATIVE=ON` (tune the CPU backend for the host CPU). Off by default so artifacts are portable. |
| `OFFGRID_TEST_MODEL` | GGUF model for the `tokenize_with_model` smoke test (skipped when unset). |

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
git add vendor/llama.cpp
cargo clean -p offgrid-sys && cargo build -p offgrid-sys
```
