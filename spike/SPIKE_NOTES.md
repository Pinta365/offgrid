# Phase 0 Spike Notes — Embeddable Local-AI Core

## Environment

- Host: Linux (Omarchy / Arch-based)
- GPU: AMD Radeon RX 9070 XT (Navi 48, RADV `GFX1201`), plus integrated Radeon 9800X3D iGPU
- Vulkan: `vulkaninfo` reports loader/ICD API 1.4.357; `glslc` and `glslangValidator` present
- Build tools: cmake, ninja, glslc already installed (no sudo used, no system packages installed)
- CMake option `LLAMA_CURL=OFF` was accepted as-is (no changes needed to `docs/build.md` instructions)

## llama.cpp

- Repository: https://github.com/ggml-org/llama.cpp
- Clone: `git clone --depth 1 https://github.com/ggml-org/llama.cpp spike/llama.cpp`
- **Commit hash: `78651c410dd8d97e3e22e533e0e7117889e3863a`** (short `78651c4`)
  - Commit subject: `sycl: add IQ3_S multi-column MMVQ (#29500)`
  - Commit date: 2026-10-07 13:28:23 +0800
- Reported versions: llama.cpp `0.6.0-dev`, ggml `0.26.0`, ggml commit `78651c4`

### Configure command (exact, accepted without modification)

```bash
cmake -S spike/llama.cpp -B spike/llama.cpp/build -G Ninja \
  -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
```

### Build command

```bash
cmake --build spike/llama.cpp/build -j 8
```

### Build result

- Success: `[760/760] Linking CXX executable bin/test-backend-ops`
- Vulkan backend detected:
  - `-- Found Vulkan: /usr/lib/libvulkan.so (found version "1.4.357") found components: glslc glslangValidator`
  - `-- GL_KHR_cooperative_matrix supported by glslc`
  - `-- GL_EXT_integer_dot_product supported by glslc`
  - `-- Including Vulkan backend`
- Binaries produced: `build/bin/llama-cli`, `build/bin/llama-server`, `build/bin/llama-bench`, etc.

### Build problems and fixes

- No blocking problems; build required no option adjustments.
- Non-fatal warnings only, in test code (`tests/test-backend-ops.cpp`): `-Wdeprecated-enum-enum-conversion` on bitwise ops between `ggml_scale_mode` and `ggml_scale_flag`. No action taken.
- Cosmetic: `Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF` (left as default).

## Models

Downloaded to `spike/models/`. Both verified as real GGUF binaries (first 4 bytes == `GGUF`, checked with `od`).

| File | Size (bytes) | Size | Source URL |
|---|---|---|---|
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2,497,281,120 | ~2.4 GiB | https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639,150,592 | ~610 MiB | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf |

Notes:
- The official `Qwen/Qwen3-4B-Instruct-2507-GGUF` Hugging Face repo is gated (API returned `Invalid username or password`), so the `unsloth` mirror was used for the chat model.
- The embedding model comes from the official `Qwen` repo. `Q8_0` chosen over `f16` for a smaller footprint with negligible quality loss.
- Download command form: `curl -L --fail -C - -o <file> <url>` (resume-capable).

## Device list output

Command: `spike/llama.cpp/build/bin/llama-cli --list-devices`

```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
Available devices:
  Vulkan0: AMD Radeon RX 9070 XT (RADV GFX1201) (16304 MiB, 14477 MiB free)
  Vulkan1: AMD Ryzen 7 9800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO) (32990 MiB, 32961 MiB free)
```

**Confirmed: the discrete RX 9070 XT is exposed as `Vulkan0` (16 GB VRAM, ~14.5 GB free).** The integrated GPU appears as `Vulkan1`.

## Next steps (not performed — benchmarks deferred to user)

- Run benchmarks against `Vulkan0` (e.g. `llama-bench -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`)
- Validate chat inference and embeddings (e.g. `llama-server` with `--embeddings`)

## Benchmarks (run by lead, 2026-10-07)

`llama-bench -p 512 -n 128`, Qwen3-4B-Instruct-2507 Q4_K_M (2.32 GiB):

| Device | Prompt processing (pp512) | Generation (tg128) |
|---|---:|---:|
| Vulkan0 — RX 9070 XT, all layers offloaded | ~5,590 t/s | ~175 t/s |
| CPU only — Ryzen 7 9800X3D, 8 threads | ~311 t/s | ~20 t/s |

## Functional checks

- Chat: `llama-completion` with ChatML prompt on Vulkan0 produced a coherent answer.
- Embeddings: Qwen3-Embedding-0.6B (last-token pooling, L2-normalized) → 1024 dims.
  Cosine: cat/kitten 0.678, cat/tax 0.237, kitten/tax 0.169 — ranks semantically as expected.

## Takeaways for the core

- Vulkan on RADV works out of the box; it is the right default GPU path for AMD (and works on NVIDIA/Intel too).
- CPU fallback at ~20 t/s for a 4B Q4 model is usable for small/background tasks.
- Default build links backends statically into libggml; for shipping one binary across machines,
  build with `GGML_BACKEND_DL=ON` (+ `GGML_CPU_ALL_VARIANTS=ON`) so CPU/Vulkan/CUDA backends load at runtime.
- Must call the model's chat template (llama.cpp has `llama_chat_apply_template` / jinja) rather than hand-writing ChatML.

## Tool-calling test models

Additional model used to exercise the tool-call output parsers
(`crates/aicore/src/tool_parse.rs`) and chat-template tool conventions.

| File | Size (bytes) | Source URL |
|---|---|---|
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 2,019,377,696 (~1.88 GiB) | https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf |

Verified: first 4 bytes are `GGUF` (not an HTML error page). Source repo
`bartowski/Llama-3.2-3B-Instruct-GGUF` (unsloth mirror not needed).
| `Qwen3-0.6B-Q8_0.gguf` | 639,446,688 (~609.8 MiB) | https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf |

`Qwen3-0.6B-Q8_0.gguf` is a hybrid thinking model (thinking/non-thinking modes),
used for reasoning streaming tests. Source repo `Qwen/Qwen3-0.6B-GGUF`.

## Vision test models

Multimodal models with their `mmproj` projectors, for image-input tests. All
verified: first 4 bytes are `GGUF`. Source repos are `ggml-org` (not gated).

| File | Size (bytes) | Source URL |
|---|---|---|
| `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` | 1,929,901,056 | https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf |
| `mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf` | 1,338,428,128 | https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf |
| `gemma-3-4b-it-Q4_K_M.gguf` | 2,489,757,856 | https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf |
| `mmproj-model-f16.gguf` | 851,251,104 | https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf |

Deterministic vision test images in `spike/images/` (generated with ImageMagick):

| File | Dims | Size (bytes) | Content |
|---|---|---|---|
| `red_square.png` | 256x256 | 316 | solid red #FF0000 |
| `three_circles.png` | 384x256 | 4268 | white bg, three solid blue #0000FF circles r=40 |
| `text_hello.png` | 512x160 | 3010 | white bg, large black "HELLO 42" |
| `stripes.jpg` | 320x240 | 1397 | left half green #00FF00, right half yellow #FFFF00 |

## Audio test models

Speech/audio multimodal models with their `mmproj` projectors. All verified:
first 4 bytes are `GGUF`. Source repos are `ggml-org`.

| File | Size (bytes) | Source URL |
|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807,694,464 | https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf |
| `mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf` | 1,371,123,616 | https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf |
| `Qwen2.5-Omni-3B-Q4_K_M.gguf` | 2,104,931,648 | https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/main/Qwen2.5-Omni-3B-Q4_K_M.gguf |
| `mmproj-Qwen2.5-Omni-3B-f16.gguf` | 2,623,983,328 | https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/main/mmproj-Qwen2.5-Omni-3B-f16.gguf |

Audio assets in `spike/audio/` (see `spike/audio/README.md`): `jfk.wav`,
`jfk.mp3`, `jfk.flac`, `tone.wav`.
