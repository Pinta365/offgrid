# offgrid Tauri example

A desktop chat app that embeds offgrid as a Rust dependency. There's no sidecar process, no
FFI and no network access: the model runs inside the app process.

- `src-tauri/src/main.rs`: loads the model on a background thread at startup and exposes three
  commands: `status`, `chat` (streams reply text and reasoning over a `Channel`) and `cancel`.
- `ui/`: plain HTML, CSS and JavaScript with no build step. The frontend keeps the conversation
  and sends all of it on every turn; the session's KV cache reuses the shared prefix, so only
  the new tokens are processed.

## Requirements

- The offgrid build requirements from the main README (Rust, cmake, ninja, Vulkan headers and
  `glslc`).
- Tauri's system dependencies. On Linux these are `webkit2gtk-4.1` and its dev files; see
  <https://v2.tauri.app/start/prerequisites/>.
- The Tauri CLI only for bundling (`cargo install tauri-cli --version '^2'`). `cargo run` is
  enough during development.

## Run

```sh
cd examples/tauri/src-tauri
OFFGRID_MODEL=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf cargo run --release
```

The first build compiles llama.cpp, which takes a few minutes.

## Bundling the model

Without `OFFGRID_MODEL`, the app loads the first `.gguf` file in its `models/` resource folder
(skipping `mmproj-*` files). To ship a model with the app, put it in `src-tauri/models/` and
build a package:

```sh
cp Qwen3-4B-Instruct-2507-Q4_K_M.gguf src-tauri/models/
cd src-tauri && cargo tauri build
```

`tauri.conf.json` maps `models/` into the bundle's resources, so the model ends up next to the
app in the `.deb`, `.rpm`, AppImage or Windows installer. Tauri copies the resources on every
build, so use `OFFGRID_MODEL` during development instead of a large file in `models/`.

## Notes

- offgrid is built here with its default features: llama.cpp is linked statically, with the
  Vulkan backend. The app needs `libvulkan` at runtime (part of every GPU driver) and uses the
  CPU when no GPU is available. The static CPU code targets a baseline x86-64, so it runs
  anywhere but is not the fastest. `OFFGRID_NATIVE=1` tunes it for the build machine (for
  local use only). offgrid's `dynamic-backends` layout picks the best CPU variant at runtime
  (see the main README).
- `chat` takes OpenAI-format messages and returns offgrid's `ChatResponse`, so adding JSON
  output, tools or images means adding fields to the request; see `ChatRequest` in
  `crates/offgrid/src/chat.rs`.
- Generation runs on `spawn_blocking`. One session serves one reply at a time; a second
  request waits for the first one to finish.
