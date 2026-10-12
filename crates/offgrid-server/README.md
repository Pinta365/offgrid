# offgrid-server

OpenAI-compatible HTTP server for [offgrid](https://github.com/Pinta365/offgrid): local, offline chat,
tool calling, structured output, embeddings, and image and audio input on top of llama.cpp (Vulkan or CPU).

```sh
cargo install offgrid-server
offgrid-server --model qwen3-4b.gguf [--embedding-model embed.gguf] [--mmproj mmproj.gguf] [--port 8080] [--api-key KEY]
```

Point any OpenAI client at `http://127.0.0.1:8080/v1`. Endpoints: `POST /v1/chat/completions` (with SSE
streaming, `reasoning_content` and `tool_calls` deltas), `POST /v1/embeddings`, `GET /v1/models`, `GET /health`.
The prebuilt packages also serve `POST /v1/images/generations` (`--image-diffusion-model` /
`--image-text-encoder` / `--image-vae`); from source that needs the `image` feature and offgrid from git.

Building compiles llama.cpp from source, which needs CMake, a C/C++ compiler and, with the default `vulkan`
feature, the Vulkan headers and `glslc`. Use `--no-default-features` for a CPU-only build. Prebuilt binaries
are attached to the [GitHub releases](https://github.com/Pinta365/offgrid/releases).

`OFFGRID_LOG=debug` shows llama.cpp's own logs; `offgrid-server --help` lists all options.
