# offgrid C examples

Small programs exercising the `offgrid` C ABI in `crates/offgrid-ffi/include/offgrid.h`.

## Prerequisites

Build the shared/static library first (from the repo root):

```sh
cargo build --release -p offgrid-ffi
```

This produces `target/release/liboffgrid.so` and `target/release/liboffgrid.a`.

## Build

```sh
make            # builds chat, embed, cancel (dynamic, -loffgrid)
make static     # builds chat-static (links liboffgrid.a)
make clean      # removes the built binaries
```

The dynamic binaries use an rpath of `$ORIGIN/../../target/release`, so they find
`liboffgrid.so` without setting `LD_LIBRARY_PATH`.

## Run

```sh
# Streaming chat: version/devices JSON on stdout, streamed reply on stdout,
# final response JSON on stderr.
./chat ../../spike/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf "In one sentence, what is a llama?"

# Embeddings: prints the dimension and pairwise cosine similarity.
./embed ../../spike/models/Qwen3-Embedding-0.6B-Q8_0.gguf

# Cancellation: starts a long completion, cancels after ~1s, checks that
# finish_reason == "cancelled". Exit 0 on success.
./cancel ../../spike/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

# Statically linked chat.
./chat-static ../../spike/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf "hi"

# Or run chat + embed + cancel with the default model paths:
make run
```

Override model paths for `make run` with `MODEL=` / `EMBED_MODEL=`.

## Files

| File | Purpose |
|---|---|
| `chat.c` | Streaming chat with a system + user message; JSON-escapes the question. |
| `embed.c` | Embeds three fixed sentences via `offgrid_embed` (float buffer) and prints pairwise cosine. |
| `cancel.c` | Cancels an in-flight completion from another thread via `offgrid_chat_cancel`. |
| `Makefile` | `all` / `static` / `run` / `clean`. |

## Notes

- Error handling uses the trailing `char **error_out` parameter on fallible calls;
  the returned message is owned by the caller and freed with `offgrid_string_free`.
- All returned strings (`offgrid_version_json`, `offgrid_devices_json`, response JSON)
  are freed with `offgrid_string_free`.
- `OFFGRID_LOG=4` (or `offgrid_set_log_callback`) enables llama.cpp logging if needed.
- valgrind was not installed in this environment, so no valgrind run was performed.
