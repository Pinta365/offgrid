# @pinta365/offgrid

Run AI models locally from Deno: no server, no cloud, no API keys. offgrid loads GGUF models with
[llama.cpp](https://github.com/ggml-org/llama.cpp) inside your process and runs them on any GPU through Vulkan, or on
the CPU.

- **Chat** with streaming, cancellation and KV-cache reuse across turns
- **Tool calling** with OpenAI-style `tools` and a ready-made call/execute loop (`runTools`)
- **Structured output**: JSON that always matches your JSON Schema
- **Reasoning models**: `<think>` text streamed separately
- **Images and audio** as input, with multimodal models
- **Embeddings** for search and similarity
- **Image generation**: text-to-image and image-to-image (Z-Image, FLUX, Stable Diffusion)
- OpenAI-compatible request and response shapes throughout

This is the Deno binding of [offgrid](https://github.com/Pinta365/offgrid), which is also available for Rust, C and over
HTTP.

## Quick start

```sh
deno add jsr:@pinta365/offgrid
```

```ts
import { Offgrid } from "@pinta365/offgrid";

using core = await Offgrid.load(); // finds or downloads the native library (see below)
using model = await core.loadModel("Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
using chat = await model.createChat();

const res = await chat.complete(
  { messages: [{ role: "user", content: "Name three rivers." }], max_tokens: 200 },
  { onToken: (t) => void Deno.stdout.writeSync(new TextEncoder().encode(t)) },
);
console.log(res.finish_reason, res.usage);
```

```sh
deno run --allow-ffi --allow-env --allow-read --allow-write --allow-net=github.com,release-assets.githubusercontent.com app.ts
```

**Models** are GGUF files you provide; nothing is downloaded for you. Good starting points on Hugging Face:
[Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) (Q4_K_M, 2.5 GB) for chat and tools,
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) for embeddings, and
[Qwen2.5-VL-3B](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF) (with its `mmproj` file) for images.

> **Just published?** Recent Deno versions refuse package versions younger than 24 hours by default. To use a new
> release right away, pass `--minimum-dependency-age=0` (or set `"minimumDependencyAge"` in `deno.json`).

## The native library

The package is TypeScript; the inference engine is a native library (about 20–26 MB with its GPU and CPU backends).
`Offgrid.load()` looks for it in this order and uses the first one found:

1. The `path` argument: `Offgrid.load("/opt/myapp/offgrid/lib/liboffgrid.so")`.
2. The `OFFGRID_LIB` environment variable.
3. A build in the offgrid repository (`target/release` or `target/debug`), when the binding is used from a checkout.
4. The download cache.
5. **Otherwise it downloads it**, once:
   - It fetches the prebuilt package for your platform from the
     [GitHub release](https://github.com/Pinta365/offgrid/releases) with the same version as this package
     (`offgrid-<version>-linux-x86_64.tar.gz` or `offgrid-<version>-windows-x86_64.tar.gz`).
   - It checks the archive's SHA-256 against the checksum published inside this package, and refuses it if it doesn't
     match.
   - It unpacks it into the cache and loads it from there. Later runs use the cached copy and need no network.

Only the library is ever downloaded. Models are never downloaded; you pass paths to GGUF files you provide.

**Cache location** (override with `OFFGRID_CACHE`):

| OS      | Default                                            |
| ------- | -------------------------------------------------- |
| Linux   | `$XDG_CACHE_HOME/offgrid`, else `~/.cache/offgrid` |
| Windows | `%LOCALAPPDATA%\offgrid`                           |

Each version gets its own folder (`<cache>/<version>/<platform>/`); deleting it forces a fresh download.

**Supported platforms:** Linux x86_64 (glibc 2.28+, e.g. Ubuntu 20.04+, Debian 10+, RHEL 8+) and Windows x86_64. On
other platforms, build the library from source and point `OFFGRID_LIB` at it.

**Windows: the library isn't code-signed yet.** With Windows 11's Smart App Control turned on (the default on clean
installs), loading `offgrid.dll` fails because it is unsigned, and there are no per-app exceptions; turn Smart App
Control off under Windows Security → App & browser control (it can be turned back on later since the April 2026 update).
See [Windows: unsigned binaries](https://github.com/Pinta365/offgrid#windows-unsigned-binaries).

### Permissions

|                 | First run (download)                              | Later runs, or with `OFFGRID_LIB` |
| --------------- | ------------------------------------------------- | --------------------------------- |
| `--allow-ffi`   | yes                                               | yes                               |
| `--allow-env`   | yes (library and cache lookup)                    | yes                               |
| `--allow-read`  | yes                                               | yes                               |
| `--allow-write` | yes (the cache)                                   | no                                |
| `--allow-net`   | `github.com,release-assets.githubusercontent.com` | no                                |

### Offline and bundled apps

`Offgrid.open()` is the same lookup **without** step 5: it never touches the network, and falls back to the system
library search path. For an app that must run offline from the start (or a `deno compile` binary), ship the unpacked
release package next to it and point at it:

```ts
import { dirname, join } from "jsr:@std/path";

using core = Offgrid.open(join(dirname(Deno.execPath()), "offgrid", "lib", "liboffgrid.so"));
```

or set `OFFGRID_LIB` (on Windows: `...\offgrid\bin\offgrid.dll`). Keep the package folder together: the library loads
its GPU and CPU backends from the files next to it.

To fill the cache ahead of time, for example in a setup or install step, call `await downloadLibrary()`.

## Usage

**Streaming** with an `AsyncGenerator`; breaking out of the loop cancels generation:

```ts
for await (const piece of chat.stream({ messages: [{ role: "user", content: "Tell me a story." }] })) {
  Deno.stdout.writeSync(new TextEncoder().encode(piece));
}
```

**Cancellation**: pass `signal` (an `AbortSignal`) to `complete` or `stream`, or call `chat.cancel()`.

**Reasoning models** (Qwen3 thinking mode, ...): `onReasoning` streams the `<think>` text; the response has it as
`reasoning_content`.

**Structured output**: `response_format: { type: "json_schema", json_schema: { schema } }` makes the output valid JSON
for the schema.

**Tool calling** with OpenAI-style `tools`; `runTools` runs the call, execute, answer loop:

```ts
const { response } = await chat.runTools(
  {
    messages: [{ role: "user", content: "What is 1234 * 5678?" }],
    tools: [{
      type: "function",
      function: {
        name: "multiply",
        parameters: { type: "object", properties: { a: { type: "integer" }, b: { type: "integer" } } },
      },
    }],
  },
  { multiply: ({ a, b }) => ({ product: (a as number) * (b as number) }) },
);
```

**Images and audio** need the model's projector (`mmproj-*.gguf`):

```ts
import { imagePart } from "@pinta365/offgrid";

using vl = await core.loadModel("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", { mmproj: "mmproj-Qwen2.5-VL-3B.gguf" });
using vchat = await vl.createChat();
const r = await vchat.complete({
  messages: [{
    role: "user",
    content: [{ type: "text", text: "What's in this picture?" }, await imagePart("photo.jpg")],
  }],
});
```

`audioPart()` does the same for WAV, MP3 and FLAC with an audio-capable model.

**Embeddings**:

```ts
import { dot } from "@pinta365/offgrid";

using embedModel = await core.loadModel("Qwen3-Embedding-0.6B-Q8_0.gguf");
using emb = await embedModel.createEmbedder();
const [a, b] = await emb.embed(["The cat sits on the mat.", "A kitten rests on a rug."]); // normalized
console.log(dot(a, b)); // cosine similarity
```

**Image generation** (Z-Image, FLUX, SD 1.x/2.x/XL, ...; the downloaded library includes it):

```ts
using painter = await core.loadImageModel({
  diffusion_model: "z_image_turbo-Q4_K.gguf",
  text_encoder: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
  vae: "ae-f16.gguf",
});
const { png } = await painter.generate(
  { prompt: "A lighthouse at dusk", width: 768, height: 512, steps: 8, cfg_scale: 1 },
  { onProgress: (p) => console.log(p.phase, p.done, p.total) },
);
Deno.writeFileSync("lighthouse.png", png);

// Image-to-image: start from an existing PNG; `strength` 0 keeps it, 1 ignores it.
const refined = await painter.generate({ prompt: "...", steps: 8, cfg_scale: 1, init_image: png, strength: 0.85 });
```

One image generates at a time; `signal` or `onProgress` returning `false` cancels. Z-Image-Turbo needs about 6 GB of GPU
memory. Next to a chat model, `vae_tiling: true` keeps its weights resident between images; on small GPUs add
`text_encoder_on_cpu: true` (saves 2.4 GB) or `max_vram_gib`.

**GPU**: Vulkan is used automatically when available, with a fallback to the CPU. `core.devices()` lists the devices;
`loadModel(path, { gpu_layers: "none" })` forces the CPU and `{ devices: ["Vulkan0"] }` picks a GPU. Logging:
`OFFGRID_LOG=debug`.

## Rules

- One request at a time per `ChatSession` or `Embedder`; a second concurrent call throws "busy". Create several sessions
  for parallel work.
- Handles are `Disposable` (`using`). A session keeps its model alive natively.
- `free()` throws while a request is running; cancel it and await it first.
- Calls run off the JS thread (`nonblocking` FFI), so they never block the event loop.
