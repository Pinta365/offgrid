# aicore for Deno

FFI bindings to `libaicore` (see the repo root README). Long-running calls use
`nonblocking` FFI, so they return Promises and never block the event loop; tokens
stream back through a thread-safe callback.

```ts
import { AiCore } from "./mod.ts";

using core = AiCore.open(); // $AICORE_LIB, else ../../target/{release,debug}
using model = await core.loadModel("model.gguf", { gpu_layers: "all" });
using chat = await model.createChat({ n_ctx: 8192 });

const res = await chat.complete(
  { messages: [{ role: "user", content: "Hi!" }], max_tokens: 200 },
  { onToken: (t) => void Deno.stdout.writeSync(new TextEncoder().encode(t)), signal: AbortSignal.timeout(10_000) },
);
console.log(res.finish_reason, res.usage);

for await (const piece of chat.stream({ messages: [{ role: "user", content: "Tell me a story" }] })) {
  // breaking out of the loop cancels generation
}

using embedModel = await core.loadModel("embed.gguf");
using emb = await embedModel.createEmbedder();
const [a, b] = await emb.embed(["first text", "second text"]); // Float32Array, normalized
```

Permissions: `--allow-ffi --allow-env --allow-read`.

Rules:
- One request at a time per `ChatSession`/`Embedder` (a second concurrent call throws
  "busy"). Create several sessions for parallel work.
- Handles are `Disposable` (`using`); a session keeps its model alive natively.
- `free()` throws while a request is running; cancel and await it first.

Tests: `AICORE_TEST_MODEL=… AICORE_TEST_EMBED_MODEL=… deno task test`
