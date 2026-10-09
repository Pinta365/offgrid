// Interactive streaming chat from Deno.
// cargo build --release -p offgrid-ffi
// deno run --allow-ffi --allow-env --allow-read examples/deno/chat.ts spike/models/<model>.gguf ["one-shot question"]
import { Offgrid, type ChatMessage } from "../../bindings/deno/mod.ts";

const path = Deno.args[0];
if (!path) {
  console.error("usage: chat.ts <model.gguf>");
  Deno.exit(1);
}

using core = Offgrid.open();
console.error("offgrid", core.version());
using model = await core.loadModel(path);
console.error("model:", model.info().description);
using chat = await model.createChat();

const messages: ChatMessage[] = [{ role: "system", content: "You are a helpful assistant." }];
const out = new TextEncoder();

// Ctrl+C stops the current answer instead of quitting.
let current: AbortController | null = null;
Deno.addSignalListener("SIGINT", () => current ? current.abort() : Deno.exit(0));

const oneShot = Deno.args[1];
while (true) {
  const line = oneShot ?? prompt("\n>");
  if (line === null) break;
  if (!line.trim()) continue;
  messages.push({ role: "user", content: line });
  current = new AbortController();
  const res = await chat.complete({ messages }, {
    signal: current.signal,
    onToken: (t) => void Deno.stdout.writeSync(out.encode(t)),
  });
  current = null;
  const u = res.usage;
  console.error(
    `\n[${res.finish_reason} | ${u.prompt_tokens} prompt (${u.cached_tokens} cached) | ` +
      `${u.completion_tokens} tok, ${(u.completion_tokens / (u.generation_ms / 1000)).toFixed(1)} tok/s]`,
  );
  messages.push({ role: "assistant", content: res.content });
  if (oneShot) break;
}
