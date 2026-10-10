// Tool calling from Deno with local tools.
// deno run --allow-ffi --allow-env --allow-read examples/deno/tools.ts <model.gguf> ["question"]
import { Offgrid } from "../../bindings/deno/mod.ts";

const [path, question = "What time is it in UTC, and what is 1234 * 5678?"] = Deno.args;
if (!path) {
  console.error("usage: tools.ts <model.gguf> [question]");
  Deno.exit(1);
}

using core = Offgrid.open();
using model = await core.loadModel(path);
using chat = await model.createChat();
const out = new TextEncoder();

const { response } = await chat.runTools(
  {
    messages: [
      { role: "system", content: "Use the tools for facts and arithmetic." },
      { role: "user", content: question },
    ],
    tools: [
      { type: "function", function: { name: "utc_now", description: "Current UTC date and time." } },
      {
        type: "function",
        function: {
          name: "multiply",
          description: "Multiply two integers exactly.",
          parameters: {
            type: "object",
            properties: { a: { type: "integer" }, b: { type: "integer" } },
            required: ["a", "b"],
          },
        },
      },
    ],
  },
  {
    utc_now: () => ({ utc: new Date().toISOString() }),
    multiply: ({ a, b }) => ({ product: BigInt(a as number) * BigInt(b as number) + "" }),
  },
  {
    onToken: (t) => void Deno.stdout.writeSync(out.encode(t)),
    onToolCall: (call, result) =>
      console.error(`[tool] ${call.function.name}(${call.function.arguments}) -> ${result}`),
  },
);
console.log();
console.error(`[${response.finish_reason}]`);
