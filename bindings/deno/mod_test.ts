// deno test --allow-ffi --allow-env --allow-read bindings/deno/mod_test.ts
// Model tests need AICORE_TEST_MODEL / AICORE_TEST_EMBED_MODEL; they are skipped otherwise.
import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { AiCore, type ChatRequest, dot } from "./mod.ts";

const chatModel = Deno.env.get("AICORE_TEST_MODEL");
const embedModel = Deno.env.get("AICORE_TEST_EMBED_MODEL");

const greedy = (content: string, max_tokens: number): ChatRequest => ({
  messages: [{ role: "user", content }],
  max_tokens,
  temperature: 0,
});

Deno.test("version and devices", () => {
  using core = AiCore.open();
  assertEquals(core.version().abi, 1);
  assert(core.devices().length > 0);
});

Deno.test("load error is reported", async () => {
  using core = AiCore.open();
  await assertRejects(() => core.loadModel("/nope.gguf"), Error, "not found");
});

Deno.test({ name: "chat: stream, prefix reuse, cancel", ignore: !chatModel }, async () => {
  using core = AiCore.open();
  using model = await core.loadModel(chatModel!);
  using chat = await model.createChat();

  // Event loop stays responsive while generating.
  let ticks = 0;
  const timer = setInterval(() => ticks++, 5);
  let streamed = "";
  const r1 = await chat.complete(greedy("What is the capital of France? One word.", 16), {
    onToken: (t) => void (streamed += t),
  });
  clearInterval(timer);
  assertStringIncludes(r1.content, "Paris");
  assertEquals(streamed, r1.content);
  assertEquals(r1.finish_reason, "stop");
  assert(ticks > 0, "event loop was blocked");

  const r2 = await chat.complete({
    messages: [
      { role: "user", content: "What is the capital of France? One word." },
      { role: "assistant", content: r1.content },
      { role: "user", content: "And Germany?" },
    ],
    max_tokens: 16,
    temperature: 0,
  });
  assertStringIncludes(r2.content, "Berlin");
  assert(r2.usage.cached_tokens > 5);

  // Cancel via onToken returning false.
  let n = 0;
  const r3 = await chat.complete(greedy("Count from 1 to 100.", 300), { onToken: () => ++n < 3 });
  assertEquals(r3.finish_reason, "cancelled");

  // Cancel via AbortSignal.
  const ac = new AbortController();
  const r4 = await chat.complete(greedy("Count from 1 to 100.", 300), {
    signal: ac.signal,
    onToken: () => ac.abort(),
  });
  assertEquals(r4.finish_reason, "cancelled");

  // Async iterator; breaking early cancels and leaves the session usable.
  const pieces: string[] = [];
  for await (const p of chat.stream(greedy("Count from 1 to 100.", 300))) {
    pieces.push(p);
    if (pieces.length === 4) break;
  }
  assertEquals(pieces.length, 4);
  const r5 = await chat.complete(greedy("Say OK.", 5));
  assert(r5.content.length > 0);

  // Concurrent use of one session is rejected, not a crash.
  const a = chat.complete(greedy("Say A.", 5));
  await assertRejects(() => chat.complete(greedy("Say B.", 5)), Error, "busy");
  await a;
});

Deno.test({ name: "chat: json_schema response_format", ignore: !chatModel }, async () => {
  using core = AiCore.open();
  using model = await core.loadModel(chatModel!);
  using chat = await model.createChat();
  const res = await chat.complete({
    ...greedy("Extract: 'Grace Hopper, 85, computer scientist'.", 200),
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "person",
        schema: {
          type: "object",
          properties: { name: { type: "string" }, age: { type: "integer" } },
          required: ["name", "age"],
          additionalProperties: false,
        },
      },
    },
  });
  const v = JSON.parse(res.content);
  assertEquals(Object.keys(v), ["name", "age"]);
  assertStringIncludes(v.name, "Hopper");
  assertEquals(v.age, 85);
});

Deno.test({ name: "chat: tool calling loop", ignore: !chatModel }, async () => {
  using core = AiCore.open();
  using model = await core.loadModel(chatModel!);
  using chat = await model.createChat();
  const calls: string[] = [];
  let streamed = "";
  const { response, messages } = await chat.runTools(
    {
      messages: [{ role: "user", content: "What is the weather in Paris? Answer in one sentence." }],
      temperature: 0,
      max_tokens: 200,
      tools: [{
        type: "function",
        function: {
          name: "get_weather",
          description: "Current weather for a city",
          parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
        },
      }],
    },
    {
      get_weather: ({ city }) => {
        calls.push(String(city));
        return { temperature_c: 21, conditions: "sunny" };
      },
    },
    { onToken: (t) => void (streamed += t) },
  );
  assertEquals(calls.length, 1);
  assertStringIncludes(calls[0], "Paris");
  assertEquals(response.finish_reason, "stop");
  assertStringIncludes(response.content, "21");
  assertEquals(messages.map((m) => m.role), ["user", "assistant", "tool", "assistant"]);
  assert(!streamed.includes("<tool_call>"), streamed);
});

Deno.test({ name: "embeddings", ignore: !embedModel }, async () => {
  using core = AiCore.open();
  using model = await core.loadModel(embedModel!);
  using emb = await model.createEmbedder();
  const [cat, kitten, tax] = await emb.embed([
    "The cat sits on the mat.",
    "A kitten is resting on a rug.",
    "Quarterly tax filing deadlines",
  ]);
  assertEquals(cat.length, emb.dim);
  assert(dot(cat, kitten) > dot(cat, tax) + 0.2);
});
