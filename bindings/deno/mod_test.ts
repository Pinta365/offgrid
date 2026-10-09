// deno test --allow-ffi --allow-env --allow-read bindings/deno/mod_test.ts
// Model tests need OFFGRID_TEST_MODEL / OFFGRID_TEST_EMBED_MODEL; they are skipped otherwise.
import { assert, assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { Offgrid, audioPart, type ChatRequest, dot, imagePart } from "./mod.ts";

const chatModel = Deno.env.get("OFFGRID_TEST_MODEL");
const thinkingModel = Deno.env.get("OFFGRID_TEST_THINKING_MODEL");
const vlModel = Deno.env.get("OFFGRID_TEST_VL_MODEL");
const vlMmproj = Deno.env.get("OFFGRID_TEST_VL_MMPROJ");
const testImages = Deno.env.get("OFFGRID_TEST_IMAGES");
const audioModel = Deno.env.get("OFFGRID_TEST_AUDIO_MODEL");
const audioMmproj = Deno.env.get("OFFGRID_TEST_AUDIO_MMPROJ");
const testAudio = Deno.env.get("OFFGRID_TEST_AUDIO");
const embedModel = Deno.env.get("OFFGRID_TEST_EMBED_MODEL");

const greedy = (content: string, max_tokens: number): ChatRequest => ({
  messages: [{ role: "user", content }],
  max_tokens,
  temperature: 0,
});

Deno.test("version and devices", () => {
  using core = Offgrid.open();
  assertEquals(core.version().abi, 1);
  assert(core.devices().length > 0);
});

Deno.test("load error is reported", async () => {
  using core = Offgrid.open();
  await assertRejects(() => core.loadModel("/nope.gguf"), Error, "not found");
});

Deno.test({ name: "chat: stream, prefix reuse, cancel", ignore: !chatModel }, async () => {
  using core = Offgrid.open();
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

  // Abort issued before the native call can start must still cancel this request...
  const early = new AbortController();
  const pEarly = chat.complete(greedy("Count from 1 to 100.", 300), { signal: early.signal });
  early.abort();
  assertEquals((await pEarly).finish_reason, "cancelled");
  // ...and must not leak into the next request.
  assertEquals((await chat.complete(greedy("Say OK.", 3))).finish_reason !== "cancelled", true);

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

  // Concurrent use of one session is rejected, not a crash (or a deadlock).
  const a = chat.complete(greedy("Say A.", 5), { onToken: () => {} });
  await assertRejects(() => chat.complete(greedy("Say B.", 5)), Error, "busy");
  assertThrows(() => chat.reset(), Error, "while a request is running");
  assertThrows(() => chat.free(), Error, "while a request is running");
  await a;
  chat.reset(); // fine once idle

  // A model can't be freed while a session is being created from it.
  const pending = model.createChat();
  assertThrows(() => model.free(), Error, "while a request is running");
  (await pending).free();
});

Deno.test({ name: "chat: json_schema response_format", ignore: !chatModel }, async () => {
  using core = Offgrid.open();
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
  using core = Offgrid.open();
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

Deno.test({ name: "chat: reasoning streams via onReasoning", ignore: !thinkingModel }, async () => {
  using core = Offgrid.open();
  using model = await core.loadModel(thinkingModel!);
  using chat = await model.createChat();
  const order: string[] = [];
  let reasoning = "", content = "";
  const res = await chat.complete(
    {
      messages: [{ role: "user", content: "What is 17 + 25? Answer with just the number." }],
      temperature: 0.6,
      top_p: 0.95,
      top_k: 20,
      seed: 7,
      max_tokens: 3000,
    },
    {
      onReasoning: (t) => {
        reasoning += t;
        if (order.at(-1) !== "r") order.push("r");
      },
      onToken: (t) => {
        content += t;
        if (order.at(-1) !== "c") order.push("c");
      },
    },
  );
  assertEquals(order, ["r", "c"]);
  assertEquals(reasoning, res.reasoning_content);
  assertEquals(content, res.content);
  assertStringIncludes(res.content, "42");
});

Deno.test({ name: "chat: image input", ignore: !vlModel || !vlMmproj || !testImages }, async () => {
  using core = Offgrid.open();
  using model = await core.loadModel(vlModel!, { mmproj: vlMmproj });
  assert(model.info().vision);
  using chat = await model.createChat();
  const res = await chat.complete({
    temperature: 0,
    max_tokens: 40,
    messages: [{
      role: "user",
      content: [{ type: "text", text: "What text is in this image?" }, await imagePart(`${testImages}/text_hello.png`)],
    }],
  });
  assertStringIncludes(res.content.toUpperCase(), "HELLO 42");
});

Deno.test({ name: "chat: audio input", ignore: !audioModel || !audioMmproj || !testAudio }, async () => {
  using core = Offgrid.open();
  using model = await core.loadModel(audioModel!, { mmproj: audioMmproj });
  assert(model.info().audio);
  using chat = await model.createChat();
  const res = await chat.complete({
    temperature: 0,
    max_tokens: 60,
    messages: [{
      role: "user",
      content: [{ type: "text", text: "Transcribe this audio exactly." }, await audioPart(`${testAudio}/jfk.wav`)],
    }],
  });
  assertStringIncludes(res.content.toLowerCase(), "ask not");
});

Deno.test({ name: "embeddings", ignore: !embedModel }, async () => {
  using core = Offgrid.open();
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
