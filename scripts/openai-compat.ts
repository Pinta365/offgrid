// OpenAI SDK compatibility checks for the offgrid server.
//
// Usage:
//   deno run -A scripts/openai-compat.ts <baseURL> <apiKey> [thinkingBaseURL]
//
// <baseURL> should include /v1, e.g. http://127.0.0.1:18080/v1

import OpenAI, { AuthenticationError } from "npm:openai@^5";

const [baseURL, apiKey, thinkingBaseURL] = Deno.args;
if (!baseURL || !apiKey) {
  console.error(
    "usage: deno run -A scripts/openai-compat.ts <baseURL> <apiKey> [thinkingBaseURL]",
  );
  Deno.exit(2);
}

const CHAT_MODEL = "qwen3-4b";
const EMBED_MODEL = "Qwen3-Embedding-0.6B-Q8_0";
const THINK_MODEL = "qwen3-0.6b";

const client = new OpenAI({ baseURL, apiKey });

let failures = 0;
const results: { name: string; ok: boolean }[] = [];

async function run(name: string, fn: () => Promise<void>): Promise<void> {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`PASS: ${name}`);
  } catch (err) {
    results.push({ name, ok: false });
    failures++;
    console.error(
      `FAIL: ${name}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const greedyReq = {
  model: CHAT_MODEL,
  messages: [{ role: "user" as const, content: "The capital of France is, in one word:" }],
  temperature: 0,
  max_tokens: 20,
};
let nonStreamContent = "";

const weatherTool = {
  type: "function" as const,
  function: {
    name: "get_weather",
    description: "Get the weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
};
const weatherToolPrompt = "What is the weather in Paris? Use the tool.";

await run("models.list contains chat model id", async () => {
  const list = await client.models.list();
  const ids = list.data.map((m) => m.id);
  assert(
    ids.includes(CHAT_MODEL),
    `expected "${CHAT_MODEL}" in ${JSON.stringify(ids)}`,
  );
});

await run("chat non-stream reply mentions Paris", async () => {
  const r = await client.chat.completions.create(greedyReq);
  nonStreamContent = r.choices[0]?.message?.content ?? "";
  assert(/paris/i.test(nonStreamContent), `got ${JSON.stringify(nonStreamContent)}`);
});

await run("chat stream matches non-stream, finish_reason stop, usage", async () => {
  assert(nonStreamContent.length > 0, "non-stream content empty (check 2 failed)");
  const stream = await client.chat.completions.create({
    ...greedyReq,
    stream: true,
    stream_options: { include_usage: true },
  });
  let content = "";
  let finish: string | null = null;
  let usage: unknown = null;
  for await (const chunk of stream) {
    const choice = chunk.choices[0];
    if (choice) {
      if (choice.finish_reason) finish = choice.finish_reason;
      const delta = choice.delta as { content?: string | null };
      if (delta.content) content += delta.content;
    }
    if (chunk.usage) usage = chunk.usage;
  }
  assert(
    content.trim() === nonStreamContent.trim(),
    `stream ${JSON.stringify(content)} != non-stream ${JSON.stringify(nonStreamContent)}`,
  );
  assert(finish === "stop", `finish_reason ${JSON.stringify(finish)}`);
  assert(usage !== null, "no usage chunk despite stream_options.include_usage");
});

await run("tools non-stream: call + tool result follow-up", async () => {
  const r = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: weatherToolPrompt }],
    temperature: 0,
    max_tokens: 200,
    tools: [weatherTool],
  });
  const choice = r.choices[0];
  const msg = choice.message;
  const call = msg.tool_calls?.[0];
  assert(call, "no tool_calls returned");
  assert(call.function.name === "get_weather", `name ${call.function.name}`);
  const args = JSON.parse(call.function.arguments);
  assert(/paris/i.test(args.city ?? ""), `arguments ${call.function.arguments}`);
  assert(choice.finish_reason === "tool_calls", `finish_reason ${choice.finish_reason}`);

  const r2 = await client.chat.completions.create({
    model: CHAT_MODEL,
    temperature: 0,
    max_tokens: 100,
    tools: [weatherTool],
    messages: [
      { role: "user", content: weatherToolPrompt },
      msg as never,
      {
        role: "tool",
        tool_call_id: call.id,
        content: "The temperature in Paris is 22 degrees Celsius.",
      },
    ],
  });
  const answer = r2.choices[0]?.message?.content ?? "";
  assert(answer.includes("22"), `final answer did not use result: ${JSON.stringify(answer)}`);
});

await run("tools streaming: accumulate delta.tool_calls", async () => {
  const stream = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: weatherToolPrompt }],
    temperature: 0,
    max_tokens: 200,
    stream: true,
    tools: [weatherTool],
  });
  const acc: Record<number, { name?: string; args: string; id?: string }> = {};
  for await (const chunk of stream) {
    const choice = chunk.choices[0];
    if (!choice) continue;
    for (const tc of choice.delta.tool_calls ?? []) {
      const idx = tc.index;
      if (!acc[idx]) acc[idx] = { args: "" };
      if (tc.id) acc[idx].id = tc.id;
      if (tc.function?.name) acc[idx].name = tc.function.name;
      if (tc.function?.arguments) acc[idx].args += tc.function.arguments;
    }
  }
  const call = acc[0];
  assert(call, `no streamed tool_calls: ${JSON.stringify(acc)}`);
  assert(call.name === "get_weather", `name ${call.name}`);
  const args = JSON.parse(call.args);
  assert(/paris/i.test(args.city ?? ""), `arguments ${call.args}`);
});

// 6 -- SDK runTools/zod helpers: intentionally skipped.

await run("response_format json_schema parses to {name,age}", async () => {
  const r = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: "user", content: "Return a person: Ada Lovelace, 36." }],
    temperature: 0,
    max_tokens: 100,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "person",
        strict: true,
        schema: {
          type: "object",
          properties: { name: { type: "string" }, age: { type: "integer" } },
          required: ["name", "age"],
        },
      },
    },
  });
  const text = r.choices[0]?.message?.content ?? "";
  const parsed = JSON.parse(text);
  assert(typeof parsed.name === "string", `name is ${typeof parsed.name}`);
  assert(typeof parsed.age === "number", `age is ${typeof parsed.age}`);
});

await run("embeddings base64 decode + cosine ordering", async () => {
  const emb = await client.embeddings.create({
    model: EMBED_MODEL,
    input: [
      "The cat sits on the mat.",
      "A kitten is resting on a rug.",
      "Quarterly tax filing deadlines",
    ],
  });
  assert(emb.data.length === 3, `expected 3 embeddings, got ${emb.data.length}`);
  const vecs = emb.data.map((d) => d.embedding as number[]);
  const len = vecs[0].length;
  assert(len > 0, "empty embedding vector");
  assert(vecs.every((v) => v.length === len), "vectors differ in length");
  const close = cosine(vecs[0], vecs[1]);
  const far = cosine(vecs[0], vecs[2]);
  assert(close > far, `cosine(cat,kitten)=${close} !> cosine(cat,tax)=${far}`);
});

await run("wrong API key throws AuthenticationError 401", async () => {
  const bad = new OpenAI({ baseURL, apiKey: "definitely-wrong" });
  let thrown: unknown = null;
  try {
    await bad.models.list();
  } catch (e) {
    thrown = e;
  }
  assert(thrown !== null, "expected an error for a wrong API key");
  assert(
    thrown instanceof AuthenticationError,
    `expected AuthenticationError, got ${(thrown as object)?.constructor?.name}`,
  );
  assert((thrown as { status?: number }).status === 401, "status was not 401");
});

if (thinkingBaseURL) {
  await run("thinking stream: reasoning_content before content, content has 42", async () => {
    const tc = new OpenAI({ baseURL: thinkingBaseURL, apiKey });
    const stream = await tc.chat.completions.create({
      model: THINK_MODEL,
      messages: [{ role: "user", content: "What is 6*7? Answer with just the number." }],
      temperature: 0.6,
      top_p: 0.95,
      seed: 7,
      max_tokens: 1500,
      stream: true,
    });
    let content = "";
    let reasoning = "";
    let firstReasoning = Infinity;
    let firstContent = Infinity;
    let i = 0;
    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      if (choice) {
        const delta = choice.delta as { content?: string | null; reasoning_content?: string | null };
        if (delta.reasoning_content) {
          if (firstReasoning === Infinity) firstReasoning = i;
          reasoning += delta.reasoning_content;
        }
        if (delta.content) {
          if (firstContent === Infinity) firstContent = i;
          content += delta.content;
        }
      }
      i++;
    }
    assert(reasoning.length > 0, "no reasoning_content deltas");
    assert(
      firstReasoning < firstContent,
      `reasoning started at chunk ${firstReasoning}, content at ${firstContent}`,
    );
    assert(content.includes("42"), `content missing 42: ${JSON.stringify(content)}`);
  });
} else {
  console.log("SKIP: thinking server check (no thinkingBaseURL given)");
}

console.log("");
if (failures === 0) {
  console.log(`ALL CHECKS PASSED (${results.length})`);
  Deno.exit(0);
} else {
  console.log(`${failures} CHECK(S) FAILED of ${results.length}`);
  Deno.exit(1);
}
