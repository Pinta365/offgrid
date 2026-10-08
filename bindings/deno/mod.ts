/**
 * Deno bindings for aicore (embeddable offline AI core).
 *
 * Long-running calls (model load, chat, embeddings) run off the JS thread via
 * `nonblocking` FFI, so they return Promises and never block the event loop.
 *
 * Requires `--allow-ffi` (and `--allow-env`/`--allow-read` to locate the library).
 *
 * @example
 * ```ts
 * import { AiCore } from "./mod.ts";
 * using core = AiCore.open();
 * using model = await core.loadModel("model.gguf");
 * using chat = await model.createChat();
 * const res = await chat.complete(
 *   { messages: [{ role: "user", content: "Hi!" }] },
 *   { onToken: (t) => Deno.stdout.writeSync(new TextEncoder().encode(t)) },
 * );
 * ```
 * @module
 */

import type {
  ChatMessage,
  ChatRequest,
  ChatResponse,
  ContentPart,
  DeviceInfo,
  EmbedParams,
  GenerateRequest,
  ModelInfo,
  ModelParams,
  SessionParams,
  ToolCall,
  VersionInfo,
} from "./types.ts";

export type * from "./types.ts";

/** ABI version these bindings were written against. */
export const ABI_VERSION = 1;

const symbols = {
  aicore_abi_version: { parameters: [], result: "u32" },
  aicore_version_json: { parameters: [], result: "pointer" },
  aicore_string_free: { parameters: ["pointer"], result: "void" },
  aicore_devices_json: { parameters: [], result: "pointer" },
  aicore_model_load: { parameters: ["buffer", "buffer", "buffer"], result: "pointer", nonblocking: true },
  aicore_model_info_json: { parameters: ["pointer"], result: "pointer" },
  aicore_model_free: { parameters: ["pointer"], result: "void" },
  aicore_chat_new: { parameters: ["pointer", "buffer", "buffer"], result: "pointer", nonblocking: true },
  aicore_chat_complete: {
    parameters: ["pointer", "buffer", "function", "pointer", "buffer"],
    result: "pointer",
    nonblocking: true,
  },
  aicore_chat_complete_events: {
    parameters: ["pointer", "buffer", "function", "pointer", "buffer"],
    result: "pointer",
    nonblocking: true,
  },
  aicore_chat_generate: {
    parameters: ["pointer", "buffer", "function", "pointer", "buffer"],
    result: "pointer",
    nonblocking: true,
  },
  aicore_chat_cancel: { parameters: ["pointer"], result: "void" },
  aicore_chat_next_request_id: { parameters: ["pointer"], result: "u64" },
  aicore_chat_cancel_request: { parameters: ["pointer", "u64"], result: "void" },
  aicore_chat_reset: { parameters: ["pointer"], result: "void" },
  aicore_chat_free: { parameters: ["pointer"], result: "void" },
  aicore_embedder_new: { parameters: ["pointer", "buffer", "buffer"], result: "pointer", nonblocking: true },
  aicore_embedder_dim: { parameters: ["pointer"], result: "i64" },
  aicore_embed: {
    parameters: ["pointer", "buffer", "usize", "buffer", "usize", "buffer"],
    result: "i32",
    nonblocking: true,
  },
  aicore_embedder_free: { parameters: ["pointer"], result: "void" },
} as const satisfies Deno.ForeignLibraryInterface;

type Lib = Deno.DynamicLibrary<typeof symbols>["symbols"];

const TOKEN_CALLBACK = {
  parameters: ["pointer", "usize", "pointer"],
  result: "bool",
} as const satisfies Deno.UnsafeCallbackDefinition;

const EVENT_CALLBACK = {
  parameters: ["i32", "pointer", "usize", "pointer"],
  result: "bool",
} as const satisfies Deno.UnsafeCallbackDefinition;

/** Event kinds of aicore_chat_complete_events. */
const EVENT_CONTENT = 0;
const EVENT_REASONING = 1;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Thrown for errors reported by the native library. */
export class AiCoreError extends Error {
  override name = "AiCoreError";
}

function cstr(s: string): Uint8Array<ArrayBuffer> {
  const bytes = encoder.encode(s);
  const buf = new Uint8Array(bytes.length + 1);
  buf.set(bytes);
  return buf;
}

function jsonArg(value: unknown): Uint8Array<ArrayBuffer> {
  return cstr(value === undefined ? "" : JSON.stringify(value));
}

/** Read and free a library-owned string. */
function takeString(lib: Lib, ptr: Deno.PointerValue): string | null {
  if (ptr === null) return null;
  try {
    return Deno.UnsafePointerView.getCString(ptr);
  } finally {
    lib.aicore_string_free(ptr);
  }
}

/** A `char **error_out` slot. */
class ErrorSlot {
  readonly buf = new BigUint64Array(1);
  throwIfSet(lib: Lib, what: string): never | void {
    const addr = this.buf[0];
    if (addr === 0n) return;
    const msg = takeString(lib, Deno.UnsafePointer.create(addr));
    throw new AiCoreError(`${what}: ${msg ?? "unknown error"}`);
  }
  fail(lib: Lib, what: string): never {
    this.throwIfSet(lib, what);
    throw new AiCoreError(`${what} failed`);
  }
}

/** file: URL to a native path (decodes %XX, handles Windows drive letters). */
function fromFileUrl(url: URL): string {
  const path = decodeURIComponent(url.pathname);
  return Deno.build.os === "windows" ? path.replace(/^\/([A-Za-z]:)/, "$1").replaceAll("/", "\\") : path;
}

function defaultLibPath(): string {
  const env = Deno.env.get("AICORE_LIB");
  if (env) return env;
  const file = Deno.build.os === "windows"
    ? "aicore.dll"
    : Deno.build.os === "darwin"
    ? "libaicore.dylib"
    : "libaicore.so";
  for (const profile of ["release", "debug"]) {
    const url = new URL(`../../target/${profile}/${file}`, import.meta.url);
    try {
      Deno.statSync(url);
      return fromFileUrl(url);
    } catch { /* try next */ }
  }
  return file; // fall back to the system loader search path
}

/** Handle to the loaded native library. */
export class AiCore implements Disposable {
  readonly #dylib: Deno.DynamicLibrary<typeof symbols>;
  /** @internal */
  readonly lib: Lib;

  private constructor(dylib: Deno.DynamicLibrary<typeof symbols>) {
    this.#dylib = dylib;
    this.lib = dylib.symbols;
  }

  /** Load the native library. Path defaults to `$AICORE_LIB`, then the repo's target/ dir. */
  static open(path: string = defaultLibPath()): AiCore {
    const dylib = Deno.dlopen(path, symbols);
    const abi = dylib.symbols.aicore_abi_version();
    if (abi !== ABI_VERSION) {
      dylib.close();
      throw new AiCoreError(`ABI mismatch: library ${abi}, bindings ${ABI_VERSION}`);
    }
    return new AiCore(dylib);
  }

  version(): VersionInfo {
    return JSON.parse(takeString(this.lib, this.lib.aicore_version_json())!);
  }

  devices(): DeviceInfo[] {
    return JSON.parse(takeString(this.lib, this.lib.aicore_devices_json())!);
  }

  async loadModel(path: string, params?: ModelParams): Promise<Model> {
    const err = new ErrorSlot();
    const ptr = await this.lib.aicore_model_load(cstr(path), jsonArg(params), err.buf);
    if (ptr === null) return err.fail(this.lib, `load model ${path}`);
    return new Model(this, ptr);
  }

  /** Unload the library. All models/sessions must be disposed first. */
  close(): void {
    this.#dylib.close();
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

abstract class Handle implements Disposable {
  #ptr: Deno.PointerObject | null;
  #busy = false;
  /** Native calls in flight that use this handle (it must not be freed meanwhile). */
  #pending = 0;

  constructor(readonly core: AiCore, ptr: Deno.PointerObject) {
    this.#ptr = ptr;
  }

  protected get ptr(): Deno.PointerObject {
    if (this.#ptr === null) throw new AiCoreError(`${this.constructor.name} is disposed`);
    return this.#ptr;
  }

  protected get lib(): Lib {
    return this.core.lib;
  }

  /** Run a native call exclusively (one request at a time); the handle cannot be freed meanwhile. */
  protected async exclusive<T>(fn: (ptr: Deno.PointerObject) => Promise<T>): Promise<T> {
    if (this.#busy) throw new AiCoreError(`${this.constructor.name} is busy with another request`);
    this.#busy = true;
    try {
      return await this.pinned(fn);
    } finally {
      this.#busy = false;
    }
  }

  /** Run a (possibly concurrent) native call; the handle cannot be freed until it settles. */
  protected async pinned<T>(fn: (ptr: Deno.PointerObject) => Promise<T>): Promise<T> {
    const ptr = this.ptr;
    this.#pending++;
    try {
      return await fn(ptr);
    } finally {
      this.#pending--;
    }
  }

  /** Throw if a native call is running (sync calls that take the native lock would deadlock). */
  protected assertIdle(what: string): void {
    if (this.#busy || this.#pending > 0) {
      throw new AiCoreError(`cannot ${what} ${this.constructor.name} while a request is running`);
    }
  }

  protected abstract release(ptr: Deno.PointerObject): void;

  /** Free the native handle. Throws if a request is still running. */
  free(): void {
    if (this.#ptr === null) return;
    this.assertIdle("free");
    this.release(this.#ptr);
    this.#ptr = null;
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/** A loaded GGUF model. Sessions/embedders keep it alive natively, so it may be freed first. */
export class Model extends Handle {
  info(): ModelInfo {
    return JSON.parse(takeString(this.lib, this.lib.aicore_model_info_json(this.ptr))!);
  }

  createChat(params?: SessionParams): Promise<ChatSession> {
    return this.pinned(async (model) => {
      const err = new ErrorSlot();
      const ptr = await this.lib.aicore_chat_new(model, jsonArg(params), err.buf);
      if (ptr === null) return err.fail(this.lib, "create chat session");
      return new ChatSession(this.core, ptr);
    });
  }

  createEmbedder(params?: EmbedParams): Promise<Embedder> {
    return this.pinned(async (model) => {
      const err = new ErrorSlot();
      const ptr = await this.lib.aicore_embedder_new(model, jsonArg(params), err.buf);
      if (ptr === null) return err.fail(this.lib, "create embedder");
      const dim = Number(this.lib.aicore_embedder_dim(ptr));
      return new Embedder(this.core, ptr, dim);
    });
  }

  protected release(ptr: Deno.PointerObject): void {
    this.lib.aicore_model_free(ptr);
  }
}

export interface CompleteOptions {
  /** Called with each streamed piece of reply text. Return `false` to stop. */
  onToken?: (piece: string) => boolean | void;
  /** Called with each streamed piece of reasoning (`<think>`) text. Return `false` to stop. */
  onReasoning?: (piece: string) => boolean | void;
  /** Aborting cancels the generation; the promise resolves with `finish_reason: "cancelled"`. */
  signal?: AbortSignal;
}

/** Implementations of tools, keyed by function name. Receives parsed arguments. */
export type ToolHandlers = Record<string, (args: Record<string, unknown>, call: ToolCall) => unknown>;

export interface RunToolsOptions extends CompleteOptions {
  /** Max tool rounds before a final answer is forced with `tool_choice: "none"` (default 5). */
  maxRounds?: number;
  /** Observe each executed call and its result text. */
  onToolCall?: (call: ToolCall, result: string) => void;
}

/** One inference context. Requests are stateless; the KV cache prefix is reused between calls. */
export class ChatSession extends Handle {
  /**
   * Run a tool loop: complete, execute requested calls with `handlers`, append the calls and
   * results, and repeat until the model answers. Handler return values are sent as-is if
   * strings, else JSON-encoded; thrown errors and unknown tools become `{"error": ...}` results.
   * Returns the final response and the whole conversation for continuing the chat.
   */
  async runTools(
    request: ChatRequest,
    handlers: ToolHandlers,
    options: RunToolsOptions = {},
  ): Promise<{ response: ChatResponse; messages: ChatMessage[] }> {
    const maxRounds = options.maxRounds ?? 5;
    const messages = [...request.messages];
    for (let round = 0; round <= maxRounds; round++) {
      const req: ChatRequest = { ...request, messages };
      if (round === maxRounds) req.tool_choice = "none";
      const response = await this.complete(req, options);
      messages.push({
        role: "assistant",
        content: response.content,
        reasoning_content: response.reasoning_content,
        tool_calls: response.tool_calls,
      });
      if (response.finish_reason !== "tool_calls" || !response.tool_calls?.length) return { response, messages };
      for (const call of response.tool_calls) {
        const result = await runHandler(handlers, call);
        options.onToolCall?.(call, result);
        messages.push({ role: "tool", tool_call_id: call.id, name: call.function.name, content: result });
      }
    }
    throw new AiCoreError("unreachable: last round runs with tool_choice none");
  }

  /** Chat completion using the model's chat template. */
  complete(request: ChatRequest, options: CompleteOptions = {}): Promise<ChatResponse> {
    return this.#run("aicore_chat_complete_events", request, options);
  }

  /** Raw text completion without a chat template. */
  generate(request: GenerateRequest, options: CompleteOptions = {}): Promise<ChatResponse> {
    return this.#run("aicore_chat_generate", request, options);
  }

  /** Stream pieces as an async iterator. Breaking out of the loop cancels generation. */
  async *stream(request: ChatRequest, options: { signal?: AbortSignal } = {}): AsyncGenerator<string, ChatResponse> {
    const queue: string[] = [];
    let wake: (() => void) | null = null;
    let done = false;
    const pending = this.complete(request, {
      signal: options.signal,
      onToken: (t) => {
        queue.push(t);
        wake?.();
      },
    }).finally(() => {
      done = true;
      wake?.();
    });
    try {
      while (true) {
        while (queue.length) yield queue.shift()!;
        if (done) break;
        await new Promise<void>((r) => (wake = r));
        wake = null;
      }
      return await pending;
    } finally {
      if (!done) {
        this.cancel();
        await pending.catch(() => {});
      }
    }
  }

  /** Cancel the running request (safe to call any time). */
  cancel(): void {
    this.lib.aicore_chat_cancel(this.ptr);
  }

  /** Clear the KV cache. Throws while a request is running (cancel and await it first). */
  reset(): void {
    this.assertIdle("reset");
    this.lib.aicore_chat_reset(this.ptr);
  }

  async #run(
    fn: "aicore_chat_complete_events" | "aicore_chat_generate",
    request: unknown,
    { onToken, onReasoning, signal }: CompleteOptions,
  ): Promise<ChatResponse> {
    signal?.throwIfAborted();
    const text = (ptr: Deno.PointerValue, len: number | bigint) =>
      ptr === null ? "" : decoder.decode(new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(ptr, Number(len))));
    return await this.exclusive(async (ptr) => {
      // Native code calls back from a worker thread; threadSafe marshals it to the JS thread.
      let callback:
        | Deno.UnsafeCallback<typeof EVENT_CALLBACK>
        | Deno.UnsafeCallback<typeof TOKEN_CALLBACK>
        | null = null;
      if (fn === "aicore_chat_complete_events" && (onToken || onReasoning)) {
        callback = Deno.UnsafeCallback.threadSafe(EVENT_CALLBACK, (kind, piecePtr, len) => {
          if (kind === EVENT_CONTENT) return onToken?.(text(piecePtr, len)) !== false;
          if (kind === EVENT_REASONING) return onReasoning?.(text(piecePtr, len)) !== false;
          return true; // tool calls arrive parsed in the response
        });
      } else if (onToken) {
        callback = Deno.UnsafeCallback.threadSafe(
          TOKEN_CALLBACK,
          (piecePtr, len) => onToken(text(piecePtr, len)) !== false,
        );
      }
      // Cancel this specific request: works even if the abort fires before the native call
      // has started (and can't hit a later request). Safe: the session is exclusive here.
      const requestId = this.lib.aicore_chat_next_request_id(ptr);
      const onAbort = () => this.lib.aicore_chat_cancel_request(ptr, requestId);
      signal?.addEventListener("abort", onAbort, { once: true });
      const err = new ErrorSlot();
      try {
        const res = await this.lib[fn](ptr, jsonArg(request), callback?.pointer ?? null, null, err.buf);
        if (res === null) return err.fail(this.lib, "chat");
        return JSON.parse(takeString(this.lib, res)!) as ChatResponse;
      } finally {
        signal?.removeEventListener("abort", onAbort);
        callback?.close();
      }
    });
  }

  protected release(ptr: Deno.PointerObject): void {
    this.lib.aicore_chat_free(ptr);
  }
}

/** Computes embedding vectors (L2-normalized by default). */
export class Embedder extends Handle {
  constructor(core: AiCore, ptr: Deno.PointerObject, readonly dim: number) {
    super(core, ptr);
  }

  /** Embed texts; returns one vector per input, in order. */
  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    return await this.exclusive(async (ptr) => {
      const strings = texts.map(cstr);
      const ptrs = new BigUint64Array(strings.map((s) => BigInt(Deno.UnsafePointer.value(Deno.UnsafePointer.of(s)))));
      const out = new Float32Array(texts.length * this.dim);
      const err = new ErrorSlot();
      const rc = await this.lib.aicore_embed(ptr, ptrs, BigInt(texts.length), out, BigInt(out.length), err.buf);
      void strings; // keep input buffers alive until the call returns
      if (rc !== 0) err.fail(this.lib, "embed");
      return texts.map((_, i) => out.subarray(i * this.dim, (i + 1) * this.dim));
    });
  }

  protected release(ptr: Deno.PointerObject): void {
    this.lib.aicore_embedder_free(ptr);
  }
}

async function runHandler(handlers: ToolHandlers, call: ToolCall): Promise<string> {
  const handler = handlers[call.function.name];
  if (!handler) return JSON.stringify({ error: `unknown tool ${call.function.name}` });
  try {
    const args = JSON.parse(call.function.arguments || "{}");
    const out = await handler(args, call);
    return typeof out === "string" ? out : JSON.stringify(out ?? null);
  } catch (e) {
    return JSON.stringify({ error: e instanceof Error ? e.message : String(e) });
  }
}

/**
 * An `image_url` content part from encoded image bytes (PNG, JPEG, GIF, BMP) or a file path.
 * @example `{ role: "user", content: [{ type: "text", text: "Describe" }, await imagePart("cat.png")] }`
 */
export async function imagePart(image: Uint8Array | string): Promise<ContentPart> {
  const bytes = typeof image === "string" ? await Deno.readFile(image) : image;
  const mime = bytes[0] === 0x89
    ? "image/png"
    : bytes[0] === 0xff
    ? "image/jpeg"
    : bytes[0] === 0x47
    ? "image/gif"
    : "image/bmp";
  return { type: "image_url", image_url: { url: `data:${mime};base64,${toBase64(bytes)}` } };
}

/**
 * An `input_audio` content part from WAV / MP3 / FLAC bytes or a file path (format detected).
 * @example `{ role: "user", content: [{ type: "text", text: "Transcribe" }, await audioPart("clip.wav")] }`
 */
export async function audioPart(audio: Uint8Array | string): Promise<ContentPart> {
  const bytes = typeof audio === "string" ? await Deno.readFile(audio) : audio;
  const format = audioFormat(bytes);
  if (!format) throw new AiCoreError("audioPart: data is not WAV, MP3 or FLAC");
  return { type: "input_audio", input_audio: { data: toBase64(bytes), format } };
}

/** Container from magic bytes; same rules as the native library (and llama.cpp mtmd). */
function audioFormat(b: Uint8Array): "wav" | "mp3" | "flac" | null {
  const ascii = (from: number, to: number) => String.fromCharCode(...b.subarray(from, to));
  if (b.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "WAVE") return "wav";
  if (ascii(0, 4) === "fLaC") return "flac";
  if (ascii(0, 3) === "ID3" || (b[0] === 0xff && (b[1] & 0xe0) === 0xe0)) return "mp3";
  return null;
}

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

/** Dot product; equals cosine similarity for normalized vectors. */
export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
