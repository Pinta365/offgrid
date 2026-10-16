// JSON shapes exchanged with the native library (mirror the Rust serde types).

/** How many layers to offload to the GPU: all that fit (default), none (CPU only), or a count. */
export type GpuLayers = "all" | "none" | { count: number };

/** Options for `Offgrid.loadModel`. */
export interface ModelParams {
  gpu_layers?: GpuLayers;
  /** Restrict offloading to these device names, e.g. `["Vulkan0"]`. */
  devices?: string[] | null;
  /** Multimodal projector (`mmproj-*.gguf`); enables image and/or audio input. */
  mmproj?: string | null;
}

/** Options for `Model.createChat`. */
export interface SessionParams {
  /** Context size in tokens; 0 = model default capped at 8192. */
  n_ctx?: number;
  /** Tokens per decode call while processing the prompt. */
  n_batch?: number;
  /** CPU threads; default half the hardware threads. */
  n_threads?: number | null;
  /** Reuse cached chunks of at least this many tokens that moved to an earlier position; 0 = off. */
  cache_reuse?: number;
  /** KV cache type: "f16" (default), "q8_0" (half the memory) or "q4_0" (a quarter). */
  cache_type?: "f16" | "q8_0" | "q4_0";
}

/** Options for `Model.createEmbedder`. */
export interface EmbedParams {
  /** Max tokens per input; 0 = model default capped at 8192. */
  n_ctx?: number;
  /** Max inputs embedded together in one batch. */
  max_batch_inputs?: number;
  /** L2-normalize the vectors (default true), so `dot` gives cosine similarity. */
  normalize?: boolean;
  /** CPU threads; default half the hardware threads. */
  n_threads?: number | null;
}

/** Sampling settings (OpenAI / llama.cpp server names). */
export interface SamplingParams {
  /** <= 0 means greedy. */
  temperature?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  repeat_penalty?: number;
  repeat_last_n?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  /** Fixed seed for reproducible sampling; default random. */
  seed?: number | null;
  /** GBNF grammar with a `root` rule. */
  grammar?: string | null;
}

/** Usually "system" | "user" | "assistant"; templates may accept others. */
export type Role = string;

/**
 * OpenAI content parts. Images: `image_url` with a `data:` URL (see `imagePart`); audio:
 * `input_audio` with base64 data and its format (see `audioPart`). Remote URLs are refused.
 */
export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "input_audio"; input_audio: { data: string; format: "wav" | "mp3" | "flac" } };

/** An OpenAI-style chat message. */
export interface ChatMessage {
  role: Role;
  /** Text, parts (text + images), or `null` for assistant turns that only call tools. */
  content: string | ContentPart[] | null;
  /** Reasoning (`<think>`) of an earlier assistant turn. */
  reasoning_content?: string;
  /** Calls made by an assistant turn. */
  tool_calls?: ToolCall[];
  /** For `role: "tool"`: the call this result answers. */
  tool_call_id?: string;
  /** For `role: "tool"`: the function name. */
  name?: string;
}

/** A function the model may call. */
export interface ToolFunction {
  name: string;
  description?: string;
  /** JSON Schema of the arguments object. */
  parameters?: JsonSchema;
}

/** OpenAI `tools` entry. */
export interface Tool {
  type: "function";
  function: ToolFunction;
}

/** A tool call made by the model. */
export interface ToolCall {
  id: string;
  type: "function";
  /** `arguments` is JSON text (OpenAI convention). */
  function: { name: string; arguments: string };
}

/** Whether and which tool the model must call. */
export type ToolChoice = "none" | "auto" | "required" | { type: "function"; function: { name: string } };

/** A chat completion request (OpenAI format). Requests are stateless: send the whole conversation each time. */
export interface ChatRequest extends SamplingParams {
  messages: ChatMessage[];
  /** Max tokens to generate; default until the end of the reply or the context is full. */
  max_tokens?: number | null;
  stop?: string[];
  /** Jinja template text or a llama.cpp built-in name (e.g. "chatml"); default = the model's own template. */
  chat_template?: string | null;
  /** Extra chat template variables, e.g. `{ enable_thinking: false }`. */
  chat_template_kwargs?: Record<string, unknown> | null;
  /** Constrain output to JSON (OpenAI-compatible). Cannot be combined with `grammar`. */
  response_format?: ResponseFormat | null;
  /** Functions the model may call. */
  tools?: Tool[];
  /** Default `"auto"`. `"required"` / a named function force a call (grammar-enforced where supported). */
  tool_choice?: ToolChoice | null;
  /** Allow several calls per turn (default true). */
  parallel_tool_calls?: boolean | null;
}

/** JSON Schema object (draft 2020-12 subset supported by the grammar converter). */
export type JsonSchema = Record<string, unknown>;

/** OpenAI `response_format`; JSON formats are enforced with a grammar. */
export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object"; schema?: JsonSchema }
  | { type: "json_schema"; json_schema: { name?: string; schema: JsonSchema; strict?: boolean } };

/** Raw text completion request for `ChatSession.generate` (no chat template). */
export interface GenerateRequest extends SamplingParams {
  prompt: string;
  max_tokens?: number | null;
  stop?: string[];
}

/** Why generation stopped. */
export type FinishReason = "stop" | "length" | "cancelled" | "tool_calls";

/** Token counts and timings of a request. */
export interface Usage {
  prompt_tokens: number;
  /** Prompt tokens reused from the KV cache of the previous request. */
  cached_tokens: number;
  completion_tokens: number;
  /** Prompt processing time in milliseconds. */
  prompt_ms: number;
  /** Generation time in milliseconds. */
  generation_ms: number;
}

/** Result of a chat completion. */
export interface ChatResponse {
  /** Reply text without reasoning or tool-call markup. */
  content: string;
  /** `<think>` reasoning, if any. */
  reasoning_content?: string;
  /** Present when `finish_reason` is `"tool_calls"`. */
  tool_calls?: ToolCall[];
  finish_reason: FinishReason;
  usage: Usage;
}

/** A compute device llama.cpp can use. */
export interface DeviceInfo {
  /** Backend device name for `ModelParams.devices`, e.g. `Vulkan0`. */
  name: string;
  description: string;
  kind: "cpu" | "gpu" | "igpu" | "accel" | "other";
  /** Bytes. */
  memory_free: number;
  /** Bytes. */
  memory_total: number;
}

/** Metadata of a loaded model. */
export interface ModelInfo {
  name: string | null;
  architecture: string | null;
  description: string;
  n_params: number;
  size_bytes: number;
  /** Context length the model was trained with. */
  n_ctx_train: number;
  n_embd: number;
  n_layer: number;
  has_encoder: boolean;
  has_decoder: boolean;
  chat_template: string | null;
  /** Image input available. */
  vision: boolean;
  /** Audio input available. */
  audio: boolean;
  /** Fill-in-the-middle tokens present (code infill). */
  infill: boolean;
}

/** Versions of the native library. */
export interface VersionInfo {
  offgrid: string;
  /** C ABI version; must equal `ABI_VERSION`. */
  abi: number;
  llama_cpp: string;
}

/** Model files for `Offgrid.loadImageModel`: a single-file checkpoint (`model`), or a
 * `diffusion_model` with the encoders and VAE its family needs (Z-Image: `text_encoder` =
 * Qwen3-4B, `vae`). */
export interface ImageModelParams {
  /** Single-file checkpoint (SD 1.x/2.x/XL). */
  model?: string;
  /** Diffusion model alone (Z-Image, FLUX, ...). */
  diffusion_model?: string;
  /** LLM text encoder (Z-Image and Qwen-Image use a Qwen GGUF). */
  text_encoder?: string;
  clip_l?: string;
  clip_g?: string;
  t5xxl?: string;
  vae?: string;
  /** Use the GPU when available (default true); false runs on the CPU. */
  gpu?: boolean;
  /** GiB of GPU memory the model may use; weights beyond it stay in RAM. */
  max_vram_gib?: number;
  /** Run the text encoder on the CPU (~0.4 s per image); keeps its weights off the GPU. */
  text_encoder_on_cpu?: boolean;
  /** Decode in tiles: a smaller GPU memory peak, so weights stay resident when memory is tight. */
  vae_tiling?: boolean;
  /** Flash attention in the diffusion model (default true). */
  flash_attn?: boolean;
  /** CPU threads; default half the hardware threads. */
  n_threads?: number;
}

/** An image generation request; unset options use the model's defaults. */
export interface ImageRequest {
  prompt: string;
  negative_prompt?: string;
  /** Pixels, a multiple of 16 (default 1024). */
  width?: number;
  /** Pixels, a multiple of 16 (default 1024). */
  height?: number;
  /** Sampling steps (Z-Image-Turbo: 8). */
  steps?: number;
  /** Classifier-free guidance scale (Z-Image-Turbo: 1.0). */
  cfg_scale?: number;
  /** Default random. */
  seed?: number;
  /** Sampler, e.g. `euler`, `euler_a`, `dpm++2m`. */
  sampler?: string;
  /** Noise schedule, e.g. `discrete`, `karras`, `simple`. */
  scheduler?: string;
  /** Image-to-image: start from this PNG (bytes, or base64). The output has its size. */
  init_image?: Uint8Array | string;
  /** How far to move away from `init_image`: 0 keeps it, 1 ignores it. Default 0.75. */
  strength?: number;
}

/** Image generation progress. Weights load lazily, so `loading` can follow `sampling`. */
export interface ImageProgress {
  phase: "loading" | "sampling" | "decoding";
  /** Steps or items done. */
  done: number;
  total: number;
}
