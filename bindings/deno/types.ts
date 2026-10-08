/** JSON shapes exchanged with the native library (mirror the Rust serde types). */

export type GpuLayers = "all" | "none" | { count: number };

export interface ModelParams {
  gpu_layers?: GpuLayers;
  /** Restrict offloading to these device names, e.g. `["Vulkan0"]`. */
  devices?: string[] | null;
  /** Multimodal projector (`mmproj-*.gguf`); enables image and/or audio input. */
  mmproj?: string | null;
}

export interface SessionParams {
  /** Context size in tokens; 0 = model default capped at 8192. */
  n_ctx?: number;
  n_batch?: number;
  n_threads?: number | null;
}

export interface EmbedParams {
  n_ctx?: number;
  max_batch_inputs?: number;
  normalize?: boolean;
  n_threads?: number | null;
}

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

export interface ToolFunction {
  name: string;
  description?: string;
  /** JSON Schema of the arguments object. */
  parameters?: JsonSchema;
}

export interface Tool {
  type: "function";
  function: ToolFunction;
}

export interface ToolCall {
  id: string;
  type: "function";
  /** `arguments` is JSON text (OpenAI convention). */
  function: { name: string; arguments: string };
}

export type ToolChoice = "none" | "auto" | "required" | { type: "function"; function: { name: string } };

export interface ChatRequest extends SamplingParams {
  messages: ChatMessage[];
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

export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object"; schema?: JsonSchema }
  | { type: "json_schema"; json_schema: { name?: string; schema: JsonSchema; strict?: boolean } };

export interface GenerateRequest extends SamplingParams {
  prompt: string;
  max_tokens?: number | null;
  stop?: string[];
}

export type FinishReason = "stop" | "length" | "cancelled" | "tool_calls";

export interface Usage {
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  prompt_ms: number;
  generation_ms: number;
}

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

export interface DeviceInfo {
  name: string;
  description: string;
  kind: "cpu" | "gpu" | "igpu" | "accel" | "other";
  memory_free: number;
  memory_total: number;
}

export interface ModelInfo {
  name: string | null;
  architecture: string | null;
  description: string;
  n_params: number;
  size_bytes: number;
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
}

export interface VersionInfo {
  aicore: string;
  abi: number;
  llama_cpp: string;
}
