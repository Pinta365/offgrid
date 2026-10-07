/** JSON shapes exchanged with the native library (mirror the Rust serde types). */

export type GpuLayers = "all" | "none" | { count: number };

export interface ModelParams {
  gpu_layers?: GpuLayers;
  /** Restrict offloading to these device names, e.g. `["Vulkan0"]`. */
  devices?: string[] | null;
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

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface ChatRequest extends SamplingParams {
  messages: ChatMessage[];
  max_tokens?: number | null;
  stop?: string[];
  /** Built-in template name (e.g. "chatml") or template text; default = model's own. */
  chat_template?: string | null;
}

export interface GenerateRequest extends SamplingParams {
  prompt: string;
  max_tokens?: number | null;
  stop?: string[];
}

export type FinishReason = "stop" | "length" | "cancelled";

export interface Usage {
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  prompt_ms: number;
  generation_ms: number;
}

export interface ChatResponse {
  content: string;
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
}

export interface VersionInfo {
  aicore: string;
  abi: number;
  llama_cpp: string;
}
