// TypeScript types matching backend Pydantic schemas

// ─── Auth ────────────────────────────────────────────────────────────────────

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: "admin" | "user";
  kind: "internal" | "client";
}

export interface UserResponse {
  id: number;
  email: string;
  role: "admin" | "user";
  kind: "internal" | "client";
  is_active: boolean;
}

export interface CreateUserRequest {
  email: string;
  password: string;
  role?: string;
  kind?: string;
}

// ─── Chat ────────────────────────────────────────────────────────────────────

export interface ChatRequest {
  question: string;
  conversation_id?: number | null;
}

export interface ChatResponse {
  answer: string;
  conversation_id: number;
  sources: Source[];
}

export interface Source {
  source: string;
  pages: number[];
}

// ─── Conversations ───────────────────────────────────────────────────────────

export interface NewConversationResponse {
  conversation_id: number;
}

export interface MessageResponse {
  role: "user" | "assistant";
  content: string;
}

export interface ConversationHistoryResponse {
  conversation_id: number;
  messages: MessageResponse[];
}

// ─── Documents ───────────────────────────────────────────────────────────────

export type DocumentVisibility =
  | "internal_public"
  | "internal_group"
  | "internal_private"
  | "client_private";

export type DocumentStatus = "pending" | "processing" | "done" | "failed";

export interface DocumentResponse {
  id: number;
  filename: string;
  visibility: DocumentVisibility;
  status: DocumentStatus;
  error_message?: string | null;
  chunks?: number | null;
  chars?: number | null;
}

// ─── Ingest ──────────────────────────────────────────────────────────────────

export interface UploadResponse {
  files: string[];
}

export interface IngestStatusResponse {
  status: string;
  mode?: string | null;
  file?: string | null;
  force?: boolean | null;
  docs_dir?: string | null;
}

export interface IngestRegistryItem {
  filename: string;
  chunks: number;
  chars: number;
  indexed_at: string;
  source: string;
}

export interface IngestRegistryResponse {
  total_files: number;
  total_chunks: number;
  files: IngestRegistryItem[];
}

// ─── Groups ──────────────────────────────────────────────────────────────────

export interface CreateGroupRequest {
  name: string;
}

export interface GroupResponse {
  id: number;
  name: string;
}

export interface GroupMemberResponse {
  id: number;
  email: string;
}

export interface GroupMemberRequest {
  user_id: number;
}

// ─── Clients ─────────────────────────────────────────────────────────────────

export interface AssignClientRequest {
  internal_user_id: number;
}

export interface ClientAssignmentResponse {
  internal_user_id: number;
  email: string;
  assigned_at: string;
}

// ─── Health ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  api: string;
  qdrant: string;
  ollama: string;
  ollama_models?: string[] | null;
}

// ─── Benchmark ───────────────────────────────────────────────────────────────

export interface BenchmarkRequest {
  questions_path?: string | null;
  out_dir?: string | null;
  top_k?: number | null;
  judge_model?: string | null;
}

export interface BenchmarkResponse {
  status: string;
}
