// TypeScript types matching backend Pydantic schemas

// ─── Auth ────────────────────────────────────────────────────────────────────

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
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
  depth?: "short" | "detailed" | null;
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
  sources?: Source[];
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
  source_path: string;
  visibility: DocumentVisibility;
  owner_id?: number | null;
  group_id?: number | null;
  status: DocumentStatus;
  error_message?: string | null;
  warning_message?: string | null;
  chunks?: number | null;
  chars?: number | null;
  creation_date?: string | null;
  indexed_at?: string | null;
}

export interface UploadStatusResponse {
  status: string;
  document_id: number;
  filename: string;
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

// ─── API Keys ─────────────────────────────────────────────────────────────────

export interface ApiKeyCreateRequest {
  name?: string | null;
}

export interface ApiKeyCreatedResponse {
  id: number;
  api_key: string; // shown only in this response, never stored
  key_prefix: string;
  name?: string | null;
  created_at: string;
}

export interface ApiKeyResponse {
  id: number;
  key_prefix: string;
  name?: string | null;
  created_at: string;
  revoked_at?: string | null;
  last_used_at?: string | null;
  is_active: boolean;
}

// ─── Health ──────────────────────────────────────────────────────────────────

export interface HealthCheck {
  status: string;
  latency_ms?: number | null;
  models?: string[] | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
  checks: {
    api: HealthCheck;
    qdrant: HealthCheck;
    ollama: HealthCheck;
    postgres: HealthCheck;
  };
  background_jobs: { running: number };
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

// ─── Background Jobs ───────────────────────────────────────────────────────

export interface JobResponse {
  id: number;
  job_type: string;
  status: "pending" | "running" | "done" | "failed";
  related_id?: number | null;
  request_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  error_message?: string | null;
  creation_date?: string | null;
}

export interface JobsListResponse {
  total: number;
  jobs: JobResponse[];
}

export interface JobsStatsResponse {
  total: number;
  by_status: Record<string, number>;
}

// ─── Monitoring / Metrics ──────────────────────────────────────────────────

export interface MetricsResponse {
  db_pool: Record<string, number>;
  qdrant: Record<string, number>;
  bm25: Record<string, number>;
  ollama: Array<{ model: string; gpu_bytes: number; ram_bytes: number }>;
  rag: Record<string, number | Record<string, number>>;
  ingestion: Record<string, number | Record<string, number>>;
}

// ─── Logs ──────────────────────────────────────────────────────────────────

export interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  request_id: string;
  message: string;
  filename?: string | null;
  lineno?: number | null;
}

export interface LogsResponse {
  logs: LogEntry[];
  total: number;
}
