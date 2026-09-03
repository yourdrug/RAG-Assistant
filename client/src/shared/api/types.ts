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
  articles?: string[];
  max_score?: number;
  edited?: boolean;
  edited_at?: string | null;
  manual?: boolean;
  document_id?: number;
  content_hashes?: string[];
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

export interface ConversationListItem {
  id: number;
  title?: string | null;
  created_at?: string | null;
  message_count: number;
}

export interface ConversationListResponse {
  conversations: ConversationListItem[];
}

// ─── Documents ───────────────────────────────────────────────────────────────

export type DocumentVisibility =
  | "internal_public"
  | "internal_group"
  | "internal_private"
  | "client_private";

export type DocumentStatus = "pending" | "processing" | "indexing" | "done" | "failed";

export type DocumentDomain = "legal" | "general";

export interface DocumentResponse {
  id: number;
  filename: string;
  source_path: string;
  visibility: DocumentVisibility;
  owner_id?: number | null;
  group_id?: number | null;
  status: DocumentStatus;
  doc_domain: DocumentDomain;
  source_type?: "file" | "manual";
  has_manual_edits?: boolean;
  in_search_scope?: boolean;
  error_message?: string | null;
  warning_message?: string | null;
  quality_score?: number | null;
  chunks?: number | null;
  chars?: number | null;
  creation_date?: string | null;
  indexed_at?: string | null;
  outbox_pending?: number;
  outbox_failed?: number;
  outbox_failed_details?: {
    operation: string;
    attempts: number;
    max_attempts: number;
    last_error: string | null;
  }[];
}

export interface UploadStatusResponse {
  status: string;
  document_id: number;
  filename: string;
}

// ─── Chunks ──────────────────────────────────────────────────────────────────

export interface ChunkResponse {
  id: number;
  document_id: number;
  chunk_index: number;
  content: string;
  filename?: string;
  visibility?: string;
  doc_domain?: string;
  owner_id?: number | null;
  group_id?: number | null;
  edited_at?: string | null;
  edited_by?: number | null;
  manual?: boolean;
  creation_date?: string | null;
  content_hash?: string | null;
  warning?: string | null;
}

export interface ChunkCreateRequest {
  content: string;
  page?: number | null;
  section?: string | null;
}

export interface ChunkEditRequest {
  content: string;
}

export interface ChunkListResponse {
  chunks: ChunkResponse[];
  total: number;
  document_id: number;
}

export interface ManualDocumentRequest {
  title: string;
  visibility: string;
  group_id?: number | null;
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
  llm_provider: string;
  checks: {
    api: HealthCheck;
    qdrant: HealthCheck;
    llm: HealthCheck;
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

export interface BenchmarkResultSummary {
  filename: string;
  model: string | null;
  total_questions: number;
  total_time_sec: number;
  hit_rate: number | null;
  avg_mrr: number | null;
  avg_faithfulness: number | null;
  avg_relevancy: number | null;
  avg_correctness: number | null;
  avg_similarity: number | null;
}

export interface BenchmarkResultsListResponse {
  results: BenchmarkResultSummary[];
  total: number;
}

export interface BenchmarkResultDetail {
  id: number;
  summary: BenchmarkResultSummary;
  per_question_results: Array<{
    id: string | number;
    question: string;
    answer: string;
    expected_answer?: string | null;
    faithfulness?: number;
    relevancy?: number;
    correctness?: number | null;
    hit_rate?: number | null;
    mrr?: number | null;
    avg_similarity?: number;
    latency_sec?: number;
  }> | null;
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
  http_requests: Record<string, number | Record<string, number>>;
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

// ─── Chat Logs (Q&A quality tracking) ────────────────────────────────────────

export interface ChatLogEntry {
  id: number;
  creation_date: string;
  user_id?: number | null;
  conversation_id?: number | null;
  question: string;
  answer: string;
  sources?: Source[] | null;
  latency_ms?: number | null;
  model_used?: string | null;
  breadth?: string | null;
  domain?: string | null;
  retrieval_count?: number | null;
  reranker_score?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
}

export interface ChatLogsResponse {
  logs: ChatLogEntry[];
  total: number;
}

// ─── Exact Substring Search (pg_trgm) ────────────────────────────────────────

export interface ExactSearchResult {
  chunk_id: number;
  document_id: number;
  filename: string;
  content: string;
  chunk_index: number;
}

export interface ExactSearchResponse {
  query: string;
  results: ExactSearchResult[];
  total: number;
}

// ─── Benchmark Lab ────────────────────────────────────────────────────────

export interface BenchmarkQuestion {
  id: number;
  question: string;
  expected_answer?: string | null;
  source_hint?: string | null;
  tags?: string[] | null;
  dataset: string;
  is_active: boolean;
  created_by?: number | null;
  notes?: string | null;
  creation_date?: string | null;
}

export interface BenchmarkQuestionsListResponse {
  questions: BenchmarkQuestion[];
  total: number;
}

export interface BenchmarkQuestionCreate {
  question: string;
  expected_answer?: string | null;
  source_hint?: string | null;
  tags?: string[] | null;
  dataset?: string;
  notes?: string | null;
}

export interface BenchmarkQuestionUpdate {
  question?: string | null;
  expected_answer?: string | null;
  source_hint?: string | null;
  tags?: string[] | null;
  dataset?: string | null;
  is_active?: boolean | null;
  notes?: string | null;
}

export interface SweepCreateRequest {
  strategy: "grid" | "random" | "successive_halving";
  search_space: Record<string, { values?: number[]; min?: number; max?: number; step?: number }>;
  objective_weights?: Record<string, number>;
  dataset?: string;
  top_n_llm?: number;
}

export interface SweepResponse {
  id: number;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  strategy: string;
  search_space: Record<string, any>;
  objective_weights: Record<string, number>;
  dataset: string;
  top_n_llm: number;
  total_configs: number;
  evaluated_configs: number;
  best_run_id?: number | null;
  job_id?: number | null;
  creation_date?: string | null;
}

export interface SweepsListResponse {
  sweeps: SweepResponse[];
  total: number;
}

export interface BenchmarkRun {
  id: number;
  sweep_id?: number | null;
  config_json: Record<string, any>;
  summary_metrics: Record<string, any>;
  duration_sec: number;
  llm_evaluated: boolean;
  dataset: string;
  filename?: string | null;
  creation_date?: string | null;
}

export interface BenchmarkRunsListResponse {
  runs: BenchmarkRun[];
  total: number;
}

export interface RunCompareResponse {
  runs: BenchmarkRun[];
  diff: Record<string, Array<{ run_id: number; value: any }>>;
}

export interface BenchmarkHistoryPoint {
  run_id: number;
  creation_date?: string | null;
  metrics: Record<string, any>;
  config_summary: Record<string, any>;
  dataset: string;
  llm_evaluated: boolean;
}

export interface BenchmarkHistoryResponse {
  points: BenchmarkHistoryPoint[];
  total: number;
}

export interface SweepProgressEvent {
  evaluated: number;
  total: number;
  latest?: {
    config: Record<string, any>;
    avg_hit_rate?: number;
    avg_mrr?: number;
    composite_score?: number;
  } | null;
  done?: boolean;
  best_run_id?: number | null;
}

// ─── Admin Quality ────────────────────────────────────────────────────────────

export interface DocumentQualityItem {
  id: number;
  filename: string;
  status: string;
  quality_score?: number | null;
  warning_message?: string | null;
  chunks?: number | null;
  chars?: number | null;
  indexed_at?: string | null;
}

export interface DocumentQualityListResponse {
  documents: DocumentQualityItem[];
  total: number;
}

export interface PageDiagnostic {
  page: number;
  type: "text" | "scan" | "garbled" | "empty" | "table";
  chars: number;
  description: string;
}

export interface DocumentDiagnoseResponse {
  document_id: number;
  filename: string;
  total_pages: number;
  pages: PageDiagnostic[];
  summary: {
    text: number;
    scan: number;
    garbled: number;
    empty: number;
    table: number;
  };
}

export interface DryRunPageResult {
  page: number;
  type: "text" | "scan" | "garbled" | "empty" | "table" | "image_only";
  content_type: string;
  chars: number;
  preview: string;
  full_text: string;
  problem_spans: [number, number][];
  previous_type: string | null;
  image_available: boolean;
  unit_kind: "page" | "section" | "document";
  label: string;
}

export interface DryRunResponse {
  filename: string;
  total_pages: number;
  pages: DryRunPageResult[];
  total_chars: number;
  quality_score: number;
  warning?: string | null;
  full_text_preview: string;
  summary: {
    text: number;
    scan: number;
    garbled: number;
    empty: number;
    table: number;
    image_only?: number;
  };
  suggestion: string | null;
  preview_id: string;
}

export interface PageImageResponse {
  image_base64: string;
  page: number;
}

export interface IndexFromPreviewResponse {
  document_id: number;
  filename: string;
  status: string;
}
