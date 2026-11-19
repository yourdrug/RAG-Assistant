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
  role: "admin" | "curator" | "user";
  kind: "internal" | "client";
  is_active: boolean;
  client_space_ids?: number[];
}

export interface UserListResponse {
  total: number;
  limit: number;
  offset: number;
  users: UserResponse[];
}

export interface CreateUserRequest {
  email: string;
  password: string;
  role?: string;
  kind?: string;
}

export interface ChangeRoleRequest {
  role: string;
}

export interface CuratorScopeResponse {
  managed_client_ids: number[];
  managed_internal_ids: number[];
  managed_group_ids: number[];
}

// ─── Chat ────────────────────────────────────────────────────────────────────

export interface ChatRequest {
  question: string;
  conversation_id?: number | null;
  depth?: "narrow" | "broad" | null;
  /** Knowledge space of the question; a conversation keeps the space of its first question. */
  space_id?: number | null;
}

export interface ChatResponse {
  message_id: number;
  answer: string;
  conversation_id: number;
  sources: Source[];
}

export interface Source {
  citation_id?: number;
  source: string;
  pages: number[];
  articles?: string[];
  max_score?: number;
  edited?: boolean;
  edited_at?: string | null;
  manual?: boolean;
  document_id?: number;
  content_hashes?: string[];
  // Regulatory act versioning (versioned domains only)
  act_number?: string;
  effective_from?: string | null;
  effective_to?: string | null;
}

// ─── Conversations ───────────────────────────────────────────────────────────

export interface NewConversationResponse {
  conversation_id: number;
}

export type FeedbackRating = "up" | "down";

export type FeedbackReason =
  | "no_answer"
  | "wrong_fact"
  | "irrelevant_sources"
  | "incomplete"
  | "outdated"
  | "too_long"
  | "other";

export interface MessageFeedback {
  rating: FeedbackRating;
  reasons: FeedbackReason[];
  comment?: string | null;
  update_date?: string | null;
}

export interface FeedbackRequest {
  rating: FeedbackRating;
  reasons?: FeedbackReason[];
  comment?: string | null;
}

export interface MessageResponse {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  /** The current user's reaction; assistant messages only. */
  feedback?: MessageFeedback | null;
}

export interface ConversationHistoryResponse {
  conversation_id: number;
  messages: MessageResponse[];
  next_cursor: number | null;
}

export interface ConversationListItem {
  id: number;
  title?: string | null;
  created_at?: string | null;
  message_count: number;
  space_id?: number | null;
}

export interface ConversationListResponse {
  conversations: ConversationListItem[];
}

// ─── Documents ───────────────────────────────────────────────────────────────

export type DocumentVisibility =
  | "internal_public"
  | "internal_group"
  | "internal_private"
  | "client_private"
  | "client_public";

export type DocumentStatus = "pending" | "processing" | "indexing" | "done" | "failed";

// Profile key of the active domain pack (see GET /documents/domains).
export type DocumentDomain = string;

export interface DocumentDomainOption {
  key: string;
  display_name: string;
}

export interface DocumentResponse {
  id: number;
  filename: string;
  source_path: string;
  visibility: DocumentVisibility;
  owner_id?: number | null;
  group_id?: number | null;
  space_id?: number | null;
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
  section?: string | null;
  heading?: string | null;
  heading_level?: number | null;
  content_type?: string | null;
  doc_title?: string | null;
  doc_type?: string | null;
}

export interface IndexFieldComparison {
  name: string;
  expected: string | number | null;
  actual: string | number | null;
  matches: boolean;
}

/** Admin-only: the chunk's point in Qdrant compared with Postgres. */
export interface ChunkIndexStateResponse {
  chunk_id: number;
  document_id: number;
  document_status: string;
  indexed: boolean;
  in_sync: boolean;
  dense_dim: number | null;
  sparse_terms: number | null;
  embedding_model: string | null;
  content_matches: boolean | null;
  fields: IndexFieldComparison[];
  payload_metadata: Record<string, unknown>;
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

export interface ChunkCursorListResponse {
  chunks: ChunkResponse[];
  next_cursor: string | null;
  prev_cursor: string | null;
  document_id: number;
}

export interface ManualDocumentRequest {
  title: string;
  visibility: string;
  group_id?: number | null;
  space_id?: number | null;
  client_id?: number | null;
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
  ingestion_key: string;
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

// ─── Knowledge spaces ────────────────────────────────────────────────────────

/** A space selects the searched documents and the pack's rules; it never grants access. */
export interface KnowledgeSpace {
  id: number;
  name: string;
  pack_key: string;
  can_upload: boolean;
  /** Groups admitted to fill the space; returned to ADMIN only. */
  upload_group_ids?: number[] | null;
}

export interface CreateKnowledgeSpaceRequest {
  name: string;
  pack_key: string;
  upload_group_ids: number[];
}

export interface UpdateKnowledgeSpaceRequest {
  name?: string;
  upload_group_ids?: number[];
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
  key: string; // shown only in this response, never stored
  key_prefix: string;
  name?: string | null;
}

export interface ApiKeyResponse {
  id: number;
  key_prefix: string;
  name?: string | null;
  creation_date?: string | null;
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
    llm: HealthCheck;
    postgres: HealthCheck;
  };
  background_jobs: { running: number };
}

// ─── Benchmark ───────────────────────────────────────────────────────────────

export type LLMReplayMode = "off" | "record" | "replay_only" | "refresh";

export interface BenchmarkRequest {
  space_id?: number | null;
  dataset?: string;
  top_k?: number | null;
  judge_model?: string | null;
  llm_replay_mode?: LLMReplayMode;
}

export interface BenchmarkResponse {
  status: string;
}

export interface BenchmarkResultSummary {
  id: number;
  config_json: Record<string, unknown>;
  summary_metrics: Record<string, unknown>;
  duration_sec: number;
  llm_evaluated: boolean;
  dataset: string;
  sweep_id?: number | null;
  creation_date?: string | null;
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
    faithfulness?: number | null;
    relevancy?: number | null;
    correctness?: number | null;
    hit_rate?: number | null;
    mrr?: number | null;
    source_hit_rate?: number | null;
    source_mrr?: number | null;
    evidence_metrics?: Record<string, number | null>;
    avg_similarity?: number;
    latency_sec?: number;
  }> | null;
}

// ─── Background Jobs ───────────────────────────────────────────────────────

export interface JobResponse {
  id: number;
  job_type: string;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  attempt: number;
  retryable: boolean;
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

export type AnswerOutcome =
  | "generated"
  | "grounded"
  | "cache_hit"
  | "not_found"
  | "out_of_domain"
  | "input_blocked"
  | "output_blocked"
  | "error"
  | "cancelled";

export interface ConfigSnapshot {
  data: Record<string, unknown> | null;
  current: Record<string, unknown> | null;
  diff: { field: string; saved: unknown; current: unknown }[] | null;
}

export interface ConversationTurn {
  id: number | null;
  role: "user" | "assistant";
  content: string;
  creation_date: string;
}

export interface ChatLogDetail {
  entry: ChatLogEntry;
  snapshot: ConfigSnapshot;
  context: ConversationTurn[];
}

export type ReviewStatus =
  | "new"
  | "benchmark_candidate"
  | "in_benchmark"
  | "content_gap"
  | "pipeline_issue"
  | "wont_fix";

export interface ChatLogReview {
  status: ReviewStatus;
  note?: string | null;
  reviewer_id?: number | null;
  benchmark_question_id?: number | null;
  update_date?: string | null;
}

export interface BenchmarkDraft {
  question: string;
  history: { role: "user" | "assistant"; content: string }[];
  as_of_date?: string | null;
  source_hint?: string | null;
  datasets: string[];
  benchmark_question_id?: number | null;
}

export interface BenchmarkTransferRequest {
  dataset: string;
  question: string;
  history: BenchmarkDraft["history"];
  expected_answer: string | null;
  source_hint: string;
  as_of_date: string;
  annotations: Record<string, unknown> | null;
  tags: string[];
  notes: string | null;
  is_active: boolean;
}

export interface ChatLogGroupStats {
  key: string | number | null;
  total: number;
  down: number;
  not_found: number;
  errors: number;
  cache_hits: number;
  latency_p50: number | null;
  latency_p95: number | null;
}

export interface ChatLogStats {
  total: number;
  cache_hits: number;
  up: number;
  down: number;
  latency_p50: number | null;
  latency_p95: number | null;
  ttft_p50: number | null;
  ttft_p95: number | null;
  input_tokens: number;
  output_tokens: number;
  aux_tokens: number;
  outcomes: Record<string, number>;
  error_codes: Record<string, number>;
  reasons: Record<string, number>;
  by_day: ChatLogGroupStats[];
  by_space: ChatLogGroupStats[];
  by_model: ChatLogGroupStats[];
  by_config_hash: ChatLogGroupStats[];
}

export interface ChatLogEntry {
  id: number;
  creation_date: string;
  user_id?: number | null;
  user_email?: string | null;
  conversation_id?: number | null;
  question: string;
  answer: string;
  sources?: Source[] | null;
  latency_ms?: number | null;
  model_used?: string | null;
  breadth?: string | null;
  domain?: string | null;
  outcome: AnswerOutcome;
  space_id: number;
  message_id?: number | null;
  request_id?: string | null;
  pack_key?: string | null;
  depth_requested?: string | null;
  as_of_date?: string | null;
  condensed_query?: string | null;
  sub_queries?: string[] | null;
  refined_query?: string | null;
  error_code?: string | null;
  confidence?: number | null;
  sources_count?: number | null;
  candidates_count?: number | null;
  top_rerank_score?: number | null;
  aux_tokens?: number | null;
  ttft_ms?: number | null;
  retrieval_ms?: number | null;
  rerank_ms?: number | null;
  generation_ms?: number | null;
  config_hash?: string | null;
  config_version?: string | null;
  evidence_profile?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  feedback?: MessageFeedback | null;
  review?: ChatLogReview | null;
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
  limit: number;
  offset: number;
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
  is_active?: boolean;
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
  space_id?: number | null;
  evaluation_mode?: "fast" | "full";
  strategy: "grid" | "random" | "optuna_tpe";
  search_space: Record<string, { values?: number[]; min?: number; max?: number; step?: number }>;
  objective_weights?: Record<string, number>;
  dataset?: string;
  top_n_llm?: number;
  judge_model?: string | null;
  llm_replay_mode?: LLMReplayMode;
}

export interface SweepResponse {
  space_id?: number | null;
  evaluation_mode: "fast" | "full";
  id: number;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  strategy: SweepCreateRequest["strategy"];
  search_space: Record<string, any>;
  objective_weights: Record<string, number>;
  dataset: string;
  top_n_llm: number;
  judge_model?: string | null;
  llm_replay_mode?: LLMReplayMode;
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
  /** From the oldest selected run to each newer one. */
  diagnosis_transitions?: Array<{
    from_run_id: number;
    to_run_id: number;
    compared: number;
    matrix: Record<string, Record<string, number>>;
    changes: Array<{
      key: string;
      id: string | number | null;
      question: string | null;
      from: string;
      to: string;
    }>;
  }>;
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

export interface SweepProgressSnapshot {
  phase: string;
  evaluated: number;
  total: number;
  latest: NonNullable<SweepProgressEvent["latest"]>;
  updated_at: string;
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

// Regulatory Act Versions — admin review
export interface ActVersionReviewItem {
  id: number;
  act_id: number | null;
  document_id: number;
  document_filename: string;
  effective_from: string | null;
  effective_to: string | null;
  is_current: boolean;
  date_source: string;
  date_confidence: number | null;
}

export interface ActSummary {
  id: number;
  act_type: string;
  act_number: string | null;
  title: string;
  act_date?: string | null;
  visibility_scope?: string;
}

export interface ActVersionListResponse {
  versions: ActVersionReviewItem[];
  total: number;
  acts: ActSummary[];
  next_offset?: number | null;
}

export interface ActVersionUpdateRequest {
  effective_from?: string | null;
  effective_to?: string | null;
  act_id?: number | null;
}
