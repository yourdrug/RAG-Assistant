"""Application configuration — Pydantic BaseSettings with .env support.

Lifecycle:
  - Created at import time (module-level ``settings`` singleton)
  - Read-only after initialization (except dynamic config updates via event bus)
  - Process-scoped: one instance per process
  - Dynamic config: ``ConfigParameterChanged`` events update runtime values
    through settings adapters (``LiveChatSettings``, ``LiveChunkSettings``, etc.)
"""

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_version() -> str:
    """Read version from VERSION file in project root."""
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    try:
        return version_file.read_text().strip()
    except (FileNotFoundError, OSError):
        return "0.0.0"


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    database_url: str = "postgresql://raguser:ragpassword@localhost:5432/ragdb"

    # --- Individual DB params (used by async DatabaseManager) ---
    db_host: str = "localhost"
    db_port: str = "5432"
    db_user: str = "raguser"
    db_password: str = "ragpassword"
    db_name: str = "ragdb"

    # --- Cluster slave nodes (comma-separated host:port pairs, e.g. "slave1:5433,slave2:5434") ---
    db_slave_hosts: str = ""
    db_slave_ports: str = ""

    # --- Connection pool (per engine) ---
    db_pool_size: int = 100
    db_max_overflow: int = 20

    @property
    def db_slave_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.db_slave_hosts.split(",") if h.strip()]

    @property
    def db_slave_ports_list(self) -> list[str]:
        return [p.strip() for p in self.db_slave_ports.split(",") if p.strip()]

    collection_name: str = "company_docs"

    data_dir: str = "/code/project/data"

    uploads_prefix: str = "uploads/"

    allowed_origins: str = "*"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # --- TEI (Text Embeddings Inference) ---
    tei_embed_url: str = ""  # e.g. "http://tei-embed:8080"
    tei_rerank_url: str = ""  # e.g. "http://tei-rerank:8080"

    # --- LLM ---
    llm_provider: str = "ollama"  # "ollama" | "openrouter"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_num_predict_narrow: int = 400
    llm_num_predict_broad: int = 2048
    llm_num_ctx_narrow: int = 8192
    llm_num_ctx_broad: int = 16384

    # --- OpenRouter (cloud LLM API) ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen-2.5-7b-instruct"

    # --- OCR (для сканов внутри PDF) ---
    ocr_engine: str = "paddleocr"
    ocr_enabled: bool = True
    ocr_lang_paddle: str = "ru"  # ru | en | ...
    ocr_lang_surya: list = ["ru", "en"]
    ocr_dpi: int = 300
    ocr_min_chars: int = 40  # pages with fewer chars also get OCR

    # --- Redis (очередь задач, rate limiting, кэш API-ключей) — обязательный компонент ---
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_user: str = ""
    redis_password: str = ""
    redis_db: int = 0
    # Worker: максимальное количество одновременных задач
    worker_max_concurrent: int = 4

    @property
    def redis_url(self) -> str:
        """Build Redis URL from individual components."""
        auth = ""
        if self.redis_user:
            auth = self.redis_user
            if self.redis_password:
                auth = f"{auth}:{self.redis_password}"
            auth = f"{auth}@"
        elif self.redis_password:
            auth = f":{self.redis_password}@"
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # --- Файловое хранилище ---
    file_backend: str = "local"  # "local" | "s3"

    # S3 / MinIO
    s3_endpoint: str = "http://minio:9000"
    s3_bucket: str = "rag-documents"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"

    # RAG параметры — узкие вопросы
    retriever_fetch_k: int = 25
    retriever_top_k: int = 4

    # RAG параметры — широкие вопросы (подробные, обзорные)
    retriever_fetch_k_broad: int = 40
    retriever_top_k_broad: int = 10
    history_window: int = 8

    # --- Reranker score filters ---
    rerank_min_score: float | None = 0.15
    rerank_score_gap_ratio: float | None = 0.1

    # --- Source display filter ---
    source_min_score: float = 0.3

    # --- Citation filter ---
    citation_filter_enabled: bool = False
    chunk_size: int = 550
    chunk_overlap: int = 200
    embed_batch_size: int = 32

    # --- Relevance gate (Self-RAG-lite) ---
    relevance_gate_enabled: bool = False

    # --- Query condensation (rewrite follow-up questions) ---
    condense_enabled: bool = True

    # --- Query decomposition ---
    decomposition_enabled: bool = False

    # --- Rolling summary for long dialogs ---
    rolling_summary_enabled: bool = True

    # --- Semantic answer cache ---
    cache_enabled: bool = False

    # --- Hybrid search (BM25 + dense RRF) ---
    hybrid_enabled: bool = True
    bm25_fetch_k: int = 25
    rrf_k: int = 30
    dense_weight: float = 1.5
    sparse_weight: float = 0.5

    # --- Legal document chunking ---
    legal_chunk_size: int = 1000
    legal_chunk_overlap: int = 250

    # --- Exact reference sparse boost ---
    exact_ref_sparse_boost: float = 2.5

    # --- Document domain classifier ---
    document_domain_marker_threshold: float = 2.0

    # --- Авторизация ---
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 часа

    admin_email: str | None = None
    admin_password: str | None = None

    # --- Timezone (IANA tz name) ---
    timezone: str = "UTC"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    # --- App version & metadata ---
    version: str = ""
    service_start_datetime: str = ""

    # --- Background jobs cleanup ---
    job_cleanup_days: int = 30

    # Поддерживаемые расширения файлов
    supported_extensions: tuple = (".pdf", ".docx", ".doc", ".rtf", ".md", ".txt")

    # Upload limits
    max_upload_size_mb: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _init_version(self) -> "Settings":
        if not self.version:
            self.version = _read_version()
        return self

    @property
    def uptime_seconds(self) -> float:
        if not self.service_start_datetime:
            return 0.0
        try:
            start = datetime.fromisoformat(self.service_start_datetime)
            now = datetime.now(tz=UTC)
            return (now - start).total_seconds()
        except ValueError:
            return 0.0


settings = Settings()
