"""Application configuration — Pydantic BaseSettings with .env support.

All values come from server/.env. No hardcoded defaults — missing env vars
cause an immediate startup error, preventing silent misconfiguration.
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_version() -> str:
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    try:
        return version_file.read_text().strip()
    except (FileNotFoundError, OSError):
        return "0.0.0"


class Settings(BaseSettings):
    # --- PostgreSQL ---
    db_host: str
    db_port: str
    db_user: str
    db_password: str
    db_name: str

    # --- Cluster slave nodes ---
    db_slave_hosts: str = ""
    db_slave_ports: str = ""

    # --- Connection pool ---
    db_pool_size: int = 100
    db_max_overflow: int = 20

    @property
    def db_slave_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.db_slave_hosts.split(",") if h.strip()]

    @property
    def db_slave_ports_list(self) -> list[str]:
        return [p.strip() for p in self.db_slave_ports.split(",") if p.strip()]

    # --- Qdrant ---
    qdrant_url: str
    qdrant_api_key: str = ""
    collection_name: str

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"

    # --- Data ---
    data_dir: str
    uploads_prefix: str = "uploads/"

    # --- CORS ---
    allowed_origins: str

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # --- TEI ---
    tei_embed_url: str = ""
    tei_rerank_url: str = ""

    # --- ML Provider (tei | deepinfra) ---
    ml_provider: str

    # --- DeepInfra ---
    deepinfra_api_key: str = ""
    deepinfra_base_url: str = "https://api.deepinfra.com/v1"
    deepinfra_embed_model: str = "BAAI/bge-m3"
    deepinfra_rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # --- LLM ---
    llm_provider: str
    llm_model: str
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_num_predict_narrow: int = 400
    llm_num_predict_broad: int = 2048
    llm_num_ctx_narrow: int = 8192
    llm_num_ctx_broad: int = 16384

    # --- OpenRouter ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen-2.5-7b-instruct"

    # --- OCR ---
    ocr_engine: str = "paddleocr"
    ocr_enabled: bool = True
    ocr_lang_paddle: str = "ru"
    ocr_lang_surya: list = ["ru", "en"]
    ocr_dpi: int = 300
    ocr_min_chars: int = 40

    # --- Redis ---
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_user: str = ""
    redis_password: str = ""
    redis_db: int = 0
    worker_max_concurrent: int = 4

    @property
    def redis_url(self) -> str:
        auth = ""
        if self.redis_user:
            auth = self.redis_user
            if self.redis_password:
                auth = f"{auth}:{self.redis_password}"
            auth = f"{auth}@"
        elif self.redis_password:
            auth = f":{self.redis_password}@"
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # --- Storage ---
    file_backend: str = "s3"
    s3_endpoint: str = "http://minio:9000"
    s3_bucket: str = "rag-documents"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"

    # --- RAG ---
    retriever_fetch_k: int = 25
    retriever_top_k: int = 4
    retriever_fetch_k_broad: int = 40
    retriever_top_k_broad: int = 10
    history_window: int = 8
    rerank_min_score: float | None = 0.15
    rerank_score_gap_ratio: float | None = 0.1
    source_min_score: float = 0.3
    citation_filter_enabled: bool = False
    chunk_size: int = 550
    chunk_overlap: int = 200
    embed_batch_size: int = 32

    # --- Dynamic toggles ---
    relevance_gate_enabled: bool = False
    condense_enabled: bool = True
    decomposition_enabled: bool = False
    rolling_summary_enabled: bool = True
    cache_enabled: bool = False
    pii_redaction_enabled: bool = True
    hybrid_enabled: bool = True
    bm25_fetch_k: int = 25
    rrf_k: int = 30
    dense_weight: float = 1.5
    sparse_weight: float = 0.5

    # --- Legal chunking ---
    legal_chunk_size: int = 1000
    legal_chunk_overlap: int = 250

    # --- Misc ---
    exact_ref_sparse_boost: float = 2.5
    document_domain_marker_threshold: float = 2.0

    # --- Auth ---
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    admin_email: str = ""
    admin_password: str = ""

    # --- Timezone ---
    timezone: str = "UTC"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    # --- App metadata ---
    version: str = ""
    service_start_datetime: str = ""
    stage: str = "development"

    # --- Background jobs ---
    job_cleanup_days: int = 30

    # --- File types ---
    supported_extensions: tuple = (".pdf", ".docx", ".doc", ".rtf", ".md", ".txt")
    max_upload_size_mb: int = 50

    # --- Cost rate limiting ---
    cost_rate_limit_enabled: bool = False
    cost_hourly_limit: float = 1.0
    cost_daily_limit: float = 5.0

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        extra="ignore",
    )

    @model_validator(mode="after")
    def _post_init(self) -> "Settings":
        if not self.version:
            self.version = _read_version()

        is_prod = self.stage == "prod"
        errors = self._check_security(is_prod) + self._check_ml_provider()

        if errors:
            msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            if is_prod:
                sys.stderr.write(msg + "\n")
                sys.exit(1)
            else:
                logging.getLogger("default").warning(msg)

        return self

    def _check_security(self, is_prod: bool) -> list[str]:
        errors: list[str] = []
        if is_prod:
            if self.jwt_secret_key == "change-me-in-production":
                errors.append("JWT_SECRET_KEY must be changed in production")
            if self.file_backend == "local":
                errors.append("FILE_BACKEND must be 's3' in production")
        return errors

    def _check_ml_provider(self) -> list[str]:
        if self.ml_provider == "tei":
            missing = []
            if not self.tei_embed_url:
                missing.append("TEI_EMBED_URL")
            if not self.tei_rerank_url:
                missing.append("TEI_RERANK_URL")
            if missing:
                return [f"{', '.join(missing)} required when ML_PROVIDER=tei"]
        elif self.ml_provider == "deepinfra":
            if not self.deepinfra_api_key:
                return ["DEEPINFRA_API_KEY required when ML_PROVIDER=deepinfra"]
        else:
            return [f"ML_PROVIDER must be 'tei' or 'deepinfra', got '{self.ml_provider}'"]
        return []

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
