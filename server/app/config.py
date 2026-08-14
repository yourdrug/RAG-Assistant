import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULTS = {
    "jwt_secret_key": "change-me-in-production",
    "db_password": "ragpassword",
    "qdrant_api_key": "qdrant_api_key",
    "s3_access_key": "minioadmin",
    "s3_secret_key": "minioadmin",
    "admin_password": "admin",
}


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

    # --- Эмбеддинги и реранкер (лицензионно безопасный набор, MIT/Apache-2.0) ---
    embed_model: str = "BAAI/bge-m3"
    device: str = "cpu"  # "cuda" если есть GPU (fallback для всех)
    embed_device: str = ""  # отдельно для embedding; пусто = использовать device
    rerank_device: str = ""  # отдельно для reranker; пусто = использовать device
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # --- LLM ---
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_num_predict_narrow: int = 400
    llm_num_predict_broad: int = 2048
    llm_num_ctx_narrow: int = 8192
    llm_num_ctx_broad: int = 16384

    # --- OCR (для сканов внутри PDF) ---
    ocr_engine: str = "paddleocr"
    ocr_enabled: bool = True
    ocr_lang_paddle: str = "ru"  # ru | en | ...
    ocr_lang_surya: list = ["ru", "en"]
    ocr_dpi: int = 300

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

    @property
    def resolved_device(self) -> str:
        """Return 'cuda' only if DEVICE=cuda AND GPU is available."""
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def embed_resolved_device(self) -> str:
        """Return device for embedding model. Uses EMBED_DEVICE if set, else falls back to resolved_device."""
        if self.embed_device:
            if self.embed_device == "cuda" and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.resolved_device

    @property
    def rerank_resolved_device(self) -> str:
        """Return device for reranker model. Uses RERANK_DEVICE if set, else falls back to resolved_device."""
        if self.rerank_device:
            if self.rerank_device == "cuda" and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.resolved_device

    # --- App version & metadata ---
    version: str = ""
    service_start_datetime: str = ""

    stage: str = "development"  # development | prod

    # --- Background jobs cleanup ---
    job_cleanup_days: int = 30

    # Поддерживаемые расширения файлов
    supported_extensions: tuple = (".pdf", ".docx", ".doc", ".rtf", ".md", ".txt")

    # Upload limits
    max_upload_size_mb: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _check_security_defaults(self) -> "Settings":
        # Read version from VERSION file if not set via env
        if not self.version:
            self.version = _read_version()

        errors: list[str] = []
        is_prod = self.stage == "prod"

        if is_prod and self.jwt_secret_key == _INSECURE_DEFAULTS["jwt_secret_key"]:
            errors.append(
                "JWT_SECRET_KEY must be changed in production "
                "(currently 'change-me-in-production'). "
                "Generate with: openssl rand -hex 32"
            )

        if is_prod:
            for field, default in _INSECURE_DEFAULTS.items():
                if field == "jwt_secret_key":
                    continue
                val = getattr(self, field, None)
                if val == default:
                    errors.append(
                        f"{field} uses the insecure default value '{default}' "
                        f"which is not allowed in production (stage=prod). "
                        f"Set a strong value in server/.env"
                    )

        if errors:
            msg = "Security validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            if is_prod:
                print(msg, file=sys.stderr)
                sys.exit(1)
            else:
                logging.getLogger("default").warning(msg)

        if is_prod and "*" in self.allowed_origins_list:
            msg = (
                "CORS allowed_origins contains '*' which is insecure in production. "
                "Set ALLOWED_ORIGINS to specific origins in server/.env"
            )
            print(msg, file=sys.stderr)
            sys.exit(1)

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
