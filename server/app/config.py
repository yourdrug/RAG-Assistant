import os
from datetime import UTC, datetime

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Ключ авторизации Qdrant. ОБЯЗАТЕЛЕН вне чисто localhost-разработки
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    database_url: str = os.getenv("DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/ragdb")

    # --- Individual DB params (used by async DatabaseManager) ---
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_user: str = os.getenv("DB_USER", "raguser")
    db_password: str = os.getenv("DB_PASSWORD", "ragpassword")
    db_name: str = os.getenv("DB_NAME", "ragdb")

    # --- Cluster slave nodes (comma-separated host:port pairs, e.g. "slave1:5433,slave2:5434") ---
    db_slave_hosts: str = os.getenv("DB_SLAVE_HOSTS", "")
    db_slave_ports: str = os.getenv("DB_SLAVE_PORTS", "")
    collection_name: str = os.getenv("COLLECTION_NAME", "company_docs")

    data_dir: str = os.getenv("DATA_DIR", "/code/project/data")

    uploads_prefix: str = os.getenv("UPLOADS_PREFIX", "uploads/")

    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # --- Эмбеддинги и реранкер (лицензионно безопасный набор, MIT/Apache-2.0) ---
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")  # "cuda" если есть GPU

    # --- LLM ---
    # qwen2.5:14b — лучший баланс русского языка и качества среди безопасных по лицензии моделей.
    # mistral-nemo:12b — альтернатива (Apache-2.0), если нужна модель без ограничений Qwen License.
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:14b")
    llm_num_predict_narrow: int = int(os.getenv("LLM_NUM_PREDICT_NARROW", "180"))
    llm_num_predict_broad: int = int(os.getenv("LLM_NUM_PREDICT_BROAD", "900"))

    # --- OCR (для сканов внутри PDF) ---
    # "paddleocr" (по умолчанию) — Apache-2.0, без ограничений по выручке компании.
    # "surya"    — точнее на сложной вёрстке, НО веса модели лицензированы отдельно
    #              (бесплатно для research/личного использования и стартапов до $5M
    #              выручки/финансирования; коммерческое использование сверх этого — платно,
    #              см. README раздел "Лицензии"). Включай осознанно через OCR_ENGINE=surya|auto.
    # "auto"     — сначала PaddleOCR, и только если он не дал текста — Surya (если включён).
    ocr_engine: str = os.getenv("OCR_ENGINE", "paddleocr")
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    ocr_lang_paddle: str = os.getenv("OCR_LANG_PADDLE", "ru")  # ru | en | ...
    ocr_lang_surya: list = ["ru", "en"]
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))

    # --- Файловое хранилище ---
    file_backend: str = os.getenv("FILE_BACKEND", "local")  # "local" | "s3"

    # S3 / MinIO
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "http://minio:9000")
    s3_bucket: str = os.getenv("S3_BUCKET", "rag-documents")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")

    # RAG параметры — узкие вопросы
    retriever_fetch_k: int = 25  # сколько кандидатов достаём из Qdrant перед реранком
    retriever_top_k: int = 4  # сколько чанков остаётся после реранка и уходит в промпт

    # RAG параметры — широкие вопросы (подробные, обзорные)
    retriever_fetch_k_broad: int = 40
    retriever_top_k_broad: int = 10
    history_window: int = 8

    # --- Reranker score filters ---
    rerank_min_score: float | None = None  # абсолютный порог (logit); None = не фильтровать
    rerank_score_gap_ratio: float | None = None  # относительный разрыв (0..1); None = не фильтровать

    # --- Source display filter ---
    source_min_score: float = float(
        os.getenv("SOURCE_MIN_SCORE", "0.3")
    )  # мин. max_score источника для показа

    # --- Citation filter ---
    citation_filter_enabled: bool = os.getenv("CITATION_FILTER_ENABLED", "false").lower() == "true"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "550"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # --- Hybrid search (BM25 + dense RRF) ---
    hybrid_enabled: bool = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
    bm25_fetch_k: int = 25  # сколько кандидатов из BM25 перед RRF
    rrf_k: int = 30  # константа RRF
    dense_weight: float = 1.5
    sparse_weight: float = 0.5

    # --- Авторизация ---
    # ОБЯЗАТЕЛЬНО смени в проде — например: openssl rand -hex 32
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 часа

    admin_email: str | None = os.getenv("ADMIN_EMAIL")
    admin_password: str | None = os.getenv("ADMIN_PASSWORD")

    # --- Timezone (IANA tz name) ---
    timezone: str = os.getenv("TIMEZONE", "UTC")

    # --- App version & metadata ---
    version: str = os.getenv("VERSION", "0.2.0")
    service_start_datetime: str = os.getenv("SERVICE_START_DATETIME", "")

    # --- Background jobs cleanup ---
    job_cleanup_days: int = int(os.getenv("JOB_CLEANUP_DAYS", "30"))

    # Поддерживаемые расширения файлов
    supported_extensions: tuple = (".pdf", ".docx", ".doc", ".rtf", ".md", ".txt")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
