import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Ключ авторизации Qdrant. ОБЯЗАТЕЛЕН вне чисто localhost-разработки
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    database_url: str = os.getenv("DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/ragdb")
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
    ocr_dpi: int = 300  # разрешение рендера страницы перед OCR

    # --- Файловое хранилище ---
    file_backend: str = os.getenv("FILE_BACKEND", "local")  # "local" | "s3"

    # S3 / MinIO
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "http://minio:9000")
    s3_bucket: str = os.getenv("S3_BUCKET", "rag-documents")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")

    # RAG параметры
    retriever_fetch_k: int = 25  # сколько кандидатов достаём из Qdrant перед реранком
    retriever_top_k: int = 6  # сколько чанков остаётся после реранка и уходит в промпт
    history_window: int = 8
    chunk_size: int = 512
    chunk_overlap: int = 128

    # --- Авторизация ---
    # ОБЯЗАТЕЛЬНО смени в проде — например: openssl rand -hex 32
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 часа

    admin_email: str | None = os.getenv("ADMIN_EMAIL")
    admin_password: str | None = os.getenv("ADMIN_PASSWORD")

    # Поддерживаемые расширения файлов
    supported_extensions: tuple = (".pdf", ".docx", ".doc", ".rtf", ".md", ".txt")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
