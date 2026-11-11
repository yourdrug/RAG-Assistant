"""Object storage, uploads, previews and data paths."""

from __future__ import annotations

from pydantic import BaseModel, Field

from application.runtime.dynamic_params import MAX_UPLOAD_FILE_MB
from config.ownership import Secret, Static


class StorageSection(BaseModel):
    s3_endpoint: Static[str] = "http://minio:9000"
    s3_bucket: Static[str] = "rag-documents"
    s3_access_key: Secret[str] = ""
    s3_secret_key: Secret[str] = ""
    s3_region: Static[str] = "us-east-1"
    s3_max_concurrent: Static[int] = Field(default=10, ge=1)
    supported_extensions: Static[tuple[str, ...]] = (".pdf", ".docx", ".rtf", ".md", ".txt")
    max_upload_size_mb: int = Field(default=50, ge=1, le=MAX_UPLOAD_FILE_MB)
    preview_max_size_mb: int = Field(default=50, ge=1, le=MAX_UPLOAD_FILE_MB)
    preview_cache_ttl_seconds: int = Field(default=1800, ge=1)

    # ── Data paths ──────────────────────────────────────────────────────────
    data_dir: Static[str] = "/code/project/data"
    uploads_prefix: Static[str] = "uploads/"

    def check_storage_security(self) -> list[str]:
        if self.s3_access_key in ("", "minioadmin") or self.s3_secret_key in ("", "minioadmin"):
            return [
                "S3_ACCESS_KEY / S3_SECRET_KEY must be strong non-default values (set in server/.env.secrets)"
            ]

        return []
