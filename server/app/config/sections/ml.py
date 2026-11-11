"""Embedding / reranking provider, embedding identity and HTTP pools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from application.runtime.enums import MLProvider
from config.ownership import Secret, Static


class MLProviderSection(BaseModel):
    ml_provider: Static[Literal["tei", "deepinfra"]] = MLProvider.DEEPINFRA.value
    tei_embed_url: str = "http://tei-embed:8080"
    tei_embed_model: Static[str] = "BAAI/bge-m3"
    embedding_revision: Static[str] = "v1"
    tei_rerank_url: str = "http://tei-rerank:8080"
    deepinfra_api_key: Secret[str] = ""
    deepinfra_base_url: str = "https://api.deepinfra.com/v1"
    deepinfra_embed_model: str = "BAAI/bge-m3"
    deepinfra_rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    embed_batch_size: int = 32
    embed_dim: Static[int] = 1024

    # ── HTTP pools ──────────────────────────────────────────────────────────
    embedding_http_max_connections: int = Field(default=40, gt=0)
    embedding_http_max_keepalive_connections: int = Field(default=20, ge=0)
    rerank_http_max_connections: int = Field(default=20, gt=0)
    rerank_http_max_keepalive_connections: int = Field(default=10, ge=0)
    # Deployment controls the trust boundary; database writers cannot expand it.
    ml_internal_allowed_origins: Static[list[str]] = [
        "http://tei-embed:8080",
        "http://tei-rerank:8080",
        "http://qdrant:6333",
        "http://localhost:6333",
    ]

    def check_http_pools(self) -> list[str]:
        errors = []

        for prefix in ("embedding", "rerank"):
            maximum = getattr(self, f"{prefix}_http_max_connections")
            keepalive = getattr(self, f"{prefix}_http_max_keepalive_connections")

            if maximum <= 0 or keepalive < 0 or keepalive > maximum:
                errors.append(f"Invalid {prefix} HTTP pool: 0 <= keepalive <= positive connections required")

        return errors

    def check_ml_provider(self) -> list[str]:
        if self.ml_provider == MLProvider.TEI.value:
            missing = []

            if not self.tei_embed_url:
                missing.append("TEI_EMBED_URL")

            if not self.tei_rerank_url:
                missing.append("TEI_RERANK_URL")

            if missing:
                return [f"{', '.join(missing)} required when ML_PROVIDER=tei"]
        elif self.ml_provider == MLProvider.DEEPINFRA.value:
            if not self.deepinfra_api_key:
                return ["DEEPINFRA_API_KEY required when ML_PROVIDER=deepinfra"]

        return []
