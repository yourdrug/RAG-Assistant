"""Embedding and reranking endpoints, embedding identity and HTTP pools."""

from __future__ import annotations

from pydantic import BaseModel, NonNegativeInt, PositiveInt, SecretStr

from application.runtime.enums import RerankApiFormat
from config.ownership import Secret, Static


class MLProviderSection(BaseModel):
    # ── Embeddings: OpenAI-compatible /embeddings (TEI /v1, DeepInfra /v1/openai, vLLM, ...) ──
    embedding_base_url: str = "https://api.deepinfra.com/v1/openai"
    # Model, revision and dimension form the corpus embedding identity: deployment-owned.
    embedding_model: Static[str] = "BAAI/bge-m3"
    embedding_revision: Static[str] = "v2"
    embedding_api_key: Secret = SecretStr("")
    embed_batch_size: int = 32
    embed_dim: Static[PositiveInt] = 1024

    # ── Reranking: endpoint wire format selected by rerank_api_format ──────
    rerank_base_url: str = "https://api.deepinfra.com/v1"
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    rerank_api_format: str = RerankApiFormat.DEEPINFRA.value
    rerank_api_key: Secret = SecretStr("")

    # ── HTTP pools ──────────────────────────────────────────────────────────
    embedding_http_max_connections: PositiveInt = 40
    embedding_http_max_keepalive_connections: NonNegativeInt = 20
    rerank_http_max_connections: PositiveInt = 20
    rerank_http_max_keepalive_connections: NonNegativeInt = 10
    # Deployment controls the trust boundary; database writers cannot expand it.
    # External embedding/rerank origins; their keys are forwarded only there.
    ml_allowed_origins: Static[list[str]] = ["https://api.deepinfra.com"]
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

    def check_ml_endpoints(self) -> list[str]:
        required = {
            "EMBEDDING_BASE_URL": self.embedding_base_url,
            "EMBEDDING_MODEL": self.embedding_model,
            "RERANK_BASE_URL": self.rerank_base_url,
        }
        errors = [f"{key} is required" for key, value in required.items() if not value]

        # Hot-reload candidates bypass field validation, so membership is checked explicitly.
        if self.rerank_api_format not in {item.value for item in RerankApiFormat}:
            errors.append(f"Unknown rerank_api_format {self.rerank_api_format!r}")

        return errors
