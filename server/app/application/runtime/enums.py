"""Runtime ML provider, storage, deployment and health constants."""

from __future__ import annotations

from enum import StrEnum


class RerankApiFormat(StrEnum):
    """Wire format of the reranking endpoint (there is no common reranking standard)."""

    # POST {base}/rerank {model, query, documents} -> results[{index, relevance_score}]
    # (vLLM, Jina, Cohere, Infinity).
    COHERE = "cohere"
    # POST {base}/rerank {query, texts, raw_scores} -> [{index, score}] (Text Embeddings Inference).
    TEI = "tei"
    # POST {base}/inference/{model} {queries, documents} -> {scores} (DeepInfra native).
    DEEPINFRA = "deepinfra"


class AppStage(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PROD = "prod"


class HealthStatus(StrEnum):
    OK = "ok"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
