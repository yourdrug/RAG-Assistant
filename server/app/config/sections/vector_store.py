"""Qdrant endpoint and collection settings."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt, SecretStr

from config.ownership import Secret, Static


class VectorStoreSection(BaseModel):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: Secret = SecretStr("")
    collection_name: str = "company_docs"
    qdrant_timeout: Static[PositiveInt] = 10
