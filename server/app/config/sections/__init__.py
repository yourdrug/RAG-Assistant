"""Settings sections: each field of ``Settings`` belongs to exactly one section."""

from __future__ import annotations

from config.sections.app import AppSection
from config.sections.auth import AuthSection
from config.sections.benchmark import BenchmarkSection
from config.sections.chat import ChatSection
from config.sections.chunking import ChunkingSection
from config.sections.concurrency import ConcurrencySection
from config.sections.database import DatabaseSection
from config.sections.http import HttpSection
from config.sections.jobs import JobsSection
from config.sections.llm import LLMSection
from config.sections.ml import MLProviderSection
from config.sections.ocr import OCRSection
from config.sections.rag import HybridSearchSection, RagFeaturesSection, RankingSection, RetrievalSection
from config.sections.rate_limit import CostLimitSection, RateLimitSection
from config.sections.redis import RedisSection
from config.sections.storage import StorageSection
from config.sections.vector_store import VectorStoreSection


ALL_SECTIONS = (
    AppSection,
    DatabaseSection,
    VectorStoreSection,
    LLMSection,
    MLProviderSection,
    ConcurrencySection,
    OCRSection,
    RetrievalSection,
    RankingSection,
    HybridSearchSection,
    RagFeaturesSection,
    ChunkingSection,
    ChatSection,
    RedisSection,
    JobsSection,
    StorageSection,
    HttpSection,
    AuthSection,
    RateLimitSection,
    CostLimitSection,
    BenchmarkSection,
)

__all__ = [
    "ALL_SECTIONS",
    "AppSection",
    "AuthSection",
    "BenchmarkSection",
    "ChatSection",
    "ChunkingSection",
    "ConcurrencySection",
    "CostLimitSection",
    "DatabaseSection",
    "HttpSection",
    "HybridSearchSection",
    "JobsSection",
    "LLMSection",
    "MLProviderSection",
    "OCRSection",
    "RagFeaturesSection",
    "RankingSection",
    "RateLimitSection",
    "RedisSection",
    "RetrievalSection",
    "StorageSection",
    "VectorStoreSection",
]
