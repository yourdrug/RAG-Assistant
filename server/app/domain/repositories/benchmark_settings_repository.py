"""Benchmark settings protocol -- abstracts benchmark configuration for the domain layer.

Provides read-only access to RAG-related settings (data_dir, top_k, model
names) needed by the benchmark runner without depending on the concrete
``config.settings`` object.
"""

from __future__ import annotations

from typing import Protocol


class BenchmarkSettingsProtocol(Protocol):
    @property
    def data_dir(self) -> str: ...

    @property
    def retriever_top_k(self) -> int: ...

    @property
    def llm_model(self) -> str: ...
