"""Benchmark Settings Protocol — abstracts benchmark configuration."""

from __future__ import annotations

from typing import Protocol


class BenchmarkSettingsProtocol(Protocol):
    @property
    def data_dir(self) -> str: ...

    @property
    def retriever_top_k(self) -> int: ...

    @property
    def llm_model(self) -> str: ...
