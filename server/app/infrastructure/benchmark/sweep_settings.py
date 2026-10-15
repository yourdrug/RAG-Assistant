"""Live settings and context-scoped overrides for parameter sweeps."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

from config import _settings_overrides, get_setting


class SweepSettingsPort(Protocol):
    @property
    def fetch_k(self) -> int: ...

    @property
    def top_k(self) -> int: ...

    @property
    def results_path(self) -> str | None: ...

    def override(self, config: dict) -> AbstractContextManager[None]: ...


class LiveSweepSettings:
    @property
    def fetch_k(self) -> int:
        return int(get_setting("retriever_fetch_k"))

    @property
    def top_k(self) -> int:
        return int(get_setting("retriever_top_k"))

    @property
    def results_path(self) -> str | None:
        return None

    @contextmanager
    def override(self, config: dict):
        mapping = {
            "top_k": "retriever_top_k",
            "fetch_k": "retriever_fetch_k",
            "dense_weight": "dense_weight",
            "sparse_weight": "sparse_weight",
            "rrf_k": "rrf_k",
            "rerank_min_score": "rerank_min_score",
            "rerank_score_gap_ratio": "rerank_score_gap_ratio",
        }
        overrides = {
            **(_settings_overrides.get() or {}),
            **{mapping[key]: value for key, value in config.items() if key in mapping},
        }
        token = _settings_overrides.set(overrides)
        try:
            yield
        finally:
            _settings_overrides.reset(token)
