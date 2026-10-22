"""Live settings and context-scoped overrides for parameter sweeps."""

from __future__ import annotations

from contextlib import contextmanager

from config import _settings_overrides, get_setting, settings
from infrastructure.benchmark.checkpoint import fingerprint


class LiveSweepSettings:
    def snapshot(self) -> dict:
        return settings_snapshot()

    @property
    def fetch_k(self) -> int:
        return int(get_setting("retriever_fetch_k"))

    @property
    def top_k(self) -> int:
        return int(get_setting("retriever_top_k"))

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


def settings_snapshot() -> dict:
    return {
        name: get_setting(name)
        for name in type(settings).model_fields
        if not any(part in name for part in ("key", "password", "secret", "url", "dir"))
        and name
        not in {
            "benchmark_judge_initial_tokens",
            "benchmark_judge_retry_tokens",
            "benchmark_judge_grouped_enabled",
            "benchmark_judge_max_concurrent",
        }
    }


def settings_fingerprint() -> str:
    """Identify effective settings without persisting credentials or operational judge knobs."""
    return fingerprint(settings_snapshot())
