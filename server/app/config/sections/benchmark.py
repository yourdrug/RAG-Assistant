"""Benchmark runner and judge settings."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt

from config.ownership import Static


class BenchmarkSection(BaseModel):
    benchmark_max_concurrent: Static[PositiveInt] = 10
    benchmark_judge_max_concurrent: Static[PositiveInt] = 3
    benchmark_judge_grouped_enabled: Static[bool] = True
    benchmark_judge_initial_tokens: Static[PositiveInt] = 2048
    benchmark_judge_retry_tokens: Static[PositiveInt] = 8192
    benchmark_judge_model: Static[str] = ""  # empty = use fast_llm for OpenRouter
    # Recorded LLM responses hold private document text; unused ones are deleted after this.
    benchmark_llm_replay_retention_days: Static[PositiveInt] = 90
    # Part of every recorded request identity: change it to stop reusing all earlier recordings,
    # e.g. after a provider silently updated a model under the same name.
    benchmark_llm_replay_epoch: Static[str] = "1"
