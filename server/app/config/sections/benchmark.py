"""Benchmark runner and judge settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.ownership import Static


class BenchmarkSection(BaseModel):
    benchmark_max_concurrent: Static[int] = Field(default=10, ge=1)
    benchmark_judge_max_concurrent: Static[int] = Field(default=3, ge=1)
    benchmark_judge_grouped_enabled: Static[bool] = True
    benchmark_judge_initial_tokens: Static[int] = Field(default=2048, ge=1)
    benchmark_judge_retry_tokens: Static[int] = Field(default=8192, ge=1)
    benchmark_judge_model: Static[str] = ""  # empty = use fast_llm for OpenRouter
