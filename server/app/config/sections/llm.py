"""LLM provider, generation parameters and circuit breaker."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, PositiveInt, SecretStr

from application.runtime.enums import LLMProvider
from config.ownership import Secret, Static
from domain.rag.settings import (
    DEFAULT_NUM_CTX_BROAD,
    DEFAULT_NUM_CTX_NARROW,
    DEFAULT_NUM_PREDICT_BROAD,
    DEFAULT_NUM_PREDICT_NARROW,
)


class LLMSection(BaseModel):
    llm_provider: Literal["openrouter"] = LLMProvider.OPENROUTER.value
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_num_ctx_narrow: int = DEFAULT_NUM_CTX_NARROW
    llm_num_ctx_broad: int = DEFAULT_NUM_CTX_BROAD
    llm_num_predict_narrow: int = DEFAULT_NUM_PREDICT_NARROW
    llm_num_predict_broad: int = DEFAULT_NUM_PREDICT_BROAD
    llm_request_timeout: Static[PositiveInt] = 120
    llm_auxiliary_timeout: Static[PositiveInt] = (
        30  # seconds, caps semaphore hold time for auxiliary LLM calls
    )

    # ── OpenRouter ──────────────────────────────────────────────────────────
    openrouter_api_key: Secret = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen-2.5-7b-instruct"

    # ── Circuit breaker ────────────────────────────────────────────────────
    llm_breaker_fail_max: Static[PositiveInt] = 5
    llm_breaker_timeout_duration: Static[PositiveInt] = 300  # 5 min OPEN state

    def check_generation_budget(self) -> list[str]:
        # The prompt budget is num_ctx - num_predict; without room the answer cannot be generated.
        return [
            f"{predict} must be less than {ctx} ({getattr(self, predict)} >= {getattr(self, ctx)})"
            for predict, ctx in (
                ("llm_num_predict_narrow", "llm_num_ctx_narrow"),
                ("llm_num_predict_broad", "llm_num_ctx_broad"),
            )
            if getattr(self, predict) >= getattr(self, ctx)
        ]
