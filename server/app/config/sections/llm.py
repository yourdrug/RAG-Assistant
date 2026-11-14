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
    llm_provider: Literal["openrouter", "openai_compatible"] = LLMProvider.OPENROUTER.value
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

    # ── OpenAI-compatible (vLLM, Ollama, LM Studio, TGI, ...) ───────────────
    openai_compatible_api_key: Secret = SecretStr("")
    openai_compatible_base_url: str = ""
    openai_compatible_model: str = ""
    # Empty: auxiliary calls (condense, relevance, decomposition) reuse the main model.
    openai_compatible_auxiliary_model: str = ""

    # ── Circuit breaker ────────────────────────────────────────────────────
    llm_breaker_fail_max: Static[PositiveInt] = 5
    llm_breaker_timeout_duration: Static[PositiveInt] = 300  # 5 min OPEN state

    @property
    def active_llm_base_url(self) -> str:
        if self.llm_provider == LLMProvider.OPENAI_COMPATIBLE.value:
            return self.openai_compatible_base_url

        return self.openrouter_base_url

    @property
    def active_llm_api_key(self) -> SecretStr:
        if self.llm_provider == LLMProvider.OPENAI_COMPATIBLE.value:
            return self.openai_compatible_api_key

        return self.openrouter_api_key

    @property
    def active_llm_model(self) -> str:
        if self.llm_provider == LLMProvider.OPENAI_COMPATIBLE.value:
            return self.openai_compatible_model

        return self.openrouter_model

    @property
    def active_auxiliary_model(self) -> str:
        """Model for structured auxiliary calls (relevance, sufficiency, decomposition)."""
        if self.llm_provider == LLMProvider.OPENAI_COMPATIBLE.value:
            return self.openai_compatible_auxiliary_model or self.openai_compatible_model

        return self.openrouter_model

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
