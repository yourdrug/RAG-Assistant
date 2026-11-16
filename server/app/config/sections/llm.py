"""OpenAI-compatible LLM endpoint, generation parameters and circuit breaker."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt, SecretStr

from config.ownership import Secret, Static
from domain.rag.settings import (
    DEFAULT_NUM_CTX_BROAD,
    DEFAULT_NUM_CTX_NARROW,
    DEFAULT_NUM_PREDICT_BROAD,
    DEFAULT_NUM_PREDICT_NARROW,
)


class LLMSection(BaseModel):
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

    # ── OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, LM Studio, ...) ──
    # The key is deployment-owned; the endpoint is editable only within
    # LLM_ALLOWED_ORIGINS / ML_INTERNAL_ALLOWED_ORIGINS so it cannot be redirected.
    llm_api_key: Secret = SecretStr("")
    llm_allowed_origins: Static[list[str]] = ["https://openrouter.ai"]
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen-2.5-7b-instruct"
    # Empty: auxiliary calls (condense, relevance, decomposition) reuse the main model.
    llm_auxiliary_model: str = ""

    # ── Circuit breaker ────────────────────────────────────────────────────
    llm_breaker_fail_max: Static[PositiveInt] = 5
    llm_breaker_timeout_duration: Static[PositiveInt] = 300  # 5 min OPEN state

    @property
    def active_auxiliary_model(self) -> str:
        """Model for structured auxiliary calls (relevance, sufficiency, decomposition)."""
        return self.llm_auxiliary_model or self.llm_model

    def check_llm_endpoint(self) -> list[str]:
        required = {"LLM_BASE_URL": self.llm_base_url, "LLM_MODEL": self.llm_model}
        return [f"{key} is required" for key, value in required.items() if not value]

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
