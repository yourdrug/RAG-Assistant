"""Rolling summary updater adapter — wraps infrastructure RAG functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from infrastructure.ml.rag.rag_prompts import update_rolling_summary


if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable

    from application.ports.ml_clients import MLClientPort


class RollingSummaryUpdater:
    """Adapts the infrastructure rolling summary function behind the port."""

    def __init__(self, ml_clients: MLClientPort) -> None:
        self._ml_clients = ml_clients

    async def update(self, existing_summary: str | None, recent_turns: list[dict[str, Any]]) -> str:
        # The infrastructure summary pipeline composes a LangChain runnable;
        # the application port intentionally exposes only invocation methods.
        llm = cast("Runnable[Any, BaseMessage]", self._ml_clients.llm())
        return await update_rolling_summary(llm, existing_summary, recent_turns)
