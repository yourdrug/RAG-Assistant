"""Rolling summary updater adapter — wraps infrastructure RAG functions."""

from __future__ import annotations

from application.ports.ml_clients import MLClientPort
from infrastructure.ml.rag.rag_prompts import update_rolling_summary


class RollingSummaryUpdater:
    """Adapts the infrastructure rolling summary function behind the port."""

    def __init__(self, ml_clients: MLClientPort) -> None:
        self._ml_clients = ml_clients

    async def update(self, existing_summary: str | None, recent_turns: list[dict]) -> str:
        return await update_rolling_summary(self._ml_clients.llm(), existing_summary, recent_turns)
