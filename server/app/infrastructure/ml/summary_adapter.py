"""Rolling summary updater adapter — wraps infrastructure RAG functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from infrastructure.ml.rag import update_rolling_summary

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry


class RollingSummaryUpdater:
    """Adapts the infrastructure rolling summary function behind the port."""

    def __init__(self, ml_clients: MLClientRegistry) -> None:
        self._ml_clients = ml_clients

    async def update(self, existing_summary: str | None, recent_turns: list[dict]) -> str:
        return await update_rolling_summary(self._ml_clients.llm(), existing_summary, recent_turns)
