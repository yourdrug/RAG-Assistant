"""Rolling summary updater adapter — wraps infrastructure RAG functions."""

from __future__ import annotations

from infrastructure.clients import get_llm
from infrastructure.ml.rag import update_rolling_summary


class RollingSummaryUpdater:
    """Adapts the infrastructure rolling summary function behind the port."""

    async def update(self, existing_summary: str | None, recent_turns: list[dict]) -> str:
        return await update_rolling_summary(get_llm(), existing_summary, recent_turns)
