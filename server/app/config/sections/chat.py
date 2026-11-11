"""Chat history and curator scope limits."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Static


class ChatSection(BaseModel):
    history_window: int = 8
    curator_scope_max_ids: Static[int] = 1000
