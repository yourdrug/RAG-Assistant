"""Document chunking settings (generic and per domain profile)."""

from __future__ import annotations

from pydantic import BaseModel

from application.runtime.dynamic_params import PROFILE_CONFIG_DEFAULTS


class ChunkingSection(BaseModel):
    chunk_size: int = 550
    chunk_overlap: int = 200
    legal_chunk_size: int = int(PROFILE_CONFIG_DEFAULTS[("legal_chunk_size", "legal")].value)
    legal_chunk_overlap: int = int(PROFILE_CONFIG_DEFAULTS[("legal_chunk_overlap", "legal")].value)
    decree_chunk_size: int = int(PROFILE_CONFIG_DEFAULTS[("decree_chunk_size", "decree")].value)
    decree_chunk_overlap: int = int(PROFILE_CONFIG_DEFAULTS[("decree_chunk_overlap", "decree")].value)
