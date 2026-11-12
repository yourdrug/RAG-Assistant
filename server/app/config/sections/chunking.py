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

    def check_chunking(self) -> list[str]:
        # Each bound is checked per field; the splitter also needs overlap < size.
        pairs = (
            ("chunk_overlap", "chunk_size"),
            ("legal_chunk_overlap", "legal_chunk_size"),
            ("decree_chunk_overlap", "decree_chunk_size"),
        )
        return [
            f"{overlap.upper()} must be less than {size.upper()}"
            for overlap, size in pairs
            if getattr(self, overlap) >= getattr(self, size)
        ]
