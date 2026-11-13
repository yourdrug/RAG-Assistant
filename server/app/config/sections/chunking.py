"""Document chunking settings (generic and per domain profile)."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Static


class ChunkingSection(BaseModel):
    # Character limits for manual chunk edit validation. Ingestion uses tokens below.
    chunk_size: int = 550
    chunk_overlap: int = 200
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    embed_tokenizer_path: Static[str] = "data/tei_cache/bge-m3/tokenizer.json"
    # Plan 4.3: LLM situating notes added only to the dense embedding input.
    chunk_contextualization_enabled: Static[bool] = False
    chunk_contextualization_concurrency: Static[int] = 4
    chunk_contextualization_timeout: Static[int] = 30

    def check_chunking(self) -> list[str]:
        # Each bound is checked per field; the splitter also needs overlap < size.
        pairs = (("chunk_overlap", "chunk_size"),)
        errors = [
            f"{overlap.upper()} must be less than {size.upper()}"
            for overlap, size in pairs
            if getattr(self, overlap) >= getattr(self, size)
        ]

        if self.chunk_target_tokens <= 0 or self.chunk_max_tokens < self.chunk_target_tokens:
            errors.append("CHUNK_TARGET_TOKENS must be positive and <= max tokens")

        if self.chunk_contextualization_concurrency <= 0 or self.chunk_contextualization_timeout <= 0:
            errors.append("CHUNK_CONTEXTUALIZATION_CONCURRENCY and _TIMEOUT must be positive")

        if not 0 <= self.chunk_overlap_tokens < self.chunk_target_tokens:
            errors.append("CHUNK_OVERLAP_TOKENS must be non-negative and less than target tokens")

        return errors
