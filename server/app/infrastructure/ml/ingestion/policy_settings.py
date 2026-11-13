"""Read profile policy inputs while preserving global general-domain controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from config import ChunkingSection
    from domain.ingestion.ports import DomainSettingsPort


@dataclass(frozen=True)
class PolicySettings:
    domain_settings: DomainSettingsPort | None
    general_settings: ChunkingSection

    def get(self, key: str, domain_key: str) -> str:
        if self.domain_settings is not None:
            try:
                return self.domain_settings.get(key, domain_key=domain_key)
            except KeyError:
                if domain_key != "general":
                    raise

        if domain_key == "general":
            suffix = key.removeprefix("general_")

            if suffix in ("chunk_target_tokens", "chunk_max_tokens", "chunk_overlap_tokens"):
                return str(getattr(self.general_settings, suffix))

        message = f"No policy setting {key!r} for domain {domain_key!r}"
        raise KeyError(message)
