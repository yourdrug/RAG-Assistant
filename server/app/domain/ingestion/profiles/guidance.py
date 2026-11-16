"""Shared structure for unversioned FAQs, instructions and procedures."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from domain.ingestion.chunking_policy import profile_policy
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.types import BoundaryLevel, ConfigDefault


if TYPE_CHECKING:
    from domain.ingestion.chunking_policy import ChunkingPolicy
    from domain.ingestion.ports import DomainSettingsPort


QUESTION_RE = re.compile(r"^[ \t]*(?:Вопрос[ \t]*:[ \t]*|Q:[ \t]*)([^\n]+)", re.MULTILINE | re.IGNORECASE)
HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+([^\n]+)", re.MULTILINE)
STEP_RE = re.compile(r"^[ \t]*(?:Шаг[ \t]+)(\d+)[.): \t]+", re.MULTILINE | re.IGNORECASE)


class GuidanceDomainProfile(GeneralDomainProfile):
    """Keep explicit headings and steps; leave unmarked prose to the neutral splitter."""

    title_pattern: re.Pattern[str] = re.compile(r"(?!)")

    def __init__(self, _settings: DomainSettingsPort | None = None) -> None:
        pass

    def structural_fingerprint(self, text: str) -> bool:
        # A mention inside a paragraph does not identify a document's type.
        title = text.lstrip().split("\n", 1)[0].strip().lstrip("# ")
        return bool(self.title_pattern.match(title))

    def classify_score(self, text: str) -> float:
        return 3.0 if self.structural_fingerprint(text) else 0.0

    def config_defaults(self) -> list[ConfigDefault]:
        return [
            ConfigDefault(f"{self.key}_chunk_target_tokens", "350", "int", "Target chunk tokens", 50, 8192),
            ConfigDefault(f"{self.key}_chunk_max_tokens", "512", "int", "Maximum chunk tokens", 50, 8192),
            ConfigDefault("classification_threshold", "1.5", "float", "Classification threshold", 0.1, 20),
        ]

    def chunking_policy(self, settings: DomainSettingsPort | None = None) -> ChunkingPolicy:
        return profile_policy(self.key, self.config_defaults(), settings)

    def content_boundaries(self) -> list[BoundaryLevel]:
        return [
            BoundaryLevel("heading_block", HEADING_RE, always_split=True, packing_barrier=True),
            BoundaryLevel("step", STEP_RE),
        ]

    def unit_label(self, kind: str, value: str) -> str:
        return f"Шаг {value}" if kind == "step" else value
