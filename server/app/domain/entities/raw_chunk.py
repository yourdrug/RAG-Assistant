"""RawChunk — промежуточная сущность для передачи данных между парсером и pipeline.

Единый тип для обоих путей: API upload и CLI ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawChunk:
    """Распарсенный и разбитый чанк до обогащения метаданными."""

    page_content: str
    metadata: dict = field(default_factory=dict)
