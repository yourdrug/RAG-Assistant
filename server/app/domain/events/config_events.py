"""Domain Events — конфигурационные события."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class ConfigParameterChanged:
    """Параметр динамической конфигурации изменён.

    Публикуется ПОСЛЕ успешного commit в БД. Подписчики сами решают,
    релевантно ли для них изменение конкретного key — событие общее,
    фильтрация происходит внутри каждого хендлера.
    """

    key: str
    old_value: str | None
    new_value: str
    value_type: str
    changed_by: int | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
