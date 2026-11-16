"""Russian keywords of Belarusian official document types."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from domain.ingestion.doc_type import DocTypeKeywords


DOC_TYPE_KEYWORDS: DocTypeKeywords = (
    ("order", ("приказ", "распоряжение", "постановление")),
    ("contract", ("договор", "контракт", "соглашение")),
    ("protocol", ("протокол",)),
    ("instruction", ("инструкция", "регламент", "порядок действий")),
    ("policy", ("положение о",)),
    ("letter", ("письмо", "уведомление")),
    ("report", ("отчёт", "отчет")),
)
