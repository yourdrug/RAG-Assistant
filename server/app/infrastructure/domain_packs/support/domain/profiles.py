"""Support FAQs and product guides, independent of legal rules."""

from __future__ import annotations

import re

from domain.ingestion.profiles.guidance import QUESTION_RE, GuidanceDomainProfile
from domain.ingestion.profiles.types import BoundaryLevel
from domain.shared.profile_policy import AnswerPolicy


SUPPORT_POLICY = AnswerPolicy(
    "- Отвечай по статьям поддержки; указывай источник и применимые условия.\n"
    "- Для инструкции сохраняй порядок шагов и предупреждения из источника.\n"
    "- Не выдумывай функции продукта, сроки и тарифы; если данных нет, сообщи об этом.\n"
)


class SupportFaqProfile(GuidanceDomainProfile):
    key = "support_faq"
    display_name = "FAQ поддержки"
    title_pattern = re.compile(r"(?:FAQ\b|Часто задаваемые вопросы\b|Вопросы и ответы\b)", re.IGNORECASE)
    query_patterns: tuple[str, ...] = (r"\bпочему\b", r"\bчто\s+(?:делать|означает)\b")
    query_priority = 1
    answer_policy = SUPPORT_POLICY

    def content_boundaries(self) -> list[BoundaryLevel]:
        return [BoundaryLevel("heading_block", QUESTION_RE, always_split=True, packing_barrier=True)]


class SupportGuideProfile(GuidanceDomainProfile):
    key = "support_guide"
    display_name = "Инструкции поддержки"
    title_pattern = re.compile(r"(?:Инструкция|Руководство|Настройка|Создание)\b", re.IGNORECASE)
    query_patterns: tuple[str, ...] = (r"\bкак\b", r"\bнастро\w+\b", r"\bподключ\w+\b")
    query_priority = 2
    answer_policy = SUPPORT_POLICY
