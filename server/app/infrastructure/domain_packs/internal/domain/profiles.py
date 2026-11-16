"""Internal instructions and procedures, without legal edition semantics."""

import re

from domain.ingestion.profiles.guidance import GuidanceDomainProfile
from domain.shared.profile_policy import AnswerPolicy


INTERNAL_POLICY = AnswerPolicy(
    "- Отвечай по внутренним источникам с указанием документа и раздела.\n"
    "- Различай обязательные требования и рекомендации; сохраняй роли, сроки и условия.\n"
    "- Не придумывай полномочия, согласования и исключения; при пробеле укажи недостающие данные.\n"
)


class InternalInstructionProfile(GuidanceDomainProfile):
    key = "internal_instruction"
    display_name = "Внутренние инструкции"
    title_pattern = re.compile(r"(?:Инструкция|Руководство|Памятка)\b", re.IGNORECASE)
    query_patterns: tuple[str, ...] = (r"\bкак\b", r"\bнастро\w+\b")
    query_priority = 1
    answer_policy = INTERNAL_POLICY


class InternalProcedureProfile(GuidanceDomainProfile):
    key = "internal_procedure"
    display_name = "Внутренние регламенты"
    title_pattern = re.compile(r"(?:Регламент|Положение|Порядок|Политика)\b", re.IGNORECASE)
    query_patterns: tuple[str, ...] = (r"\bкто\b", r"\bсогласова\w+\b", r"\bрегламент\w*\b", r"\bсрок\w*\b")
    query_priority = 2
    answer_policy = INTERNAL_POLICY
