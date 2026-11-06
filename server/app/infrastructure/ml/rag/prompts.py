"""Assemble system instructions and a separate user-role document-context template."""

from __future__ import annotations

from domain.rag.breadth import Breadth
from infrastructure.ml.evidence_profiles.profiles import active_evidence_profile
from infrastructure.ml.rag.prompt_blocks import (
    BROAD_FORMATTING,
    CITATION_CONTEXT_HANDLING_BLOCK,
    CONDITIONAL_RULES_BLOCK,
    CRITICAL_RULES_BLOCK,
    DOCUMENT_CONTEXT_BLOCK,
    INTRO,
    LANGUAGE_PRECISION_BLOCK,
    NARROW_FORMATTING,
    SCOPE_BLOCK,
    SOURCE_REFERENCE_STYLE_BLOCK,
    TABLE_IMAGE_RULES_BLOCK,
)


DECOMPOSITION_ASSESSMENT_SYSTEM = (
    "Оцени, является ли вопрос составным (содержит 2+ независимых подтемы).\n"
    "Если да — разбей его на 2-4 независимых подвопроса.\n"
    "Если нет — верни needs_decomposition=false."
)


SUFFICIENCY_ASSESSMENT_SYSTEM = (
    "Оцени, достаточно ли контекста для ответа на вопрос.\n"
    "Если достаточно — is_sufficient=true.\n"
    "Если нет — is_sufficient=false и предложи уточнённый поисковый запрос для retry.\n"
    "Верни JSON-объект с полями is_sufficient (boolean), reasoning (строка) и "
    "suggested_refinement (строка) на верхнем уровне, без обёрток и описания JSON Schema. "
    'Пример: {"is_sufficient": false, "reasoning": "Нет данных", '
    '"suggested_refinement": "Уточнённый запрос"}.'
)


def build_context_message() -> str:
    """Return the document-context message template with ``{context}`` placeholder.

    This is passed as a **separate user-role message** (not system) so that
    retrieved content runs at user privilege level, not system privilege.
    """
    return DOCUMENT_CONTEXT_BLOCK


def build_system_prompt(
    breadth: str = Breadth.NARROW,
    domain_addendum: str | None = None,
    *,
    enumerate_cases: bool = False,
    question: str = "",
) -> str:
    """Build the system prompt text based on question breadth and context composition.

    Returns the raw system prompt string **without** the document-context
    block — retrieved content is passed as a separate user-role message via
    ``build_context_message()`` (defense-in-depth: context at user privilege,
    not system privilege).

    ``domain_addendum``: extra rules from the active ``DomainProfile`` (e.g.
    legal citation rules, temporal rules when ``as_of_date`` is set). This is
    the single source of truth for domain-specific prompt rules. The profile
    provides its own rules via ``prompt_addendum()``.

    ``enumerate_cases``: when True, adds instructions for the LLM to enumerate
    all conditional rules/cases separately (saves tokens on simple questions).
    """
    formatting_rules = BROAD_FORMATTING if breadth == Breadth.BROAD else NARROW_FORMATTING

    parts: list[str] = [
        INTRO,
        SCOPE_BLOCK,
        CRITICAL_RULES_BLOCK,
        LANGUAGE_PRECISION_BLOCK,
        CITATION_CONTEXT_HANDLING_BLOCK,
        "<formatting_rules>\n" + formatting_rules + "\n</formatting_rules>",
        TABLE_IMAGE_RULES_BLOCK,
        *active_evidence_profile.get().reading_rules(question),
        SOURCE_REFERENCE_STYLE_BLOCK,
    ]

    if enumerate_cases:
        parts.append(CONDITIONAL_RULES_BLOCK)

    if domain_addendum:
        safe_addendum = domain_addendum.strip().replace("{", "{{").replace("}", "}}")
        parts.append("<domain_specific_rules>\n" + safe_addendum + "\n</domain_specific_rules>")

    return "\n\n".join(parts)
