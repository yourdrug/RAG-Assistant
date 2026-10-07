"""Assemble system instructions and a separate user-role document-context template."""

from __future__ import annotations

from domain.value_objects.llm_provider import Breadth
from .evidence_reading import evidence_reading_rules

from .prompt_blocks import (
    _INTRO,
    _SCOPE_BLOCK,
    _CRITICAL_RULES_BLOCK,
    _LANGUAGE_PRECISION_BLOCK,
    _CITATION_CONTEXT_HANDLING_BLOCK,
    _NARROW_FORMATTING,
    _BROAD_FORMATTING,
    _TABLE_IMAGE_RULES_BLOCK,
    _CONDITIONAL_RULES_BLOCK,
    _SOURCE_REFERENCE_STYLE_BLOCK,
    _DOCUMENT_CONTEXT_BLOCK,
)


DECOMPOSITION_ASSESSMENT_SYSTEM = (
    "Оцени, является ли вопрос составным (содержит 2+ независимых подтемы).\n"
    "Если да — разбей его на 2-4 независимых подвопроса.\n"
    "Если нет — верни needs_decomposition=false."
)


SUFFICIENCY_ASSESSMENT_SYSTEM = (
    "Оцени, достаточно ли контекста для ответа на вопрос.\n"
    "Если достаточно — is_sufficient=true.\n"
    "Если нет — is_sufficient=false и предложи уточнённый поисковый запрос для retry."
)


def build_context_message() -> str:
    """Return the document-context message template with ``{context}`` placeholder.

    This is passed as a **separate user-role message** (not system) so that
    retrieved content runs at user privilege level, not system privilege.
    """
    return _DOCUMENT_CONTEXT_BLOCK


def build_system_prompt(
    breadth: str = Breadth.NARROW,
    domain_addendum: str | None = None,
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
    formatting_rules = _BROAD_FORMATTING if breadth == Breadth.BROAD else _NARROW_FORMATTING

    parts: list[str] = [
        _INTRO,
        _SCOPE_BLOCK,
        _CRITICAL_RULES_BLOCK,
        _LANGUAGE_PRECISION_BLOCK,
        _CITATION_CONTEXT_HANDLING_BLOCK,
        "<formatting_rules>\n" + formatting_rules + "\n</formatting_rules>",
        _TABLE_IMAGE_RULES_BLOCK,
        *evidence_reading_rules(question),
        _SOURCE_REFERENCE_STYLE_BLOCK,
    ]

    if enumerate_cases:
        parts.append(_CONDITIONAL_RULES_BLOCK)

    if domain_addendum:
        safe_addendum = domain_addendum.strip().replace("{", "{{").replace("}", "}}")
        parts.append("<domain_specific_rules>\n" + safe_addendum + "\n</domain_specific_rules>")

    return "\n\n".join(parts)
