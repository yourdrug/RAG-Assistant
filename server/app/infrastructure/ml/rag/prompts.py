"""Assemble system instructions and a separate user-role document-context template."""

from __future__ import annotations

import re

from domain.rag.breadth import AnswerStyle, stored_answer_style
from infrastructure.ml.evidence_profiles.profiles import current_evidence_profile
from infrastructure.ml.rag.prompt_blocks import (
    ADAPTIVE_FORMATTING,
    CITATION_CONTEXT_HANDLING_BLOCK,
    COMPLETENESS_BLOCK,
    CONCISE_FORMATTING,
    CONDITIONAL_RULES_BLOCK,
    CRITICAL_RULES_BLOCK,
    DETAILED_FORMATTING,
    DOCUMENT_CONTEXT_BLOCK,
    INTRO,
    LANGUAGE_PRECISION_BLOCK,
    SCOPE_BLOCK,
    SOURCE_REFERENCE_COMPACT_BLOCK,
    SOURCE_REFERENCE_STYLE_BLOCK,
    TABLE_IMAGE_RULES_BLOCK,
)


STYLE_FORMATTING: dict[AnswerStyle, str] = {
    AnswerStyle.CONCISE: CONCISE_FORMATTING,
    AnswerStyle.ADAPTIVE: ADAPTIVE_FORMATTING,
    AnswerStyle.DETAILED: DETAILED_FORMATTING,
}


DECOMPOSITION_ASSESSMENT_SYSTEM = (
    "<task>\n"
    "Оцени, является ли вопрос составным (содержит 2+ независимых подтемы).\n"
    "</task>\n"
    "<output_format>\n"
    "Если да — разбей его на 2-4 независимых подвопроса.\n"
    "Если нет — верни needs_decomposition=false.\n"
    "</output_format>"
)


def build_context_message() -> str:
    """Return the document-context message template with ``{context}`` placeholder.

    This is passed as a **separate user-role message** (not system) so that
    retrieved content runs at user privilege level, not system privilege.
    """
    return DOCUMENT_CONTEXT_BLOCK


def build_system_prompt(
    style: str = AnswerStyle.CONCISE,
    domain_addendum: str | None = None,
    *,
    enumerate_cases: bool = False,
    question: str = "",
) -> str:
    """Build the system prompt text for an answer style and the context composition.

    Returns the raw system prompt string **without** the document-context
    block — retrieved content is passed as a separate user-role message via
    ``build_context_message()`` (defense-in-depth: context at user privilege,
    not system privilege).

    The completeness and grounding rules are identical for every style; ``style``
    changes only how the answer is presented (and how compactly sources are cited).
    Legacy breadth labels of older results map explicitly ("narrow" -> concise,
    "broad" -> detailed); an unknown value is an error, never a silent concise prompt.

    ``domain_addendum``: extra rules from the active ``DomainProfile`` (e.g.
    legal citation rules, temporal rules when ``as_of_date`` is set). This is
    the single source of truth for domain-specific prompt rules. The profile
    provides its own rules via ``prompt_addendum()``.

    ``enumerate_cases``: when True, adds the per-category completeness rules for
    conditional evidence; it never changes the presentation style.
    """
    resolved = stored_answer_style(style)
    source_style = (
        SOURCE_REFERENCE_COMPACT_BLOCK if resolved == AnswerStyle.CONCISE else SOURCE_REFERENCE_STYLE_BLOCK
    )

    parts: list[str] = [
        INTRO,
        SCOPE_BLOCK,
        CRITICAL_RULES_BLOCK,
        COMPLETENESS_BLOCK,
        LANGUAGE_PRECISION_BLOCK,
        CITATION_CONTEXT_HANDLING_BLOCK,
        "<formatting_rules>\n" + STYLE_FORMATTING[resolved] + "\n</formatting_rules>",
        TABLE_IMAGE_RULES_BLOCK,
        *current_evidence_profile().reading_rules(question),
        source_style,
    ]

    if enumerate_cases or re.search(
        r"\b(?:ровно|не\s+(?:менее|более|ниже|выше)|свыше|сверх|превыш\w*|"
        r"менее|более|исключени\w*|при\s+условии)\b",
        question,
        re.IGNORECASE,
    ):
        parts.append(CONDITIONAL_RULES_BLOCK)

    if domain_addendum:
        safe_addendum = domain_addendum.strip().replace("{", "{{").replace("}", "}}")
        parts.append("<domain_specific_rules>\n" + safe_addendum + "\n</domain_specific_rules>")

    return "\n\n".join(parts)
