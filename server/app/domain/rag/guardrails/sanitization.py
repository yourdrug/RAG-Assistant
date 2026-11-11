"""Normalize untrusted text and escape prompt control markers before interpolation."""

from __future__ import annotations

import re

from domain.rag.guardrails.injection import normalize_unicode


# Escape prompt delimiters without corrupting XML examples or comparisons.
_MARKER_ESCAPE_TABLE: dict[str, str] = {
    "\u003c\u003c": "\u2039\u2039",
    "\u003e\u003e": "\u203a\u203a",
}

# Every system-prompt block tag. Prompt text lives in Infrastructure, so this
# list cannot be derived from it here; test_prompt_block_tags_are_escaped
# fails when a prompt introduces a block tag missing from this tuple.
PROMPT_BLOCK_TAGS: tuple[str, ...] = (
    "scope",
    "critical_rules",
    "language_and_precision",
    "untrusted_context_handling",
    "formatting_rules",
    "table_and_image_rules",
    "conditional_rules_expansion",
    "source_reference_style",
    "domain_specific_rules",
    "conversation_summary",
    "table_column_reading",
    "norm_subject_reading",
    "procedure_reading",
    "enumeration_reading",
    "editorial_note_reading",
)
SYSTEM_BLOCK_TAG = re.compile(
    r"<\s*/?\s*(?:" + "|".join(map(re.escape, PROMPT_BLOCK_TAGS)) + r")(?=[\s/>])[^>]*>",
    re.IGNORECASE,
)


def sanitize_for_prompt(text: str) -> str:
    """Normalize and escape untrusted text to prevent prompt injection.

    1. Normalizes Unicode (strips zero-width chars, NFKC fold) to defeat
       obfuscation-based injection.
    2. Replaces ``<<``, ``>>`` and system-block tags with Unicode
       angle quotation marks so that an attacker cannot:

    * forge ``<<END_DOCUMENT_CONTEXT>>`` to break out of the sandboxed
      context block;
    * inject XML-style tags (``</untrusted_context_handling>``,
      ``<critical_rules>``) to close or spoof system-prompt blocks.
    """
    text = normalize_unicode(text)

    for raw, escaped in _MARKER_ESCAPE_TABLE.items():
        text = text.replace(raw, escaped)

    return SYSTEM_BLOCK_TAG.sub(lambda match: match[0].replace("<", "\u2039").replace(">", "\u203a"), text)
