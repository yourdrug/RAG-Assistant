"""Normalize untrusted text and escape prompt control markers before interpolation."""

from __future__ import annotations

from domain.services.injection_patterns import normalize_unicode


# Replace double brackets before single ones to preserve the visible delimiters.
_MARKER_ESCAPE_TABLE: dict[str, str] = {
    "\u003c\u003c": "\u2039\u2039",  # << → ‹‹
    "\u003e\u003e": "\u203a\u203a",  # >> → ››
    "\u003c": "\u2039",  # < → ‹
    "\u003e": "\u203a",  # > → ›
}


def sanitize_for_prompt(text: str) -> str:
    """Normalize and escape untrusted text to prevent prompt injection.

    1. Normalizes Unicode (strips zero-width chars, NFKC fold) to defeat
       obfuscation-based injection.
    2. Replaces ``<<``, ``>>``, ``<``, ``>`` with visually similar Unicode
       angle quotation marks so that an attacker cannot:

    * forge ``<<END_DOCUMENT_CONTEXT>>`` to break out of the sandboxed
      context block;
    * inject XML-style tags (``</untrusted_context_handling>``,
      ``<critical_rules>``) to close or spoof system-prompt blocks.
    """
    text = normalize_unicode(text)
    for raw, escaped in _MARKER_ESCAPE_TABLE.items():
        text = text.replace(raw, escaped)
    return text
