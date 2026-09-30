"""Input scanner — prompt injection detection for user-supplied text.

Wraps domain-level injection patterns into a guardrail that can be called
from the RAG pipeline before text reaches the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from domain.services.injection_patterns import (
    detect_injection,
    has_blocking_injection,
    has_flagged_injection,
    normalize_unicode,
)

log = logging.getLogger("default")


@dataclass(frozen=True)
class InputVerdict:
    """Result of scanning untrusted input for injection patterns."""

    safe: bool
    blocked: bool
    flagged: bool
    matches: list[tuple[str, str]] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.blocked:
            return f"injection_detected:{','.join(m[0] for m in self.matches)}"
        if self.flagged:
            return f"injection_suspicious:{','.join(m[0] for m in self.matches)}"
        return "clean"


class InputScanner:
    """Scan user input for prompt injection attempts.

    Usage::

        scanner = InputScanner()
        verdict = scanner.scan(user_question)
        if verdict.blocked:
            raise ValueError(f"Blocked: {verdict.reason}")
    """

    def __init__(self, *, block_on_flag: bool = False) -> None:
        self._block_on_flag = block_on_flag

    def scan(self, text: str) -> InputVerdict:
        """Scan text for injection patterns. Returns an InputVerdict."""
        matches = detect_injection(text)
        blocked = has_blocking_injection(text)
        flagged = has_flagged_injection(text)

        if self._block_on_flag:
            blocked = blocked or flagged

        safe = not blocked and not flagged

        if blocked:
            log.warning("InputScanner BLOCKED: matches=%s text_chars=%d", matches, len(text))
        elif flagged:
            log.info("InputScanner FLAGGED: matches=%s text_chars=%d", matches, len(text))

        return InputVerdict(safe=safe, blocked=blocked, flagged=flagged, matches=matches)

    def normalize(self, text: str) -> str:
        """Normalize text to defeat Unicode obfuscation."""
        return normalize_unicode(text)
