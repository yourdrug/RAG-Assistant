"""Output scanner — detect system prompt leaks and injection echo in LLM output.

Checks the LLM's response for:
1. System prompt extraction (verbatim or paraphrased instruction content)
2. Injection echo (LLM repeating injected instructions)
3. Marker forgery in output (fake context markers)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("default")

# Phrases that indicate system prompt leakage
_LEAK_PATTERNS = [
    re.compile(
        r"(?:мой|my)\s+(?:системный\s+промпт|system\s+prompt)\s*[:=]", re.IGNORECASE,
    ),
    re.compile(
        r"(?:вот|here\s+is|below\s+is)\s+(?:мой|my)\s+"
        r"(?:системный|system)\s+(?:промпт|prompt|instructions?)", re.IGNORECASE,
    ),
    re.compile(
        r"(?:я\s+получил|I\s+(?:was\s+)?(?:given|received))\s+"
        r"(?:следующие|the\s+following)\s+(?:инструкции|instructions?)", re.IGNORECASE,
    ),
    re.compile(r"<critical_rules>|<scope>|<untrusted_context_handling>", re.IGNORECASE),
    re.compile(r"твои?\s+инструкции\s+таковы", re.IGNORECASE),
]

# Phrases that indicate the LLM is echoing injected instructions
_ECHO_PATTERNS = [
    re.compile(
        r"(?:игнорирую|ignoring)\s+(?:все\s+)?"
        r"(?:предыдущие|previous)\s+(?:инструкции|instructions?)", re.IGNORECASE,
    ),
    re.compile(
        r"(?:я\s+теперь|I\s+am\s+now)\s+(?:DAN|unrestricted|без\s+ограничений)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:режим\s+разработчика|developer\s+mode)\s+"
        r"(?:активирован|activated|enabled)", re.IGNORECASE,
    ),
]

# Fake context markers in output (attacker trying to inject markers via LLM)
_MARKER_FORGERY_RE = re.compile(r"<<(?:END_)?DOCUMENT_CONTEXT>>")


@dataclass(frozen=True)
class OutputVerdict:
    """Result of scanning LLM output for security issues."""

    safe: bool
    leak_detected: bool
    echo_detected: bool
    marker_forgery: bool
    matches: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.safe:
            return f"output_issue:{','.join(self.matches)}"
        return "clean"


class OutputScanner:
    """Scan LLM output for security issues.

    Usage::

        scanner = OutputScanner()
        verdict = scanner.scan(llm_output)
        if not verdict.safe:
            log.warning("Output issue: %s", verdict.reason)
    """

    def scan(self, text: str) -> OutputVerdict:
        """Scan LLM output for leaks, echoes, and marker forgery."""
        matches: list[str] = []

        leak_detected = False
        for i, rx in enumerate(_LEAK_PATTERNS):
            if rx.search(text):
                leak_detected = True
                matches.append(f"leak_p{i}")

        echo_detected = False
        for i, rx in enumerate(_ECHO_PATTERNS):
            if rx.search(text):
                echo_detected = True
                matches.append(f"echo_p{i}")

        marker_forgery = bool(_MARKER_FORGERY_RE.search(text))
        if marker_forgery:
            matches.append("marker_forgery")

        safe = not leak_detected and not echo_detected and not marker_forgery

        if not safe:
            log.warning("OutputScanner issue: %s text=%.200r", matches, text)

        return OutputVerdict(
            safe=safe,
            leak_detected=leak_detected,
            echo_detected=echo_detected,
            marker_forgery=marker_forgery,
            matches=matches,
        )

    def clean_output(self, text: str) -> str:
        """Remove marker forgery from output (defense-in-depth)."""
        return _MARKER_FORGERY_RE.sub("", text)
