"""Release streamed answer text only after the output guardrails accept it."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


# Longer than any output-scanner phrase or PII value, so a match is complete
# (blocked or redacted) before any of its characters are released.
HOLDBACK_CHARS = 160
# Re-sanitizing the whole text on every token would be quadratic; check in steps.
CHECK_STEP_CHARS = 48


class GuardedAnswerStream:
    """Stream a sanitized prefix of the generated answer.

    ``sanitize`` receives the whole raw text generated so far and returns the
    publishable text and whether it is safe. Only the part ending ``holdback``
    characters before its end is released. Release stops for good once a check
    is unsafe or would rewrite already released text; the final sanitized answer
    stays authoritative and is delivered separately.
    """

    def __init__(
        self,
        sanitize: Callable[[str], tuple[str, bool]],
        *,
        holdback: int = HOLDBACK_CHARS,
        step: int = CHECK_STEP_CHARS,
    ) -> None:
        self.sanitize = sanitize
        self.holdback = holdback
        self.step = step
        self.raw = ""
        self.released = ""
        self.checked_chars = 0
        self.stopped = False

    def push(self, text: str) -> str:
        """Add generated text; return newly releasable sanitized text."""
        self.raw += text

        if self.stopped or len(self.raw) - self.checked_chars < self.step:
            return ""

        self.checked_chars = len(self.raw)
        safe_text, safe = self.sanitize(self.raw)

        if not safe:
            self.stopped = True
            return ""

        stable = safe_text[: max(0, len(safe_text) - self.holdback)]

        if not stable.startswith(self.released):
            self.stopped = True
            return ""

        delta = stable[len(self.released) :]
        self.released = stable
        return delta

    def finish(self, final: str) -> str:
        """Return the rest of the final answer when it continues the released text."""
        if not final.startswith(self.released):
            return ""

        rest = final[len(self.released) :]
        self.released = final
        return rest
