"""Base classes and shared utilities for domain profiles.

Reduces duplication between LegalDomainProfile and DecreeDomainProfile
by extracting common settings access, effective date extraction, and
shared config defaults.
"""

from __future__ import annotations

import re
from datetime import date

from domain.domain_profile.date_parsing import parse_date_guess
from domain.domain_profile.protocol import ConfigDefault, EffectiveDateCandidate

EFFECTIVE_DATE_TRUST_THRESHOLD_DEFAULT = ConfigDefault(
    "effective_date_auto_trust_threshold",
    "0.85",
    "float",
    "Min confidence to auto-trust extracted effective date",
    0.0,
    1.0,
)


class SettingsBackedProfile:
    """Base mixin for profiles that read config via DomainSettingsPort."""

    key: str  # must be set by subclass

    def __init__(self, settings=None) -> None:
        self._settings = settings

    def _get(self, key: str) -> str:
        if self._settings is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a DomainSettingsPort "
                f"(register via register_all_profiles); missing param: {key}"
            )
        return self._settings.get(key, domain_key=self.key)


def extract_effective_date_generic(
        text: str,
        patterns: list[tuple[re.Pattern, float]],
        fallback_pattern: re.Pattern,
        fallback_confidence: float = 0.3,
        fallback_window: int = 1500,
) -> EffectiveDateCandidate | None:
    """Extract effective date using a list of (pattern, confidence) pairs.

    Tries each pattern in order; first match wins.  Falls back to
    ``fallback_pattern`` applied to the first ``fallback_window`` chars
    (typically the document header / signing block).
    """
    for pattern, confidence in patterns:
        if m := pattern.search(text):
            if dt := parse_date_guess(m.group(1)):
                return EffectiveDateCandidate(effective_from=dt, confidence=confidence)
    if m := fallback_pattern.search(text[:fallback_window]):
        if dt := parse_date_guess(m.group(1)):
            return EffectiveDateCandidate(signing_date=dt, confidence=fallback_confidence)
    return None


def versioned_prompt_date_stamp(as_of_date: date | None) -> str:
    """Generate the shared date-stamp bullet for versioned domain prompt addenda."""

    if as_of_date is None:
        return ""
    return (
        f"- Ответ дан по состоянию на {as_of_date.strftime('%d.%m.%Y')}. "
        "Если подходят разные редакции — укажи, какая редакция использована.\n"
    )
