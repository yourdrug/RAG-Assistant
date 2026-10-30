"""Base classes and shared utilities for domain profiles.

Reduces duplication between LegalDomainProfile and DecreeDomainProfile
by extracting common settings access, effective date extraction, and
shared config defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from domain.ingestion.profiles.types import ConfigDefault, EffectiveDateCandidate
from domain.shared.dates import parse_date_guess


if TYPE_CHECKING:
    import re

    from domain.ingestion.ports import DomainSettingsPort


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
    is_versioned: bool

    @property
    def resolution_strategy(self) -> Literal["version", "replace"]:
        return "version" if self.is_versioned else "replace"

    def __init__(self, settings: DomainSettingsPort | None = None) -> None:
        self._settings = settings

    def get_setting(self, key: str) -> str:
        if self._settings is None:
            msg = (
                f"{type(self).__name__} requires a DomainSettingsPort "
                f"(register via register_all_profiles); missing param: {key}"
            )
            raise RuntimeError(msg)

        return self._settings.get(key, domain_key=self.key)


def extract_effective_date_generic(
    text: str,
    patterns: list[tuple[re.Pattern[str], float]],
    fallback_pattern: re.Pattern[str],
    fallback_confidence: float = 0.3,
    fallback_window: int = 1500,
) -> EffectiveDateCandidate | None:
    """Extract effective date using a list of (pattern, confidence) pairs.

    Tries each pattern in order; first parseable match wins.  Falls back to
    ``fallback_pattern`` applied to the first ``fallback_window`` chars
    (typically the document header / signing block).
    """
    for pattern, confidence in patterns:
        for match in pattern.finditer(text):
            if dt := parse_date_guess(match.group(1)):
                return EffectiveDateCandidate(effective_from=dt, confidence=confidence)

    for match in fallback_pattern.finditer(text[:fallback_window]):
        if dt := parse_date_guess(match.group(1)):
            return EffectiveDateCandidate(signing_date=dt, confidence=fallback_confidence)

    return None
