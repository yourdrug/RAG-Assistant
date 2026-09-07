"""DecreeDomainProfile — presidential decrees and government resolutions.

Structural fingerprint: "УКАЗ" header + "ПОСТАНОВЛЯЮ/ПОСТАНОВЛЯЕТ" marker.
Content-based splitting by point/subpoint/sentence.
"""

from __future__ import annotations

import re
from datetime import date

from domain.domain_profile.date_parsing import parse_date_guess
from domain.domain_profile.protocol import (
    BoundaryLevel,
    ConfigDefault,
    EffectiveDateCandidate,
    ReferenceMatch,
)

_DECREE_TITLE_RE = re.compile(r"^\s*УКАЗ\b", re.MULTILINE)
_POSTANOVLYAYU_RE = re.compile(r"ПОСТАНОВЛЯ[ЮЕ]\s*:?", re.IGNORECASE)
_DECREE_NUMBER_RE = re.compile(r"№\s*(\d+)")
_DECREE_DATE_RE = re.compile(r"(\d{1,2}\s+\S+\s+\d{4}\s*г?\.?)")
_POINT_RE = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
_SUBPOINT_RE = re.compile(r"^\s*([а-я])\)\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_EFFECTIVE_PATTERNS = [
    (re.compile(r"вступает в силу\s+(?:со дня|с)\s+([^\n,.]+)", re.IGNORECASE), 0.9),
    (re.compile(r"с\s+даты\s+([^\n,.]+)", re.IGNORECASE), 0.5),
]


class DecreeDomainProfile:
    key = "decree"
    display_name = "Указы и постановления"
    is_versioned = True

    def __init__(self, settings=None) -> None:
        self._settings = settings  # единственный источник любых числовых параметров

    def _get(self, key: str) -> str:
        if self._settings is None:
            raise RuntimeError(
                f"DecreeDomainProfile requires a DomainSettingsPort "
                f"(register via register_all_profiles); missing param: {key}"
            )
        return self._settings.get(key, domain_key=self.key)

    def config_defaults(self) -> list[ConfigDefault]:
        return [
            ConfigDefault(
                "max_unit_chars",
                "1200",
                "int",
                "Safety-net: above this size, split to finer boundary level",
                200,
                8000,
            ),
            ConfigDefault(
                "fingerprint_min_points",
                "1",
                "int",
                "Minimum numbered points for fingerprint detection",
                0,
                10,
            ),
            ConfigDefault(
                "classification_threshold",
                "2.0",
                "float",
                "Density-score threshold when fingerprint doesn't fire",
                0.0,
                20.0,
            ),
            ConfigDefault(
                "effective_date_auto_trust_threshold",
                "0.85",
                "float",
                "Min confidence to auto-trust extracted effective date",
                0.0,
                1.0,
            ),
        ]

    def structural_fingerprint(self, text: str) -> bool:
        min_points = int(self._get("fingerprint_min_points"))
        has_title = bool(_DECREE_TITLE_RE.search(text))
        has_postanovlyayu = bool(_POSTANOVLYAYU_RE.search(text))
        enough_points = len(_POINT_RE.findall(text)) >= min_points
        return has_title and has_postanovlyayu and enough_points

    def classify_score(self, text: str) -> float:
        score = 0.0
        if _DECREE_TITLE_RE.search(text):
            score += 3.0
        if _POSTANOVLYAYU_RE.search(text):
            score += 2.0
        score += 0.3 * len(_POINT_RE.findall(text))
        return score

    def content_boundaries(self) -> list[BoundaryLevel]:
        return [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint", _SUBPOINT_RE),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]

    def extract_references(self, text: str) -> list[ReferenceMatch]:
        refs = []
        if m := _DECREE_NUMBER_RE.search(text):
            refs.append(ReferenceMatch("decree_number", m.group(1)))
        if m := _DECREE_DATE_RE.search(text):
            refs.append(ReferenceMatch("decree_date", m.group(1)))
        refs += [ReferenceMatch("point", m) for m in _POINT_RE.findall(text)]
        return refs

    def extract_effective_date(self, text: str) -> EffectiveDateCandidate | None:
        for pattern, confidence in _EFFECTIVE_PATTERNS:
            m = pattern.search(text)
            if m:
                dt = parse_date_guess(m.group(1))
                if dt is not None:
                    return EffectiveDateCandidate(effective_from=dt, confidence=confidence)

        m = _DECREE_DATE_RE.search(text[:1500])
        if m:
            dt = parse_date_guess(m.group(1))
            if dt is not None:
                return EffectiveDateCandidate(signing_date=dt, confidence=0.3)
        return None

    def prompt_addendum(self, breadth: str, as_of_date: date | None = None) -> str | None:
        rules = (
            "13. Каждый указ идентифицируй по номеру и дате.\n"
            "14. Ссылайся на конкретный пункт постановляющей части, не пересказывай указ целиком.\n"
        )
        if as_of_date is not None:
            rules += (
                f"15. Ответ дан по состоянию на {as_of_date.strftime('%d.%m.%Y')}. "
                "Если подходят разные редакции — укажи, какая редакция использована.\n"
            )
        return rules

    def retrieval_fallback_to_full_corpus(self) -> bool:
        return True
