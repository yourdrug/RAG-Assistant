"""DomainProfileRegistry — classification engine with fingerprint → density-score fallback.

Two-level classification:
1. Deterministic fingerprint (format-specific markers → instant match)
2. Density-score with sticky prior and margin-based ambiguity detection

Edge cases handled:
- Short/noisy text → inherits parent domain (MIN_CLASSIFIABLE_CHARS)
- Two fingerprints fire → ambiguous, logged for review
- Borderline score between domains → is_ambiguous=True
- Unknown document type → general with low confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.ports.domain_settings import DomainSettingsPort
from domain.domain_profile.protocol import DomainProfile
from domain.value_objects.doc_domain import DocDomain

MIN_CLASSIFIABLE_CHARS = 120
MARGIN_RATIO = 1.3
STICKY_BONUS_RATIO = 0.5


@dataclass(frozen=True)
class ClassificationResult:
    domain_key: str
    confidence: float  # 0..1
    is_ambiguous: bool
    is_low_signal: bool
    candidate_scores: dict[str, float] = field(default_factory=dict)


class DomainProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, DomainProfile] = {}

    def register(self, profile: DomainProfile) -> None:
        self._profiles[profile.key] = profile

    def get(self, key: str) -> DomainProfile:
        return self._profiles[key]

    def all(self) -> list[DomainProfile]:
        return list(self._profiles.values())

    def classify(
        self,
        text: str,
        *,
        prior_domain: str | None = None,
        settings: DomainSettingsPort,
    ) -> ClassificationResult:
        general = DocDomain.GENERAL.value
        candidates = [p for p in self._profiles.values() if p.key != general]

        # Step 0: too short — don't guess, inherit parent or general
        if len(text.strip()) < MIN_CLASSIFIABLE_CHARS:
            fallback = prior_domain if prior_domain and prior_domain in self._profiles else general
            return ClassificationResult(
                fallback,
                confidence=0.3,
                is_ambiguous=False,
                is_low_signal=True,
                candidate_scores={},
            )

        # Step 1: deterministic fingerprint (priority over density-score)
        fingerprint_hits = [p.key for p in candidates if p.structural_fingerprint(text)]
        if len(fingerprint_hits) == 1:
            return ClassificationResult(
                fingerprint_hits[0],
                confidence=1.0,
                is_ambiguous=False,
                is_low_signal=False,
                candidate_scores={},
            )
        if len(fingerprint_hits) > 1:
            return ClassificationResult(
                fingerprint_hits[0],
                confidence=0.4,
                is_ambiguous=True,
                is_low_signal=False,
                candidate_scores=dict.fromkeys(fingerprint_hits, 1.0),
            )

        # Step 2: density-score with sticky-prior and margin-check
        scores: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        for p in candidates:
            raw = p.classify_score(text)
            threshold = float(settings.get("classification_threshold", domain_key=p.key))
            if prior_domain == p.key and raw > 0:
                raw *= 1.0 + STICKY_BONUS_RATIO
            scores[p.key] = raw
            thresholds[p.key] = threshold

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return ClassificationResult(
                general,
                confidence=1.0,
                is_ambiguous=False,
                is_low_signal=False,
                candidate_scores=scores,
            )

        top_key, top_score = ranked[0]
        top_threshold = thresholds.get(top_key, 0.0)

        if top_score < top_threshold:
            return ClassificationResult(
                general,
                confidence=1.0 - top_score / max(top_threshold, 1e-6),
                is_ambiguous=False,
                is_low_signal=False,
                candidate_scores=scores,
            )

        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        is_ambiguous = second_score > 0 and (top_score / max(second_score, 1e-6)) < MARGIN_RATIO
        confidence = min(1.0, (top_score - top_threshold) / top_threshold) if not is_ambiguous else 0.5

        return ClassificationResult(top_key, confidence, is_ambiguous, False, scores)
