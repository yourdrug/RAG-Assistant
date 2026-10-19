"""Fallback classifier contract for document indexing."""

from typing import Protocol


class LegacyClassifier(Protocol):
    def __call__(self, text: str, *, threshold: float) -> str: ...
