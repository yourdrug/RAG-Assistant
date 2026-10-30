"""Public values and contracts of this domain package."""

from domain.ingestion.profiles.protocol import DocumentProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry


__all__ = ["DocumentProfile", "DomainProfileRegistry"]
