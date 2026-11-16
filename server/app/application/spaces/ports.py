"""Contracts between space scenarios and the domain packs loaded in this installation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from domain.shared.exceptions import BusinessRuleViolationError


if TYPE_CHECKING:
    from application.indexing.pack_hooks import DocumentPackHooks
    from domain.ingestion.profiles.registry import DomainProfileRegistry


class DomainPackNotLoadedError(BusinessRuleViolationError):
    """A space refers to a pack whose code this installation does not load."""

    def __init__(self, pack_key: str) -> None:
        super().__init__(
            message=f"Domain pack {pack_key!r} is not available in this installation",
            errors={"pack_key": pack_key},
        )


class DomainPackCatalog(Protocol):
    """Packs loaded in this installation; a space may only refer to one of them."""

    def pack_keys(self) -> tuple[str, ...]: ...

    def domain_registry(self, pack_key: str) -> DomainProfileRegistry:
        """Document profiles of the pack; raises ``DomainPackNotLoadedError`` for an unknown key."""
        ...

    def document_hooks(self, pack_key: str) -> DocumentPackHooks | None:
        """Document lifecycle hooks of the pack; ``None`` for a pack without them."""
        ...
