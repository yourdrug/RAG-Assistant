"""Protocol for preview strategy factory — creates DocumentPreviewStrategy by extension."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from application.documents.ports.preview import DocumentPreviewStrategy


@runtime_checkable
class PreviewStrategyFactoryPort(Protocol):
    """Factory that returns a DocumentPreviewStrategy for a given file extension."""

    def for_extension(
        self,
        extension: str,
        *,
        diag_service: object | None = None,
        domain_registry: object | None = None,
        domain_settings: object | None = None,
    ) -> DocumentPreviewStrategy: ...

    def supported_extensions(self) -> set[str]: ...
