"""Config admin ports — abstract interfaces for system info providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OllamaProbePort(Protocol):
    async def get_models(self) -> list[str]: ...


@runtime_checkable
class VectorDBInfoPort(Protocol):
    def get_status(self) -> str: ...
    def get_collections(self) -> list[dict]: ...
