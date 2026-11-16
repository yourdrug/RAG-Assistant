"""Catalog of the domain packs loaded in this installation, as Application sees them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.spaces.ports import DomainPackNotLoadedError


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import date

    from application.indexing.pack_hooks import DocumentPackHooks
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow import UnitOfWorkFactory
    from composition.domain_packs import LoadedPack
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.knowledge.pack_metadata import MetadataCondition
    from infrastructure.domain_profile.packs import DomainPack

    type ChunkSelection = Callable[[date | None], MetadataCondition | None]


class LoadedPackCatalog:
    """Profiles, evidence rules and document hooks of each loaded pack, by pack key."""

    def __init__(
        self,
        packs: Mapping[str, LoadedPack],
        registries: Mapping[str, DomainProfileRegistry],
        *,
        uow_factory: UnitOfWorkFactory,
        domain_settings: DomainSettingsPort,
        cache_invalidator: CacheInvalidatorPort | None,
    ) -> None:
        self._packs = dict(packs)
        self._registries = dict(registries)
        self._uow_factory = uow_factory
        self._domain_settings = domain_settings
        self._cache_invalidator = cache_invalidator
        self._hooks: dict[str, DocumentPackHooks | None] = {}

    def pack_keys(self) -> tuple[str, ...]:
        return tuple(self._packs)

    def pack(self, pack_key: str) -> DomainPack:
        return self.loaded(pack_key).pack

    def chunk_selections(self) -> dict[str, ChunkSelection]:
        """Each pack's selection by pack key, for scenarios that search outside a RAG request."""
        return {key: loaded.pack.evidence_profile.chunk_selection for key, loaded in self._packs.items()}

    def domain_registry(self, pack_key: str) -> DomainProfileRegistry:
        self.loaded(pack_key)
        return self._registries[pack_key]

    def document_hooks(self, pack_key: str) -> DocumentPackHooks | None:
        if pack_key not in self._hooks:
            self._hooks[pack_key] = self.build_hooks(pack_key, self._uow_factory)

        return self._hooks[pack_key]

    def build_hooks(self, pack_key: str, uow_factory: UnitOfWorkFactory) -> DocumentPackHooks | None:
        factory = self.loaded(pack_key).wiring.document_hooks

        if factory is None:
            return None

        return factory(
            uow_factory, self.domain_registry(pack_key), self._domain_settings, self._cache_invalidator
        )

    def loaded(self, pack_key: str) -> LoadedPack:
        try:
            return self._packs[pack_key]
        except KeyError:
            raise DomainPackNotLoadedError(pack_key) from None
