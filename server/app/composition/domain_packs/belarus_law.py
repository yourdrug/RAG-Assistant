"""Composition of the Belarusian law pack: act edition scenarios and their API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.domain_packs.belarus_law.act_versioning.document_hooks import ActEditionHooks
from application.domain_packs.belarus_law.act_versioning.service import ActVersioningService
from composition.domain_packs import PackWiring
from presentation.domain_packs.belarus_law.act_versions_routes import router as act_versions_router


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import APIRouter

    from application.indexing.pack_hooks import DocumentPackHooks
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow import UnitOfWorkFactory
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry


def document_hooks(
    uow_factory: UnitOfWorkFactory,
    domain_registry: DomainProfileRegistry,
    domain_settings: DomainSettingsPort,
    cache_invalidator: CacheInvalidatorPort | None,
) -> DocumentPackHooks:
    service = ActVersioningService(uow_factory, domain_settings, cache_invalidator)
    return ActEditionHooks(service, domain_registry)


def routers() -> Sequence[APIRouter]:
    return (act_versions_router,)


BELARUS_LAW_WIRING = PackWiring(document_hooks=document_hooks, routers=routers)
