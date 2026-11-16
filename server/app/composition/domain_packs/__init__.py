"""Explicit registry of domain packs; every registered pack is loaded automatically.

The neutral pack is always loaded. A knowledge space selects one registered pack,
so its documents and questions use that pack's rules. A pack has an Infrastructure
definition (profiles, evidence rules, repositories) and an optional Composition wiring of
its scenarios: document lifecycle hooks and API routes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Protocol

from application.runtime.errors import ConfigurationError
from infrastructure.domain_profile.packs import NEUTRAL_PACK, DomainPack


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from fastapi import APIRouter

    from application.indexing.pack_hooks import DocumentPackHooks
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow import UnitOfWorkFactory
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry


class DocumentHooksFactory(Protocol):
    def __call__(
        self,
        uow_factory: UnitOfWorkFactory,
        domain_registry: DomainProfileRegistry,
        domain_settings: DomainSettingsPort,
        cache_invalidator: CacheInvalidatorPort | None,
    ) -> DocumentPackHooks: ...


def no_routers() -> Sequence[APIRouter]:
    return ()


@dataclass(frozen=True)
class PackWiring:
    document_hooks: DocumentHooksFactory | None = None
    routers: Callable[[], Sequence[APIRouter]] = no_routers


NEUTRAL_WIRING = PackWiring()


@dataclass(frozen=True)
class PackRegistration:
    # (module, attribute) of the Infrastructure DomainPack and of the Composition PackWiring.
    pack: tuple[str, str]
    wiring: tuple[str, str] | None = None


PACK_REGISTRATIONS: dict[str, PackRegistration] = {
    "belarus_law": PackRegistration(
        pack=("infrastructure.domain_packs.belarus_law.pack", "BELARUS_LAW_PACK"),
        wiring=("composition.domain_packs.belarus_law", "BELARUS_LAW_WIRING"),
    ),
}


def registration(key: str) -> PackRegistration:
    try:
        return PACK_REGISTRATIONS[key]
    except KeyError:
        known = ", ".join(sorted([NEUTRAL_PACK.key, *PACK_REGISTRATIONS]))
        message = f"Unknown domain pack {key!r}; known packs: {known}"
        raise ConfigurationError(message) from None


def selected_pack(key: str) -> DomainPack:
    if key == NEUTRAL_PACK.key:
        return NEUTRAL_PACK

    module_name, attribute = registration(key).pack
    pack = getattr(importlib.import_module(module_name), attribute)

    if not isinstance(pack, DomainPack) or pack.key != key:
        message = f"Domain pack {key!r} must resolve to a DomainPack with that key"
        raise ConfigurationError(message)

    return pack


def selected_wiring(key: str) -> PackWiring:
    if key == NEUTRAL_PACK.key:
        return NEUTRAL_WIRING

    target = registration(key).wiring

    if target is None:
        return NEUTRAL_WIRING

    module_name, attribute = target
    wiring = getattr(importlib.import_module(module_name), attribute)

    if not isinstance(wiring, PackWiring):
        message = f"Domain pack {key!r} wiring must be a PackWiring"
        raise ConfigurationError(message)

    return wiring


@dataclass(frozen=True)
class LoadedPack:
    pack: DomainPack
    wiring: PackWiring


def loaded_packs() -> dict[str, LoadedPack]:
    """Load every registered pack, with the neutral pack first."""
    ordered = dict.fromkeys((NEUTRAL_PACK.key, *PACK_REGISTRATIONS))
    return {key: LoadedPack(selected_pack(key), selected_wiring(key)) for key in ordered}


def pack_routers(packs: Mapping[str, LoadedPack]) -> list[APIRouter]:
    return [router for loaded in packs.values() for router in loaded.wiring.routers()]
