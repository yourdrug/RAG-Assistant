"""Explicit registry of domain packs; enabling a pack is one entry in DOMAIN_PACKS.

A pack other than the neutral one is imported only when DOMAIN_PACK selects it, so the
core (a neutral deployment, the core test suite) runs without the pack's code. A pack has
an Infrastructure definition (profiles, evidence rules, repositories) and an optional
Composition wiring of its scenarios: document lifecycle hooks and API routes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Protocol

from application.runtime.errors import ConfigurationError
from infrastructure.domain_profile.packs import NEUTRAL_PACK, DomainPack


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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


DOMAIN_PACKS: dict[str, PackRegistration] = {
    "belarus_law": PackRegistration(
        pack=("infrastructure.domain_packs.belarus_law.pack", "BELARUS_LAW_PACK"),
        wiring=("composition.domain_packs.belarus_law", "BELARUS_LAW_WIRING"),
    ),
}


def registration(key: str) -> PackRegistration:
    try:
        return DOMAIN_PACKS[key]
    except KeyError:
        known = ", ".join(sorted([NEUTRAL_PACK.key, *DOMAIN_PACKS]))
        message = f"Unknown DOMAIN_PACK {key!r}; known packs: {known}"
        raise ConfigurationError(message) from None


def selected_pack(key: str) -> DomainPack:
    if key == NEUTRAL_PACK.key:
        return NEUTRAL_PACK

    module_name, attribute = registration(key).pack
    pack = getattr(importlib.import_module(module_name), attribute)

    if not isinstance(pack, DomainPack) or pack.key != key:
        message = f"DOMAIN_PACK {key!r} must resolve to a DomainPack with that key"
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
        message = f"DOMAIN_PACK {key!r} wiring must be a PackWiring"
        raise ConfigurationError(message)

    return wiring
