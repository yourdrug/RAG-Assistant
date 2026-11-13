"""Explicit registry of domain packs; enabling a pack is one entry in DOMAIN_PACKS."""

from __future__ import annotations

from application.runtime.errors import ConfigurationError
from infrastructure.domain_profile.packs import NEUTRAL_PACK, DomainPack
from packs.belarus_law.pack import BELARUS_LAW_PACK


DOMAIN_PACKS: dict[str, DomainPack] = {pack.key: pack for pack in (NEUTRAL_PACK, BELARUS_LAW_PACK)}


def selected_pack(key: str) -> DomainPack:
    try:
        return DOMAIN_PACKS[key]
    except KeyError:
        known = ", ".join(sorted(DOMAIN_PACKS))
        message = f"Unknown DOMAIN_PACK {key!r}; known packs: {known}"
        raise ConfigurationError(message) from None
