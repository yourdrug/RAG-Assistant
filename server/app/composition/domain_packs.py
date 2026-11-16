"""Explicit registry of domain packs; enabling a pack is one entry in DOMAIN_PACKS.

A pack other than the neutral one is imported only when DOMAIN_PACK selects it, so the
core (a neutral deployment, the core test suite) runs without the pack's code.
"""

from __future__ import annotations

import importlib

from application.runtime.errors import ConfigurationError
from infrastructure.domain_profile.packs import NEUTRAL_PACK, DomainPack


# Pack key -> (module, attribute) holding its DomainPack definition.
DOMAIN_PACKS: dict[str, tuple[str, str]] = {
    "belarus_law": ("infrastructure.domain_packs.belarus_law.pack", "BELARUS_LAW_PACK"),
}


def selected_pack(key: str) -> DomainPack:
    if key == NEUTRAL_PACK.key:
        return NEUTRAL_PACK

    try:
        module_name, attribute = DOMAIN_PACKS[key]
    except KeyError:
        known = ", ".join(sorted([NEUTRAL_PACK.key, *DOMAIN_PACKS]))
        message = f"Unknown DOMAIN_PACK {key!r}; known packs: {known}"
        raise ConfigurationError(message) from None

    pack = getattr(importlib.import_module(module_name), attribute)

    if not isinstance(pack, DomainPack) or pack.key != key:
        message = f"DOMAIN_PACK {key!r} must resolve to a DomainPack with that key"
        raise ConfigurationError(message)

    return pack
