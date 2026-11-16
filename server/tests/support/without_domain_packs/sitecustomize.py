"""Make every domain pack unimportable, so ``task test:core`` proves the core runs without one.

Loaded by Python at startup when this directory is on PYTHONPATH.
"""

import importlib.abc
import sys


# Pack code in every layer; ``composition.domain_packs`` itself is the core registry.
PACKS = (
    "domain.domain_packs",
    "application.domain_packs",
    "presentation.domain_packs",
    "infrastructure.domain_packs",
)
PACK_WIRING = "composition.domain_packs."


def is_pack_module(fullname: str) -> bool:
    return fullname.startswith(PACK_WIRING) or any(
        fullname == pack or fullname.startswith(f"{pack}.") for pack in PACKS
    )


class BlockDomainPacks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:  # noqa: ARG002 - MetaPathFinder signature
        if is_pack_module(fullname):
            message = f"domain packs are blocked in the core run: {fullname}"
            raise ModuleNotFoundError(message, name=fullname)


sys.meta_path.insert(0, BlockDomainPacks())
