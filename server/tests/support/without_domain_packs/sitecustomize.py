"""Make every domain pack unimportable, so ``task test:core`` proves the core runs without one.

Loaded by Python at startup when this directory is on PYTHONPATH.
"""

import importlib.abc
import sys


PACKS = "infrastructure.domain_packs"


class BlockDomainPacks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:  # noqa: ARG002 - MetaPathFinder signature
        if fullname == PACKS or fullname.startswith(f"{PACKS}."):
            message = f"domain packs are blocked in the core run: {fullname}"
            raise ModuleNotFoundError(message, name=fullname)


sys.meta_path.insert(0, BlockDomainPacks())
