"""Password hasher protocol -- abstracts password hashing and verification for the domain layer.

The concrete implementation lives in ``infrastructure.auth.password_hasher``;
the domain layer depends only on this protocol.
"""

from __future__ import annotations

from typing import Protocol


class IPasswordHasher(Protocol):
    async def hash(self, password: str) -> str: ...
    async def verify(self, password: str, hashed: str) -> bool: ...
