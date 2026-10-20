"""Token provider protocol -- abstracts JWT creation and decoding for the domain layer.

The concrete implementation lives in ``infrastructure.auth.jwt_provider``;
the domain layer depends only on this protocol.
"""

from __future__ import annotations

from typing import Protocol


class TokenProvider(Protocol):
    def create_token(self, user_id: int, role: str) -> str: ...
    def decode_token(self, token: str) -> dict: ...


class PasswordHasher(Protocol):
    async def hash(self, password: str) -> str: ...
    async def verify(self, password: str, hashed: str) -> bool: ...
