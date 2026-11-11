"""Token provider protocol -- abstracts JWT creation and decoding for the domain layer.

The concrete implementation lives in ``infrastructure.auth.jwt_provider``;
the domain layer depends only on this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """Verified claims of an access token."""

    user_id: int
    role: str
    expires_at: datetime


class TokenProvider(Protocol):
    def create_token(self, user_id: int, role: str) -> str: ...
    def decode_token(self, token: str) -> TokenClaims:
        """Verify a token and return its claims; raise on any invalid or incomplete token."""
        ...


class PasswordHasher(Protocol):
    async def hash(self, password: str) -> str: ...
    async def verify(self, password: str, hashed: str) -> bool: ...
