"""Password Hasher Protocol — abstracts password hashing for the domain layer."""

from __future__ import annotations

from typing import Protocol


class IPasswordHasher(Protocol):
    def hash(self, password: str) -> str: ...
    def verify(self, password: str, hashed: str) -> bool: ...
