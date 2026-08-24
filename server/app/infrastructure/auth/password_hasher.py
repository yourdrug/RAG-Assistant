"""BCrypt password hasher -- hashes and verifies user passwords.

Uses ``bcrypt`` with automatic salt generation.  The ``hash`` method returns
a Base64-encoded string; ``verify`` performs a constant-time comparison.

All CPU-bound bcrypt operations are offloaded to a thread pool via
``asyncio.to_thread`` so they never block the async event loop.
"""

from __future__ import annotations

import asyncio

import bcrypt


class BCryptPasswordHasher:
    async def hash(self, password: str) -> str:
        return await asyncio.to_thread(
            lambda: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        )

    async def verify(self, password: str, hashed: str) -> bool:
        def _check() -> bool:
            try:
                return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
            except ValueError:
                return False

        return await asyncio.to_thread(_check)
