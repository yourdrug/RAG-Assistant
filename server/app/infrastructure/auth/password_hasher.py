"""BCrypt password hasher -- hashes and verifies user passwords.

Uses ``bcrypt`` with automatic salt generation.  The ``hash`` method returns
a Base64-encoded string; ``verify`` performs a constant-time comparison.
"""

from __future__ import annotations

import bcrypt


class BCryptPasswordHasher:
    def hash(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify(self, password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except ValueError:
            return False
