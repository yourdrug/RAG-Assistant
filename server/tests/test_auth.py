"""Tests for infrastructure/auth — password hashing and JWT tokens.

Uses real bcrypt/jwt (deterministic, no external services needed).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

from pydantic import SecretStr


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import jwt as _jwt
import pytest

from config import settings
from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.auth.password_hasher import BCryptPasswordHasher


if TYPE_CHECKING:
    from domain.identity.ports import TokenClaims


_hasher = BCryptPasswordHasher()
_token_provider = JWTProvider(settings)


async def hash_password(password: str) -> str:
    return await _hasher.hash(password)


async def verify_password(password: str, hashed: str) -> bool:
    return await _hasher.verify(password, hashed)


def create_access_token(user_id: int, role: str) -> str:
    return _token_provider.create_token(user_id, role)


def decode_access_token(token: str) -> TokenClaims:
    return _token_provider.decode_token(token)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    @pytest.mark.asyncio
    async def test_hash_produces_bcrypt_string(self) -> None:
        h = await hash_password("mypassword")
        assert h.startswith("$2")

    @pytest.mark.asyncio
    async def test_hash_is_deterministic_for_same_input_different_salts(self) -> None:
        h1 = await hash_password("test")
        h2 = await hash_password("test")
        # Different salts -> different hashes, but both valid
        assert h1 != h2
        assert await verify_password("test", h1)
        assert await verify_password("test", h2)

    @pytest.mark.asyncio
    async def test_verify_correct_password(self) -> None:
        h = await hash_password("secret123")
        assert await verify_password("secret123", h) is True

    @pytest.mark.asyncio
    async def test_verify_wrong_password(self) -> None:
        h = await hash_password("secret123")
        assert await verify_password("wrong", h) is False

    @pytest.mark.asyncio
    async def test_verify_empty_password_against_hash(self) -> None:
        h = await hash_password("notempty")
        assert await verify_password("", h) is False

    @pytest.mark.asyncio
    async def test_verify_with_invalid_hash_returns_false(self) -> None:
        assert await verify_password("password", "not-a-valid-hash") is False

    @pytest.mark.asyncio
    async def test_verify_with_empty_hash_returns_false(self) -> None:
        assert await verify_password("password", "") is False

    @pytest.mark.asyncio
    async def test_unicode_password(self) -> None:
        h = await hash_password("Пароль123!@#")
        assert await verify_password("Пароль123!@#", h) is True

    @pytest.mark.asyncio
    async def test_long_password(self) -> None:
        long_pw = "a" * 1000
        h = await hash_password(long_pw)
        assert await verify_password(long_pw, h) is True


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


class TestJWT:
    def test_create_and_decode_token(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret-key-for-testing")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            patch.object(settings, "jwt_expire_minutes", 60),
        ):
            token = create_access_token(user_id=42, role="admin")
            claims = decode_access_token(token)
            assert claims.user_id == 42
            assert claims.role == "admin"

    def test_token_contains_exp(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            patch.object(settings, "jwt_expire_minutes", 60),
        ):
            token = create_access_token(user_id=1, role="user")
            claims = decode_access_token(token)
            assert claims.expires_at > datetime.now(UTC)

    def test_decode_with_wrong_secret_raises(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("correct-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            patch.object(settings, "jwt_expire_minutes", 60),
        ):
            token = create_access_token(user_id=1, role="user")

        with (
            patch.object(settings, "jwt_secret_key", SecretStr("different-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            pytest.raises(_jwt.exceptions.InvalidSignatureError),
        ):
            decode_access_token(token)

    def test_decode_garbage_token_raises(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            pytest.raises((_jwt.exceptions.DecodeError, _jwt.exceptions.InvalidTokenError)),
        ):
            decode_access_token("not.a.jwt")

    def test_decode_empty_string_raises(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            pytest.raises((_jwt.exceptions.DecodeError, _jwt.exceptions.InvalidTokenError)),
        ):
            decode_access_token("")

    def test_token_with_different_role(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            patch.object(settings, "jwt_expire_minutes", 60),
        ):
            token = create_access_token(user_id=1, role="superadmin")
            claims = decode_access_token(token)
            assert claims.role == "superadmin"

    def test_different_users_different_tokens(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            patch.object(settings, "jwt_expire_minutes", 60),
        ):
            t1 = create_access_token(user_id=1, role="user")
            t2 = create_access_token(user_id=2, role="user")
            p1 = decode_access_token(t1)
            p2 = decode_access_token(t2)
            assert p1.user_id != p2.user_id

    def test_expired_token_raises(self) -> None:
        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            patch.object(settings, "jwt_expire_minutes", -1),
        ):
            token = create_access_token(user_id=1, role="user")

        with (
            patch.object(settings, "jwt_secret_key", SecretStr("test-secret")),
            patch.object(settings, "jwt_algorithm", "HS256"),
            pytest.raises(_jwt.exceptions.ExpiredSignatureError),
        ):
            decode_access_token(token)


def signed(claims: dict[str, object]) -> str:
    return _jwt.encode(claims, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)


def valid_claims() -> dict[str, object]:
    return {"sub": "7", "role": "user", "exp": datetime.now(UTC) + timedelta(minutes=5)}


class TestJWTClaimsValidation:
    def test_complete_claims_are_accepted(self) -> None:
        claims = decode_access_token(signed(valid_claims()))
        assert (claims.user_id, claims.role) == (7, "user")

    @pytest.mark.parametrize("missing", ["sub", "role", "exp"])
    def test_missing_required_claim_is_rejected(self, missing: str) -> None:
        claims = valid_claims()
        del claims[missing]
        with pytest.raises(_jwt.InvalidTokenError):
            decode_access_token(signed(claims))

    @pytest.mark.parametrize(
        "override",
        [{"sub": "abc"}, {"sub": "-1"}, {"sub": ""}, {"role": ""}, {"role": 5}],
    )
    def test_malformed_claim_is_rejected(self, override: dict[str, object]) -> None:
        with pytest.raises(_jwt.InvalidTokenError):
            decode_access_token(signed(valid_claims() | override))

    @pytest.mark.parametrize("exp", [253402300800, 10**20])
    def test_out_of_range_expiry_is_rejected(self, exp: int) -> None:
        """A validly signed token whose exp overflows datetime must yield 401, not 500."""
        with pytest.raises(_jwt.InvalidTokenError, match="out of range"):
            decode_access_token(signed(valid_claims() | {"exp": exp}))
