"""
Tests for infrastructure/auth.py — password hashing and JWT tokens.
Uses real bcrypt/jwt (deterministic, no external services needed).
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from config import settings  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from infrastructure.auth import (  # noqa: E402
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_produces_bcrypt_string(self):
        h = hash_password("mypassword")
        assert h.startswith("$2")

    def test_hash_is_deterministic_for_same_input_different_salts(self):
        h1 = hash_password("test")
        h2 = hash_password("test")
        # Different salts -> different hashes, but both valid
        assert h1 != h2
        assert verify_password("test", h1)
        assert verify_password("test", h2)

    def test_verify_correct_password(self):
        h = hash_password("secret123")
        assert verify_password("secret123", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("secret123")
        assert verify_password("wrong", h) is False

    def test_verify_empty_password_against_hash(self):
        h = hash_password("notempty")
        assert verify_password("", h) is False

    def test_verify_with_invalid_hash_returns_false(self):
        assert verify_password("password", "not-a-valid-hash") is False

    def test_verify_with_empty_hash_returns_false(self):
        assert verify_password("password", "") is False

    def test_unicode_password(self):
        h = hash_password("Пароль123!@#")
        assert verify_password("Пароль123!@#", h) is True

    def test_long_password(self):
        long_pw = "a" * 1000
        h = hash_password(long_pw)
        assert verify_password(long_pw, h) is True


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


class TestJWT:
    def test_create_and_decode_token(self):
        with patch.object(settings, "jwt_secret_key", "test-secret-key-for-testing"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with patch.object(settings, "jwt_expire_minutes", 60):
                    token = create_access_token(user_id=42, role="admin")
                    payload = decode_access_token(token)
                    assert payload["sub"] == "42"
                    assert payload["role"] == "admin"

    def test_token_contains_exp(self):
        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with patch.object(settings, "jwt_expire_minutes", 60):
                    token = create_access_token(user_id=1, role="user")
                    payload = decode_access_token(token)
                    assert "exp" in payload

    def test_decode_with_wrong_secret_raises(self):
        with patch.object(settings, "jwt_secret_key", "correct-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with patch.object(settings, "jwt_expire_minutes", 60):
                    token = create_access_token(user_id=1, role="user")

        with patch.object(settings, "jwt_secret_key", "different-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with pytest.raises(HTTPException) as exc_info:
                    decode_access_token(token)
                assert exc_info.value.status_code == 401

    def test_decode_garbage_token_raises(self):
        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with pytest.raises(HTTPException) as exc_info:
                    decode_access_token("not.a.jwt")
                assert exc_info.value.status_code == 401

    def test_decode_empty_string_raises(self):
        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with pytest.raises(HTTPException) as exc_info:
                    decode_access_token("")
                assert exc_info.value.status_code == 401

    def test_token_with_different_role(self):
        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with patch.object(settings, "jwt_expire_minutes", 60):
                    token = create_access_token(user_id=1, role="superadmin")
                    payload = decode_access_token(token)
                    assert payload["role"] == "superadmin"

    def test_different_users_different_tokens(self):
        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with patch.object(settings, "jwt_expire_minutes", 60):
                    t1 = create_access_token(user_id=1, role="user")
                    t2 = create_access_token(user_id=2, role="user")
                    p1 = decode_access_token(t1)
                    p2 = decode_access_token(t2)
                    assert p1["sub"] != p2["sub"]

    def test_expired_token_raises(self):
        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with patch.object(settings, "jwt_expire_minutes", -1):
                    token = create_access_token(user_id=1, role="user")

        with patch.object(settings, "jwt_secret_key", "test-secret"):
            with patch.object(settings, "jwt_algorithm", "HS256"):
                with pytest.raises(HTTPException) as exc_info:
                    decode_access_token(token)
                assert exc_info.value.status_code == 401
                assert "expired" in exc_info.value.detail.lower()
