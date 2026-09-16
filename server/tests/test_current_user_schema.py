"""Smoke tests for CurrentUser and MeResponse schemas — FINDING-029."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from pydantic import ValidationError

from presentation.api.schemas.auth import CurrentUser, MeResponse


class TestCurrentUser:
    """CurrentUser schema construction and serialization."""

    def test_construction_minimal(self):
        user = CurrentUser(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert user.id == 1
        assert user.email == "a@b.com"
        assert user.api_key_id is None

    def test_construction_with_api_key_id(self):
        user = CurrentUser(
            id=1, email="a@b.com", role="admin", kind="internal", is_active=True, api_key_id=42
        )
        assert user.api_key_id == 42

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            CurrentUser(id=1, email="a@b.com", role="user", kind="internal", is_active=True, foo="bar")

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            CurrentUser(id=1, email="a@b.com")

    def test_api_key_id_optional(self):
        user = CurrentUser(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert user.api_key_id is None

    def test_model_dump_excludes_api_key_id_in_me_response(self):
        """MeResponse does not have api_key_id — explicit safety."""
        user = CurrentUser(
            id=1, email="a@b.com", role="admin", kind="internal", is_active=True, api_key_id=99
        )
        me = MeResponse.model_validate(user.model_dump(exclude={"api_key_id"}))
        dumped = me.model_dump()
        assert "api_key_id" not in dumped
        assert dumped["id"] == 1
        assert dumped["email"] == "a@b.com"


class TestMeResponse:
    """MeResponse schema — safe subset of CurrentUser."""

    def test_construction(self):
        me = MeResponse(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert me.id == 1

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            MeResponse(id=1, email="a@b.com", role="user", kind="internal", is_active=True, api_key_id=1)

    def test_serialization_no_api_key(self):
        me = MeResponse(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert "api_key_id" not in me.model_dump()
