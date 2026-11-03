"""Construction and serialization contracts for CurrentUser and MeResponse."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from presentation.api.schemas.auth import CurrentUser, MeResponse


class TestCurrentUser:
    """CurrentUser schema construction and serialization."""

    def test_construction_minimal(self) -> None:
        user = CurrentUser(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert user.id == 1
        assert user.email == "a@b.com"
        assert user.api_key_id is None

    def test_construction_with_api_key_id(self) -> None:
        user = CurrentUser(
            id=1, email="a@b.com", role="admin", kind="internal", is_active=True, api_key_id=42
        )
        assert user.api_key_id == 42

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CurrentUser.model_validate(
                {
                    "id": 1,
                    "email": "a@b.com",
                    "role": "user",
                    "kind": "internal",
                    "is_active": True,
                    "foo": "bar",
                }
            )

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            CurrentUser.model_validate({"id": 1, "email": "a@b.com"})

    def test_model_dump_excludes_api_key_id_in_me_response(self) -> None:
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

    def test_construction(self) -> None:
        me = MeResponse(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert me.id == 1

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            MeResponse.model_validate(
                {
                    "id": 1,
                    "email": "a@b.com",
                    "role": "user",
                    "kind": "internal",
                    "is_active": True,
                    "api_key_id": 1,
                }
            )

    def test_serialization_no_api_key(self) -> None:
        me = MeResponse(id=1, email="a@b.com", role="user", kind="internal", is_active=True)
        assert "api_key_id" not in me.model_dump()
