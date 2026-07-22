"""
Tests for auth use cases: CreateUser, AuthenticateUser, ToggleUserActive, ListUsers.
All dependencies are mocked — no real DB, no real passwords, no real JWT.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from application.dto.auth_dto import CreateUserCommand, LoginCommand
from application.use_cases.auth.authenticate_user import AuthenticateUser
from application.use_cases.auth.create_user import CreateUser
from application.use_cases.auth.list_users import ListUsers
from application.use_cases.auth.toggle_user_active import ToggleUserActive
from domain.entities.user import User
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.value_objects.roles import UserKind, UserRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(
    id=1, email="test@example.com", role=UserRole.USER, kind=UserKind.INTERNAL, is_active=True
):
    return User(id=id, email=email, role=role, kind=kind, is_active=is_active)


# ---------------------------------------------------------------------------
# CreateUser
# ---------------------------------------------------------------------------


class TestCreateUser:
    def setup_method(self):
        self.user_repo = MagicMock()
        self.password_hasher = MagicMock()
        self.use_case = CreateUser(self.user_repo, self.password_hasher)

    def test_successful_creation(self):
        self.user_repo.get_by_email.return_value = None
        saved_user = _make_user(id=10, email="new@test.com")
        self.user_repo.save.return_value = saved_user
        self.password_hasher.hash.return_value = "hashed_pw"

        result = self.use_case.execute(
            CreateUserCommand(email="new@test.com", password="secret", role="user", kind="internal"),
            creator_role="admin",
        )

        assert result.id == 10
        assert result.email == "new@test.com"
        self.password_hasher.hash.assert_called_once_with("secret")
        self.user_repo.save.assert_called_once()

    def test_non_admin_creator_raises(self):
        with pytest.raises(BusinessRuleViolation) as exc_info:
            self.use_case.execute(
                CreateUserCommand(email="x@x.com", password="pw"),
                creator_role="user",
            )
        assert "Only admin" in str(exc_info.value)

    def test_duplicate_email_raises(self):
        existing = _make_user(email="dup@test.com")
        self.user_repo.get_by_email.return_value = existing

        with pytest.raises(BusinessRuleViolation) as exc_info:
            self.use_case.execute(
                CreateUserCommand(email="dup@test.com", password="pw"),
                creator_role="admin",
            )
        assert "already exists" in str(exc_info.value)

    def test_client_cannot_be_admin(self):
        with pytest.raises(BusinessRuleViolation) as exc_info:
            self.use_case.execute(
                CreateUserCommand(email="c@x.com", password="pw", role="admin", kind="client"),
                creator_role="admin",
            )
        assert "Client cannot be admin" in str(exc_info.value)

    def test_password_is_hashed_before_save(self):
        self.user_repo.get_by_email.return_value = None
        self.user_repo.save.return_value = _make_user(id=1, email="a@b.com")
        self.password_hasher.hash.return_value = "hashed"

        self.use_case.execute(
            CreateUserCommand(email="a@b.com", password="raw_pw"),
            creator_role="admin",
        )

        # Verify hash was called and result was set
        self.password_hasher.hash.assert_called_once_with("raw_pw")

    def test_invalid_role_string_raises(self):
        with pytest.raises(ValidationError):
            self.use_case.execute(
                CreateUserCommand(email="x@x.com", password="pw", role="superadmin"),
                creator_role="admin",
            )


# ---------------------------------------------------------------------------
# AuthenticateUser
# ---------------------------------------------------------------------------


class TestAuthenticateUser:
    def setup_method(self):
        self.user_repo = MagicMock()
        self.password_verifier = MagicMock()
        self.token_provider = MagicMock()
        self.use_case = AuthenticateUser(self.user_repo, self.password_verifier, self.token_provider)

    def test_successful_authentication(self):
        user = _make_user(id=42, role=UserRole.USER, kind=UserKind.INTERNAL)
        self.user_repo.get_by_email.return_value = user
        self.password_verifier.verify.return_value = True
        self.token_provider.create_token.return_value = "jwt-token"

        result = self.use_case.execute(LoginCommand(email="test@test.com", password="correct"))

        assert result.access_token == "jwt-token"
        assert result.role == "user"
        assert result.kind == "internal"
        self.token_provider.create_token.assert_called_once_with(user_id=42, role=UserRole.USER)

    def test_nonexistent_email_raises(self):
        self.user_repo.get_by_email.return_value = None

        with pytest.raises(ValidationError) as exc_info:
            self.use_case.execute(LoginCommand(email="nobody@test.com", password="pw"))
        assert "Invalid email or password" in str(exc_info.value)

    def test_inactive_user_raises(self):
        user = _make_user(is_active=False)
        self.user_repo.get_by_email.return_value = user

        with pytest.raises(ValidationError) as exc_info:
            self.use_case.execute(LoginCommand(email="test@test.com", password="pw"))
        assert "Invalid email or password" in str(exc_info.value)

    def test_wrong_password_raises(self):
        user = _make_user()
        self.user_repo.get_by_email.return_value = user
        self.password_verifier.verify.return_value = False

        with pytest.raises(ValidationError) as exc_info:
            self.use_case.execute(LoginCommand(email="test@test.com", password="wrong"))
        assert "Invalid email or password" in str(exc_info.value)

    def test_same_error_message_for_nonexistent_and_wrong_password(self):
        """Security: don't reveal whether email exists."""
        self.user_repo.get_by_email.return_value = None
        with pytest.raises(ValidationError) as exc_info:
            self.use_case.execute(LoginCommand(email="x@x.com", password="pw"))
        msg1 = str(exc_info.value)

        user = _make_user()
        self.user_repo.get_by_email.return_value = user
        self.password_verifier.verify.return_value = False
        with pytest.raises(ValidationError) as exc_info:
            self.use_case.execute(LoginCommand(email="x@x.com", password="pw"))
        msg2 = str(exc_info.value)

        assert msg1 == msg2


# ---------------------------------------------------------------------------
# ToggleUserActive
# ---------------------------------------------------------------------------


class TestToggleUserActive:
    def setup_method(self):
        self.user_repo = MagicMock()
        self.use_case = ToggleUserActive(self.user_repo)

    def test_deactivate_user(self):
        user = _make_user(id=5)
        self.user_repo.get_by_id.return_value = user

        result = self.use_case.execute(user_id=5, is_active=False, admin_id=1)

        assert result == {"id": 5, "is_active": False}
        self.user_repo.set_active.assert_called_once_with(5, False)

    def test_activate_user(self):
        user = _make_user(id=5, is_active=False)
        self.user_repo.get_by_id.return_value = user

        result = self.use_case.execute(user_id=5, is_active=True, admin_id=1)

        assert result == {"id": 5, "is_active": True}
        self.user_repo.set_active.assert_called_once_with(5, True)

    def test_nonexistent_user_raises(self):
        self.user_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFound) as exc_info:
            self.use_case.execute(user_id=999, is_active=False, admin_id=1)
        assert "User" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_admin_cannot_deactivate_self(self):
        user = _make_user(id=1)
        self.user_repo.get_by_id.return_value = user

        with pytest.raises(BusinessRuleViolation) as exc_info:
            self.use_case.execute(user_id=1, is_active=False, admin_id=1)
        assert "Cannot deactivate yourself" in str(exc_info.value)

    def test_admin_can_deactivate_other(self):
        user = _make_user(id=5)
        self.user_repo.get_by_id.return_value = user

        result = self.use_case.execute(user_id=5, is_active=False, admin_id=1)
        assert result["is_active"] is False


# ---------------------------------------------------------------------------
# ListUsers
# ---------------------------------------------------------------------------


class TestListUsers:
    def setup_method(self):
        self.user_repo = MagicMock()
        self.use_case = ListUsers(self.user_repo)

    def test_list_all_users(self):
        users = [
            _make_user(id=1, email="a@test.com"),
            _make_user(id=2, email="b@test.com", role=UserRole.ADMIN),
        ]
        self.user_repo.list_all.return_value = users

        result = self.use_case.execute()

        assert len(result) == 2
        assert result[0].email == "a@test.com"
        assert result[1].role == "admin"

    def test_empty_list(self):
        self.user_repo.list_all.return_value = []

        result = self.use_case.execute()
        assert result == []

    def test_single_user(self):
        self.user_repo.list_all.return_value = [_make_user(id=1, email="only@test.com")]

        result = self.use_case.execute()
        assert len(result) == 1
        assert result[0].email == "only@test.com"
