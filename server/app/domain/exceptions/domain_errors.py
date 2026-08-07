"""Domain exception hierarchy — KinTree-style with as_dict() and Client/Server split.

All exceptions carry message + errors and expose as_dict() for uniform JSON serialization.
- ClientException  → 4xx (client errors)
- ServerException  → 5xx (server errors)
"""

from __future__ import annotations

from typing import Any

# ── Base ─────────────────────────────────────────────────────────────────────


class AppException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, errors: dict[str, Any] | None = None) -> None:
        self.message = message
        self.errors = errors
        super().__init__(message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, errors={self.errors!r})"

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"message": self.message}
        if self.errors:
            result["errors"] = self.errors
        return result


# ── Server ────────────────────────────────────────────────────────────────────


class ServerException(AppException):
    """Server-side error (5xx)."""


class DatabaseError(ServerException):
    """Database operation failure."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            message="Ошибка взаимодействия с БД",
            errors={"detail": detail} if detail else None,
        )


# ── Client ────────────────────────────────────────────────────────────────────


class ClientException(AppException):
    """Client-side error (4xx)."""


class ValidationError(ClientException):
    """Business rule or invariant violated (422)."""

    def __init__(
        self,
        message: str = "Ошибка валидации",
        errors: dict[str, Any] | None = None,
        *,
        field: str | None = None,
    ) -> None:
        if field is not None and errors is None:
            errors = {field: message}
            message = "Ошибка валидации"
        super().__init__(message=message, errors=errors)


class EntityNotFound(ClientException):
    """Requested entity not found (404)."""

    def __init__(self, entity_name: str, identifier: str | int) -> None:
        msg = f"{entity_name} with id={identifier} not found"
        super().__init__(
            message=msg,
            errors={entity_name.lower(): msg, "id": str(identifier)},
        )


class BusinessRuleViolation(ClientException):
    """Domain operation violates a business rule (409)."""


class AuthenticationError(ClientException):
    """Authentication failed — invalid/expired token (401)."""

    def __init__(self, detail: str = "Не авторизован") -> None:
        super().__init__(
            message=detail,
            errors={"detail": detail},
        )


class PermissionDeniedError(ClientException):
    """Insufficient permissions (403)."""

    def __init__(self, required: str | list[str] | None = None) -> None:
        errors: dict[str, Any] = {}
        if isinstance(required, str):
            errors["required_permission"] = required
        elif isinstance(required, list):
            errors["required_any_of"] = required
        super().__init__(message="Недостаточно прав", errors=errors or None)
