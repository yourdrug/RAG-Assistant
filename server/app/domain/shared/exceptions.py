"""Domain exception hierarchy -- client/server split with uniform JSON serialization.

All exceptions carry a ``message`` and optional ``errors`` dict and expose
``as_dict()`` for uniform JSON serialization in API error responses.

- ``AppClientError``  -- 4xx errors (bad input, not found, forbidden)
- ``AppServerError``  -- 5xx errors (database failures, unexpected errors)
"""

from __future__ import annotations

from typing import Any


# ── Base ─────────────────────────────────────────────────────────────────────


class AppError(Exception):
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


class AppServerError(AppError):
    """Server-side error (5xx)."""


class DatabaseError(AppServerError):
    """Database operation failure."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            message="Ошибка взаимодействия \u0441 БД",
            errors={"detail": detail} if detail else None,
        )


class UniqueConstraintError(DatabaseError):
    """A unique/integrity constraint was violated in the database."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail=detail)


# ── Client ────────────────────────────────────────────────────────────────────


class AppClientError(AppError):
    """Client-side error (4xx)."""


class ValidationError(AppClientError):
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


class EntityNotFoundError(AppClientError):
    """Requested entity not found (404)."""

    def __init__(self, entity_name: str, identifier: str | int) -> None:
        msg = f"{entity_name} with id={identifier} not found"
        super().__init__(
            message=msg,
            errors={entity_name.lower(): msg, "id": str(identifier)},
        )


class BusinessRuleViolationError(AppClientError):
    """Domain operation violates a business rule (409)."""


class AuthenticationError(AppClientError):
    """Authentication failed — invalid/expired token (401)."""

    def __init__(self, detail: str = "\u041d\u0435 авторизован") -> None:
        super().__init__(
            message=detail,
            errors={"detail": detail},
        )


class PermissionDeniedError(AppClientError):
    """Insufficient permissions (403)."""

    def __init__(self, required: str | list[str] | None = None) -> None:
        errors: dict[str, Any] = {}

        if isinstance(required, str):
            errors["required_permission"] = required
        elif isinstance(required, list):
            errors["required_any_of"] = required

        super().__init__(message="Недостаточно прав", errors=errors or None)


class LLMUnavailableError(AppServerError):
    """LLM provider temporarily unavailable (circuit breaker OPEN or timeout).

    Used by infrastructure/resilience/circuit_breaker.py and caught by
    the SSE layer to present a user-friendly error message.
    """

    def __init__(self, detail: str = "LLM временно недоступен, попробуйте позже") -> None:
        super().__init__(
            message=detail,
            errors={"code": "llm_unavailable"},
        )


class SemaphoreTimeoutError(AppClientError):
    """Semaphore acquire timed out — server under load (429 Too Many Requests).

    Raised by ``TimeoutSemaphore`` when the concurrency limit is reached
    and the caller cannot acquire a permit within the configured timeout.
    """

    def __init__(self, detail: str = "Сервер перегружен, повторите позже") -> None:
        super().__init__(
            message=detail,
            errors={"code": "semaphore_timeout"},
        )


class RateLimitExceededError(AppClientError):
    """Request rate limit exceeded (429 Too Many Requests).

    Carries the retry delay and the limit of the tightest policy window so the
    exception handler can emit ``Retry-After`` / ``X-RateLimit-Limit`` headers.
    """

    def __init__(self, retry_after_sec: int, limit: int) -> None:
        super().__init__(
            message="Слишком много запросов, попробуйте позже",
            errors={"code": "rate_limit_exceeded"},
        )
        self.retry_after_sec = retry_after_sec
        self.limit = limit
