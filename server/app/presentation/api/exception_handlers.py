"""Unified exception handlers for FastAPI — KinTree-style with Client/Server split.

Principles:
- Single mapping: exception_type → status_code, not scattered across functions.
- ClientException handler for 4xx, ServerException handler for 5xx.
- HTTPException and RequestValidationError handled separately.
- Unexpected errors are logged with exc_info and return 500.
"""

from __future__ import annotations

from logging import Logger, getLogger

from domain.exceptions import (
    AuthenticationError,
    BusinessRuleViolation,
    ClientException,
    DatabaseError,
    EntityNotFound,
    PermissionDeniedError,
    ServerException,
    ValidationError,
)
from fastapi import status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

logger: Logger = getLogger("default")

# ---------------------------------------------------------------------------
# Status code mapping — client exceptions (4xx)
# ---------------------------------------------------------------------------

_CLIENT_STATUS_MAP: dict[type[ClientException], int] = {
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    EntityNotFound: status.HTTP_404_NOT_FOUND,
    BusinessRuleViolation: status.HTTP_409_CONFLICT,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    PermissionDeniedError: status.HTTP_403_FORBIDDEN,
    ClientException: status.HTTP_400_BAD_REQUEST,  # fallback
}

# ---------------------------------------------------------------------------
# Status code mapping — server exceptions (5xx)
# ---------------------------------------------------------------------------

_SERVER_STATUS_MAP: dict[type[ServerException], int] = {
    DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ServerException: status.HTTP_500_INTERNAL_SERVER_ERROR,  # fallback
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json(data: dict, status_code: int) -> JSONResponse:
    return JSONResponse(content=data, status_code=status_code)


def _error(message: str, errors: dict | None = None) -> dict:
    result: dict = {"message": message}
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# Client exceptions — single handler for all ClientException subclasses
# ---------------------------------------------------------------------------


async def handle_client_exception(request: Request, exc: Exception) -> JSONResponse:
    """4xx — client errors (validation, not found, auth, permissions)."""
    assert isinstance(exc, ClientException)

    status_code = status.HTTP_400_BAD_REQUEST
    for exc_type, code in _CLIENT_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            status_code = code
            break

    return _json(exc.as_dict(), status_code)


# ---------------------------------------------------------------------------
# Server exceptions — single handler for all ServerException subclasses
# ---------------------------------------------------------------------------


async def handle_server_exception(request: Request, exc: Exception) -> JSONResponse:
    """5xx — server errors (database, infrastructure)."""
    assert isinstance(exc, ServerException)

    logger.error("Server exception: %s", exc, exc_info=True)

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    for exc_type, code in _SERVER_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            status_code = code
            break

    return _json(exc.as_dict(), status_code)


# ---------------------------------------------------------------------------
# HTTP / Pydantic / Fallback
# ---------------------------------------------------------------------------


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle standard FastAPI HTTPException."""
    assert isinstance(exc, HTTPException)

    _AUTH_MESSAGES = {
        "Not authenticated": "Не авторизован",
        "Invalid authentication credentials": "Неверные учетные данные",
    }

    if isinstance(exc.detail, dict):
        content = _error(str(exc.detail))
    else:
        content = _error(_AUTH_MESSAGES.get(exc.detail, exc.detail))

    return _json(content, exc.status_code)


async def handle_validation_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic RequestValidationError (422)."""
    assert isinstance(exc, RequestValidationError)

    _MESSAGES = {
        "missing": "Обязательное поле",
        "extra_forbidden": "Запрещено добавлять лишние поля",
        "int_parsing": "Должно быть целым числом",
        "float_parsing": "Должно быть числом (дробь разрешена)",
        "bool_parsing": "Должно быть True или False",
        "string_type": "Должно быть строкой",
        "greater_than": "Должно быть больше {gt}",
        "less_than": "Должно быть меньше {lt}",
        "multiple_of": "Должно быть кратно {multiple_of}",
        "string_too_short": "Минимум {min_length} символов",
        "string_too_long": "Максимум {max_length} символов",
        "string_pattern_mismatch": "Некорректный формат (формат {pattern})",
        "enum": "Допустимые значения: {expected}",
        "literal_error": "Допустимо только: {expected}",
        "date_parsing": "Некорректная дата (формат: ГГГГ-ММ-ДД)",
        "time_parsing": "Некорректное время (формат: ЧЧ:ММ:СС)",
        "datetime_parsing": "Некорректная дата и время",
        "value_error": "Некорректное значение: {error}",
    }

    errors: dict = {}
    for error in exc.errors():
        if "loc" not in error or not error["loc"]:
            continue
        field = str(error["loc"][-1])
        tmpl = _MESSAGES.get(error.get("type", ""), error.get("msg", "Ошибка"))
        try:
            errors[field] = tmpl.format(**error.get("ctx", {}))
        except (KeyError, IndexError):
            errors[field] = error.get("msg", "Ошибка")

    content = _error("Ошибка валидации", errors) if errors else _error("Ошибка валидации")
    return _json(content, status.HTTP_422_UNPROCESSABLE_ENTITY)


async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    """Fallback for unhandled exceptions."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return _json(_error("Internal Server Error"), status.HTTP_500_INTERNAL_SERVER_ERROR)
