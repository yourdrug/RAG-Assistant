"""
presentation/api/exception_handlers.py

Unified exception handlers for FastAPI.

Principles:
- Single mapping: exception_type → status_code, not scattered across functions.
- Domain exceptions use a single handler — each maps to its HTTP status.
- HTTPException and RequestValidationError handled separately.
- Unexpected errors are logged with exc_info and return 500.
"""

from __future__ import annotations

from logging import Logger, getLogger

from domain.exceptions import (
    BusinessRuleViolation,
    DatabaseError,
    DomainError,
    EntityNotFound,
    ValidationError,
)
from fastapi import status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

logger: Logger = getLogger("default")

# ---------------------------------------------------------------------------
# Status code mapping for domain exceptions
# ---------------------------------------------------------------------------

_DOMAIN_STATUS_MAP: dict[type[DomainError], int] = {
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    EntityNotFound: status.HTTP_404_NOT_FOUND,
    BusinessRuleViolation: status.HTTP_409_CONFLICT,
    DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
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
# Domain exceptions — single handler for all DomainError subclasses
# ---------------------------------------------------------------------------


async def handle_domain_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle all domain exceptions (ValidationError, EntityNotFound, etc.)."""
    if not isinstance(exc, DomainError):
        logger.error("Unexpected non-domain exception in domain handler", exc_info=exc)
        return _json(_error("Internal Server Error"), status.HTTP_500_INTERNAL_SERVER_ERROR)

    status_code = _DOMAIN_STATUS_MAP.get(type(exc), status.HTTP_400_BAD_REQUEST)

    if status_code >= 500:
        logger.error("Domain error: %s", exc, exc_info=True)

    return _json(_error(str(exc)), status_code)


# ---------------------------------------------------------------------------
# HTTP / Pydantic / Fallback
# ---------------------------------------------------------------------------


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle standard FastAPI HTTPException."""
    if not isinstance(exc, HTTPException):
        logger.critical("Unexpected exception in HTTP handler", exc_info=exc)
        return _json(_error("Internal Server Error"), status.HTTP_500_INTERNAL_SERVER_ERROR)

    return _json(_error(exc.detail), exc.status_code)


async def handle_validation_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic RequestValidationError (422)."""
    if not isinstance(exc, RequestValidationError):
        logger.critical("Unexpected exception in validation handler", exc_info=exc)
        return _json(_error("Internal Server Error"), status.HTTP_500_INTERNAL_SERVER_ERROR)

    _MESSAGES = {
        "missing": "Required field",
        "extra_forbidden": "Extra fields not allowed",
        "int_parsing": "Must be an integer",
        "string_type": "Must be a string",
        "greater_than": "Must be greater than {gt}",
        "less_than": "Must be less than {lt}",
        "string_too_short": "Minimum {min_length} characters",
        "string_too_long": "Maximum {max_length} characters",
        "enum": "Allowed values: {expected}",
        "value_error": "Invalid value: {error}",
    }

    errors: dict = {}
    for error in exc.errors():
        if "loc" not in error or not error["loc"]:
            continue
        field = str(error["loc"][-1])
        tmpl = _MESSAGES.get(error.get("type", ""), error.get("msg", "Validation error"))
        try:
            errors[field] = tmpl.format(**error.get("ctx", {}))
        except (KeyError, IndexError):
            errors[field] = error.get("msg", "Validation error")

    content = _error("Validation error", errors) if errors else _error("Validation error")
    return _json(content, status.HTTP_422_UNPROCESSABLE_ENTITY)


async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    """Fallback for unhandled exceptions."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return _json(_error("Internal server error"), status.HTTP_500_INTERNAL_SERVER_ERROR)
