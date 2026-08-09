"""Domain exception hierarchy -- client/server split with uniform JSON serialization."""

from domain.exceptions.domain_errors import (
    AppException,
    AuthenticationError,
    BusinessRuleViolation,
    ClientException,
    DatabaseError,
    EntityNotFound,
    PermissionDeniedError,
    ServerException,
    ValidationError,
)

__all__ = [
    "AppException",
    "ClientException",
    "ServerException",
    "ValidationError",
    "EntityNotFound",
    "BusinessRuleViolation",
    "DatabaseError",
    "AuthenticationError",
    "PermissionDeniedError",
]
