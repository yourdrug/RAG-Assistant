"""Domain exception hierarchy -- client/server split with uniform JSON serialization."""

from domain.exceptions.domain_errors import (
    AppException,
    AuthenticationError,
    BenchmarkQuestionsNotFound,
    BusinessRuleViolation,
    ClientException,
    DatabaseError,
    EntityNotFound,
    PermissionDeniedError,
    ServerException,
    UniqueConstraintViolation,
    ValidationError,
)

__all__ = [
    "AppException",
    "BenchmarkQuestionsNotFound",
    "ClientException",
    "ServerException",
    "ValidationError",
    "EntityNotFound",
    "BusinessRuleViolation",
    "DatabaseError",
    "UniqueConstraintViolation",
    "AuthenticationError",
    "PermissionDeniedError",
]
