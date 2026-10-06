"""Domain exception hierarchy -- client/server split with uniform JSON serialization."""

from domain.exceptions.domain_errors import (
    AppException,
    AuthenticationError,
    BusinessRuleViolation,
    ClientException,
    ContextBudgetExceededError,
    DatabaseError,
    EntityNotFound,
    LLMUnavailableError,
    PermissionDeniedError,
    RateLimitExceededError,
    SemaphoreTimeoutError,
    ServerException,
    UniqueConstraintViolation,
    ValidationError,
)

__all__ = [
    "AppException",
    "ClientException",
    "ServerException",
    "ValidationError",
    "ContextBudgetExceededError",
    "EntityNotFound",
    "BusinessRuleViolation",
    "DatabaseError",
    "UniqueConstraintViolation",
    "AuthenticationError",
    "PermissionDeniedError",
    "LLMUnavailableError",
    "SemaphoreTimeoutError",
    "RateLimitExceededError",
]
