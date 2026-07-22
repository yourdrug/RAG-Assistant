"""Domain exception hierarchy."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class ValidationError(DomainError):
    """Raised when a business rule or invariant is violated."""


class EntityNotFound(DomainError):
    """Raised when a requested entity does not exist."""

    def __init__(self, entity_name: str, identifier: str | int) -> None:
        super().__init__(f"{entity_name} with id={identifier} not found")
        self.entity_name = entity_name
        self.identifier = identifier


class BusinessRuleViolation(DomainError):
    """Raised when a domain operation violates a business rule."""


class DatabaseError(DomainError):
    """Raised when a database operation fails."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail = detail
