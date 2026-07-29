"""Shared domain exceptions — re-exports from the canonical location."""

from domain.exceptions.domain_errors import (
    AppException,
    ClientException,
    DatabaseError,
    ServerException,
)

__all__ = ["AppException", "ClientException", "ServerException", "DatabaseError"]
