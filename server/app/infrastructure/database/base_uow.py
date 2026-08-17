"""Base Unit of Work -- async transaction management with automatic commit/rollback.

Concrete subclasses expose repository attributes (``documents``, ``users``, etc.)
and override ``_commit()`` to flush changes.  Used as an async context manager:
on exit, ``commit`` is called on success and ``rollback`` on exception.

This module re-exports ``BaseUnitOfWork`` from the application layer for
backward compatibility.  New code should import from ``application.ports.base_uow``.
"""

from __future__ import annotations

# Re-export from application layer — infrastructure no longer owns this abstraction.
from application.ports.base_uow import BaseUnitOfWork  # noqa: F401
from application.ports.session_protocol import SessionProtocol  # noqa: F401

__all__ = ["BaseUnitOfWork", "SessionProtocol"]
