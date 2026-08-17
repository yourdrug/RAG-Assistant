"""Async Session Protocol -- abstract database session interface.

Re-exports ``SessionProtocol`` from the application layer for backward
compatibility.  New code should import from ``application.ports.session_protocol``.
"""

from __future__ import annotations

from application.ports.session_protocol import SessionProtocol  # noqa: F401

__all__ = ["SessionProtocol"]
