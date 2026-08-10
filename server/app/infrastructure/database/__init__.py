"""Database package -- async SQLAlchemy engine, session management, and ORM models.

Exports the module-level ``database`` singleton (``DatabaseManager``) used
throughout the application for read/write session access.
"""

from infrastructure.database.database import DatabaseManager, database

__all__ = ["DatabaseManager", "database"]
