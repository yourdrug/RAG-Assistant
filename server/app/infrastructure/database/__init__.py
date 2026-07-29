"""Database package — exports the async DatabaseManager singleton."""

from infrastructure.database.database import DatabaseManager, database

__all__ = ["DatabaseManager", "database"]
