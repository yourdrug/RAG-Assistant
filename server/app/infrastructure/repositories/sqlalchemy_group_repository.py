"""SQLAlchemy implementation of GroupRepository."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


class SQLAlchemyGroupRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, name: str) -> int:
        result = self._db.execute(
            text("INSERT INTO groups (name) VALUES (:name) RETURNING id"),
            {"name": name},
        )
        return result.scalar()

    def list_all(self) -> list[dict]:
        rows = self._db.execute(text("SELECT id, name FROM groups ORDER BY name")).fetchall()
        return [{"id": r.id, "name": r.name} for r in rows]

    def list_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        rows = self._db.execute(
            text("SELECT id, name FROM groups WHERE id = ANY(:ids) ORDER BY name"),
            {"ids": ids},
        ).fetchall()
        return [{"id": r.id, "name": r.name} for r in rows]

    def get_user_group_ids(self, user_id: int) -> list[int]:
        rows = self._db.execute(
            text("SELECT group_id FROM user_groups WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchall()
        return [r.group_id for r in rows]

    def add_user(self, user_id: int, group_id: int) -> None:
        self._db.execute(
            text("INSERT INTO user_groups (user_id, group_id) VALUES (:uid, :gid) ON CONFLICT DO NOTHING"),
            {"uid": user_id, "gid": group_id},
        )

    def remove_user(self, user_id: int, group_id: int) -> None:
        self._db.execute(
            text("DELETE FROM user_groups WHERE user_id = :uid AND group_id = :gid"),
            {"uid": user_id, "gid": group_id},
        )

    def list_members(self, group_id: int) -> list[dict]:
        rows = self._db.execute(
            text("""
                SELECT u.id, u.email FROM users u
                JOIN user_groups ug ON ug.user_id = u.id
                WHERE ug.group_id = :gid
                ORDER BY u.email
            """),
            {"gid": group_id},
        ).fetchall()
        return [{"id": r.id, "email": r.email} for r in rows]
