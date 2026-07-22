"""SQLAlchemy implementation of ClientAssignmentRepository."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


class SQLAlchemyClientAssignmentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def assign(self, internal_user_id: int, client_user_id: int, assigned_by: int) -> None:
        self._db.execute(
            text("""
                INSERT INTO client_assignments (internal_user_id, client_user_id, assigned_by)
                VALUES (:iu, :cu, :by)
                ON CONFLICT DO NOTHING
            """),
            {"iu": internal_user_id, "cu": client_user_id, "by": assigned_by},
        )

    def unassign(self, internal_user_id: int, client_user_id: int) -> None:
        self._db.execute(
            text("DELETE FROM client_assignments WHERE internal_user_id = :iu AND client_user_id = :cu"),
            {"iu": internal_user_id, "cu": client_user_id},
        )

    def get_assigned_client_ids(self, internal_user_id: int) -> list[int]:
        rows = self._db.execute(
            text("SELECT client_user_id FROM client_assignments WHERE internal_user_id = :iu"),
            {"iu": internal_user_id},
        ).fetchall()
        return [r.client_user_id for r in rows]

    def list_for_client(self, client_user_id: int) -> list[dict]:
        rows = self._db.execute(
            text("""
                SELECT u.id AS internal_user_id, u.email, ca.assigned_at
                FROM client_assignments ca
                JOIN users u ON u.id = ca.internal_user_id
                WHERE ca.client_user_id = :cu
                ORDER BY ca.assigned_at
            """),
            {"cu": client_user_id},
        ).fetchall()
        return [
            {"internal_user_id": r.internal_user_id, "email": r.email, "assigned_at": r.assigned_at}
            for r in rows
        ]
