"""SQLAlchemy implementation of UserRepository."""

from __future__ import annotations

from domain.entities.user import User
from domain.value_objects.roles import UserKind, UserRole
from sqlalchemy import text
from sqlalchemy.orm import Session


class SQLAlchemyUserRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, user_id: int) -> User | None:
        row = self._db.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user_id}).fetchone()
        return self._to_entity(row) if row else None

    def get_by_email(self, email: str) -> User | None:
        row = self._db.execute(
            text("SELECT * FROM users WHERE email = :email"), {"email": email.lower()}
        ).fetchone()
        return self._to_entity(row) if row else None

    def save(self, user: User) -> User:
        result = self._db.execute(
            text("""
                INSERT INTO users (email, hashed_password, role, kind)
                VALUES (:email, :hashed_password, :role, :kind)
                RETURNING id, email, hashed_password, role, kind, is_active, created_at
            """),
            {
                "email": user.email.lower(),
                "hashed_password": user.hashed_password,
                "role": user.role,
                "kind": user.kind,
            },
        )
        return self._to_entity(result.fetchone())

    def exists_admin(self) -> bool:
        row = self._db.execute(text("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")).fetchone()
        return row is not None

    def list_all(self) -> list[User]:
        rows = self._db.execute(
            text(
                "SELECT id, email, hashed_password, role, kind, is_active, created_at FROM users ORDER BY created_at"
            )
        ).fetchall()
        return [self._to_entity(r) for r in rows]

    def set_active(self, user_id: int, is_active: bool) -> bool:
        result = self._db.execute(
            text("UPDATE users SET is_active = :active WHERE id = :id"),
            {"active": is_active, "id": user_id},
        )
        return result.rowcount > 0

    @staticmethod
    def _to_entity(row) -> User:
        return User(
            id=row.id,
            email=row.email,
            hashed_password=row.hashed_password,
            role=UserRole(row.role),
            kind=UserKind(row.kind),
            is_active=row.is_active,
            created_at=row.created_at,
        )
