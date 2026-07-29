"""SQLAlchemy implementation of ApiKeyRepository."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from domain.entities.api_key import ApiKey


class SQLAlchemyApiKeyRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, user_id: int, key_hash: str, key_prefix: str, name: str | None = None) -> ApiKey:
        row = self._db.execute(
            text("""
                 INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
                 VALUES (:uid, :hash, :prefix,
                         :name) RETURNING id, user_id, key_hash, key_prefix, name, created_at, revoked_at, last_used_at
                 """),
            {"uid": user_id, "hash": key_hash, "prefix": key_prefix, "name": name},
        ).fetchone()
        return self._to_entity(row)

    def list_for_user(self, user_id: int) -> list[ApiKey]:
        rows = self._db.execute(
            text("""
                 SELECT id,
                        user_id,
                        key_hash,
                        key_prefix,
                        name,
                        created_at,
                        revoked_at,
                        last_used_at
                 FROM api_keys
                 WHERE user_id = :uid
                 ORDER BY created_at DESC
                 """),
            {"uid": user_id},
        ).fetchall()
        return [self._to_entity(r) for r in rows]

    def revoke(self, api_key_id: int, user_id: int | None = None) -> bool:
        params: dict = {"id": api_key_id}
        clause = "id = :id AND revoked_at IS NULL"
        if user_id is not None:
            clause += " AND user_id = :uid"
            params["uid"] = user_id
        result = self._db.execute(
            text(f"UPDATE api_keys SET revoked_at = NOW() WHERE {clause}"),
            params,
        )
        return result.rowcount > 0

    def get_active_client_by_hash(self, key_hash: str) -> dict | None:
        """Ключ действителен, только если не отозван И владелец — kind='client' и активен."""
        row = self._db.execute(
            text("""
                 SELECT ak.id AS api_key_id, u.id AS id, u.email, u.role, u.kind, u.is_active
                 FROM api_keys ak
                          JOIN users u ON u.id = ak.user_id
                 WHERE ak.key_hash = :hash
                   AND ak.revoked_at IS NULL
                   AND u.kind = 'client'
                   AND u.is_active = TRUE LIMIT 1
                 """),
            {"hash": key_hash},
        ).fetchone()
        if row is None:
            return None
        return {
            "api_key_id": row.api_key_id,
            "id": row.id,
            "email": row.email,
            "role": row.role,
            "kind": row.kind,
            "is_active": row.is_active,
        }

    def touch_last_used(self, api_key_id: int) -> None:
        self._db.execute(
            text("UPDATE api_keys SET last_used_at = NOW() WHERE id = :id"),
            {"id": api_key_id},
        )

    @staticmethod
    def _to_entity(row) -> ApiKey:
        return ApiKey(
            id=row.id,
            user_id=row.user_id,
            key_hash=row.key_hash,
            key_prefix=row.key_prefix,
            name=row.name,
            created_at=row.created_at,
            revoked_at=row.revoked_at,
            last_used_at=row.last_used_at,
        )
