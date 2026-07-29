"""SQLAlchemy ORM implementation of ClientAssignmentRepository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ClientAssignmentModel, UserModel


class SQLAlchemyClientAssignmentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def assign(self, internal_user_id: int, client_user_id: int, assigned_by: int) -> None:
        orm = ClientAssignmentModel(
            internal_user_id=internal_user_id,
            client_user_id=client_user_id,
            assigned_by=assigned_by,
        )
        self._db.add(orm)
        await self._db.flush()

    async def unassign(self, internal_user_id: int, client_user_id: int) -> None:
        result = await self._db.execute(
            select(ClientAssignmentModel).where(
                ClientAssignmentModel.internal_user_id == internal_user_id,
                ClientAssignmentModel.client_user_id == client_user_id,
            )
        )
        orm = result.scalar_one_or_none()
        if orm:
            await self._db.delete(orm)
            await self._db.flush()

    async def get_assigned_client_ids(self, internal_user_id: int) -> list[int]:
        result = await self._db.execute(
            select(ClientAssignmentModel.client_user_id).where(
                ClientAssignmentModel.internal_user_id == internal_user_id
            )
        )
        return [row[0] for row in result.all()]

    async def list_for_client(self, client_user_id: int) -> list[dict]:
        result = await self._db.execute(
            select(
                UserModel.id.label("internal_user_id"),
                UserModel.email,
                ClientAssignmentModel.assigned_at,
            )
            .join(UserModel, UserModel.id == ClientAssignmentModel.internal_user_id)
            .where(ClientAssignmentModel.client_user_id == client_user_id)
            .order_by(ClientAssignmentModel.assigned_at)
        )
        return [
            {
                "internal_user_id": row.internal_user_id,
                "email": row.email,
                "assigned_at": row.assigned_at,
            }
            for row in result.all()
        ]
