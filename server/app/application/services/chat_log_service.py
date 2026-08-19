"""Application service for admin chat log queries."""

from __future__ import annotations

from datetime import datetime

from application.ports.unit_of_work_factory import UnitOfWorkFactory


class ChatLogService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_logs(
        self,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.list_logs(
                user_id=user_id,
                domain=domain,
                date_from=date_from,
                date_to=date_to,
                search=search,
                limit=limit,
                offset=offset,
            )

    async def count_logs(
        self,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> int:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.count_logs(
                user_id=user_id,
                domain=domain,
                date_from=date_from,
                date_to=date_to,
                search=search,
            )
