"""Law pack repositories over one unit-of-work session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.domain_packs.belarus_law.acts.repositories import ActRepositories
from infrastructure.domain_packs.belarus_law.persistence.act_versions import SQLAlchemyActVersionRepository
from infrastructure.domain_packs.belarus_law.persistence.regulatory_acts import (
    SQLAlchemyRegulatoryActRepository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def sql_act_repositories(session: AsyncSession) -> ActRepositories:
    return ActRepositories(
        regulatory_acts=SQLAlchemyRegulatoryActRepository(session),
        act_versions=SQLAlchemyActVersionRepository(session),
    )
