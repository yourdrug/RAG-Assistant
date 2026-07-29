"""SQLAlchemy declarative bases — KinTree-style with AsyncAttrs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from shared.infrastructure.db.settings import settings


class _Base(AsyncAttrs, DeclarativeBase):
    pass


class BaseModel(_Base):
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    creation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=settings.tz),
    )


class LinkedBaseModel(_Base):
    """M2M join tables — no surrogate PK, composite PK from FK columns."""

    __abstract__ = True
