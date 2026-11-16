"""ORM models of regulatory acts and their editions.

Alembic discovers ``infrastructure.domain_packs.<pack>.persistence.models`` modules, so the
schema stays one linear migration history for the installation.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.basemodel import BaseModel


class RegulatoryActModel(BaseModel):
    """A legal act that may have multiple versions over time."""

    __tablename__ = "regulatory_acts"
    __table_args__ = (
        # Backstop for the READ→CHECK→WRITE race in find_or_create_act
        Index(
            "ux_regulatory_acts_identity",
            "act_type",
            "act_number",
            "visibility_scope",
            text("COALESCE(act_date, '0001-01-01'::date)"),
            unique=True,
        ),
    )

    act_type: Mapped[str] = mapped_column(String(32), nullable=False)
    act_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    issuing_authority: Mapped[str | None] = mapped_column(String(255), nullable=True)
    act_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    visibility_scope: Mapped[str] = mapped_column(
        String(100), nullable=False, server_default=DocumentVisibility.INTERNAL_PUBLIC.value
    )


class ActVersionModel(BaseModel):
    """A specific version of a regulatory act, linked to a document."""

    __tablename__ = "act_versions"
    __table_args__ = (
        Index("idx_act_versions_act_id", "act_id"),
        # A document is exactly one edition; chunks inherit its metadata by document_id.
        UniqueConstraint("document_id", name="ux_act_versions_document_id"),
        # Guarantee: an act cannot have two "current" versions at once
        Index(
            "ux_act_versions_one_current",
            "act_id",
            unique=True,
            postgresql_where=text("is_current"),
        ),
    )

    act_id: Mapped[int | None] = mapped_column(
        ForeignKey("regulatory_acts.id", ondelete="CASCADE"), nullable=True
    )
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    date_source: Mapped[str] = mapped_column(String(32), nullable=False, server_default="extracted")
    date_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
