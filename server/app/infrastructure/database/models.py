"""SQLAlchemy ORM models — mapped to existing PostgreSQL tables.

Uses shared BaseModel (int PK + creation_date) and LinkedBaseModel (M2M join tables).
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from infrastructure.database.basemodel import BaseModel, LinkedBaseModel


class UserModel(BaseModel):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'user')", name="users_role_check"),
        CheckConstraint("kind IN ('internal', 'client')", name="users_kind_check"),
        CheckConstraint("NOT (kind = 'client' AND role = 'admin')", name="chk_client_not_admin"),
    )

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="internal")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    groups = relationship("GroupModel", secondary="user_groups", back_populates="members")


class ConversationModel(BaseModel):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_conversations_user", "user_id"),)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan")


class MessageModel(BaseModel):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
        Index("idx_messages_conv", "conversation_id"),
    )

    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    conversation = relationship("ConversationModel", back_populates="messages")


class GroupModel(BaseModel):
    __tablename__ = "groups"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    members = relationship("UserModel", secondary="user_groups", back_populates="groups")


class UserGroupModel(LinkedBaseModel):
    __tablename__ = "user_groups"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)


class ClientAssignmentModel(LinkedBaseModel):
    __tablename__ = "client_assignments"

    internal_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    client_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    assigned_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    assigned_at: Mapped[DateTime | None] = mapped_column(DateTime)


class DocumentModel(BaseModel):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "visibility IN ('internal_public', 'internal_group', 'internal_private', 'client_private')",
            name="documents_visibility_check",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')",
            name="documents_status_check",
        ),
        Index("idx_documents_owner", "owner_id"),
        Index("idx_documents_group", "group_id"),
        Index("idx_documents_visibility", "visibility"),
        Index("idx_documents_status", "status"),
        Index(
            "ux_documents_active_slot",
            "owner_id",
            "filename",
            unique=True,
            postgresql_where="status IN ('pending', 'processing', 'done')",
        ),
    )

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    warning_message: Mapped[str | None] = mapped_column(Text)
    chunks: Mapped[int | None] = mapped_column(Integer)
    chars: Mapped[int | None] = mapped_column(Integer)
    indexed_at: Mapped[DateTime | None] = mapped_column(DateTime)


class ConfigParameterModel(BaseModel):
    """Dynamic configuration parameters — hot-reloadable without restart.

    WARNING: This table is NOT suitable for storing secrets (JWT keys, DB passwords,
    API keys, etc.). Values are returned in plaintext via GET /admin/config, logged
    in audit_log_config_change, and visible to any admin user. Use .env / environment
    variables for sensitive configuration.
    """

    __tablename__ = "config_parameters"
    __table_args__ = (Index("idx_config_parameters_category", "category"),)

    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), nullable=False, default="str")
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    min_value: Mapped[float | None] = mapped_column(nullable=True)
    max_value: Mapped[float | None] = mapped_column(nullable=True)
    allowed_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ApiKeyModel(BaseModel):
    __tablename__ = "api_keys"
    __table_args__ = (Index("idx_api_keys_user_id", "user_id"),)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    revoked_at: Mapped[DateTime | None] = mapped_column(DateTime)
    last_used_at: Mapped[DateTime | None] = mapped_column(DateTime)


class BackgroundJobModel(BaseModel):
    __tablename__ = "background_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed')",
            name="background_jobs_status_check",
        ),
        Index("idx_background_jobs_status", "status"),
        Index("idx_background_jobs_job_type", "job_type"),
        Index("idx_background_jobs_creation_date", "creation_date"),
    )

    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    related_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ChunkModel(BaseModel):
    """Chunk text stored in Postgres for exact substring search (pg_trgm).

    Chunks live primarily in Qdrant (vectors). This table stores the raw text
    for Ctrl+F style search via trigram GIN index.
    """

    __tablename__ = "chunks"

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="SET NULL"))
