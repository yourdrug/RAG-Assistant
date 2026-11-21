"""SQLAlchemy ORM models — mapped to existing PostgreSQL tables.

Uses shared BaseModel (int PK + creation_date) and LinkedBaseModel (M2M join tables).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from domain.benchmark.llm_replay import LLMReplayMode
from domain.benchmark.sweep import (
    BenchmarkStrategy,
    BenchmarkSweepStatus,
    SweepEvaluationMode,
)
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.enums import (
    DOCUMENT_SLOT_STATUSES,
    DocDomain,
    DocumentStatus,
    SourceType,
)
from domain.knowledge.index_sync import OutboxStatus
from domain.runtime.jobs import BackgroundJobStatus
from infrastructure.database.basemodel import BaseModel, LinkedBaseModel


if TYPE_CHECKING:
    from collections.abc import Callable


BENCHMARK_ACTIVE_SWEEP_INDEX = "uq_benchmark_sweeps_one_active"


# Runtime adapters use raw SQL; these declarations keep Alembic aware of their schema.
corpus_control = Table(
    "corpus_control",
    BaseModel.metadata,
    Column("id", Integer, primary_key=True),
    Column("phase", Text, nullable=False),
    Column("write_epoch", BigInteger, nullable=False, server_default="0"),
    Column("active", JSONB, nullable=False),
    Column("candidate", JSONB),
    Column("digest", Text),
    Column("owner", Text),
    Column("error", Text),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("id = 1"),
    CheckConstraint("phase IN ('active', 'preparing', 'committing', 'failed')"),
)

corpus_processes = Table(
    "corpus_processes",
    BaseModel.metadata,
    Column("process_id", Text, primary_key=True),
    Column("role", Text, nullable=False),
    Column("prepared_generation", Text),
    Column("applied_generation", Text),
    Column("error", Text),
    Column("heartbeat_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("role IN ('api', 'worker', 'cli')"),
)


def jsonb_type(*, none_as_null: bool) -> JSONB:
    """Preserve SQL NULL handling across SQLAlchemy's untyped constructor."""
    constructor = cast("Callable[[bool], JSONB]", JSONB)
    return constructor(none_as_null)


class UserModel(BaseModel):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'curator', 'user')", name="users_role_check"),
        CheckConstraint("kind IN ('internal', 'client')", name="users_kind_check"),
        CheckConstraint(
            "NOT (kind = 'client' AND role IN ('admin', 'curator'))",
            name="chk_client_not_privileged",
        ),
    )

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default=UserRole.USER.value)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default=UserKind.INTERNAL.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    groups = relationship("GroupModel", secondary="user_groups", back_populates="members")


class ConversationModel(BaseModel):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_conversations_user", "user_id"),)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # A conversation asks in one space; set by its first question.
    space_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), nullable=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

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
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    conversation = relationship("ConversationModel", back_populates="messages")


class GroupModel(BaseModel):
    __tablename__ = "groups"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    members = relationship("UserModel", secondary="user_groups", back_populates="groups")


class UserGroupModel(LinkedBaseModel):
    __tablename__ = "user_groups"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)


class KnowledgeSpaceModel(BaseModel):
    __tablename__ = "knowledge_spaces"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    pack_key: Mapped[str] = mapped_column(String(64), nullable=False)


class KnowledgeSpaceUploadGroupModel(LinkedBaseModel):
    __tablename__ = "knowledge_space_upload_groups"

    space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), primary_key=True
    )
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)


class ClientSpaceModel(LinkedBaseModel):
    """Support spaces a CLIENT user may ask in."""

    __tablename__ = "client_spaces"
    __table_args__ = (Index("idx_client_spaces_space", "space_id"),)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), primary_key=True
    )


class CuratorUserAssignmentModel(LinkedBaseModel):
    __tablename__ = "curator_user_assignments"
    __table_args__ = (
        CheckConstraint("curator_id != target_user_id", name="chk_curator_not_self_assign"),
        Index("idx_curator_user_assignments_target", "target_user_id"),
    )

    curator_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    target_user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    assigned_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


class CuratorGroupAssignmentModel(LinkedBaseModel):
    __tablename__ = "curator_group_assignments"

    curator_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)
    assigned_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


class DocumentModel(BaseModel):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "visibility IN ('internal_public', 'internal_group', 'internal_private', "
            "'client_private', 'client_public')",
            name="documents_visibility_check",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'indexing', 'done', 'failed')",
            name="documents_status_check",
        ),
        Index("idx_documents_owner", "owner_id"),
        Index("idx_documents_group", "group_id"),
        Index("idx_documents_visibility", "visibility"),
        Index("idx_documents_status", "status"),
        Index("idx_documents_doc_domain", "doc_domain"),
        Index("idx_documents_source_type", "source_type"),
        Index("idx_documents_space", "space_id"),
        Index(
            "idx_documents_version_group",
            "version_group_id",
            postgresql_where="version_group_id IS NOT NULL",
        ),
        CheckConstraint("source_type IN ('file', 'manual')", name="documents_source_type_check"),
        # COALESCE so public/group (NULL owner) rows collide too — NULLs are
        # distinct in plain unique indexes, which let duplicate public uploads
        # through.
        # Channel documents are named by their storage path and identified by the channel;
        # they take no filename slot, so a manual upload never collides with or replaces them.
        Index(
            "ux_documents_active_slot",
            "space_id",
            text("COALESCE(owner_id, 0)"),
            text("COALESCE(group_id, 0)"),
            "filename",
            unique=True,
            postgresql_where="ingestion_key IS NULL AND sync_channel_id IS NULL AND status IN ("
            + ", ".join(f"'{status.value}'" for status in DOCUMENT_SLOT_STATUSES)
            + ")",
        ),
        Index(
            "ux_documents_ingestion_slot",
            "space_id",
            "ingestion_key",
            text("COALESCE(owner_id, 0)"),
            text("COALESCE(group_id, 0)"),
            unique=True,
            postgresql_where="ingestion_key IS NOT NULL AND status IN ("
            + ", ".join(f"'{status.value}'" for status in DOCUMENT_SLOT_STATUSES)
            + ")",
        ),
        Index(
            "ux_documents_sync_file",
            "sync_channel_id",
            "sync_path",
            unique=True,
            postgresql_where="sync_channel_id IS NOT NULL",
        ),
        CheckConstraint(
            "(sync_channel_id IS NULL) = (sync_path IS NULL)", name="documents_sync_binding_check"
        ),
    )

    ingestion_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"))
    space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=DocumentStatus.PENDING.value)
    doc_domain: Mapped[str] = mapped_column(String(32), nullable=False, default=DocDomain.GENERAL.value)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default=SourceType.FILE.value)
    has_manual_edits: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    version_group_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text)
    warning_message: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float)
    chunks: Mapped[int | None] = mapped_column(Integer)
    chars: Mapped[int | None] = mapped_column(Integer)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # A channel cannot be deleted while documents are bound to it: delete or unlink them first.
    sync_channel_id: Mapped[int | None] = mapped_column(ForeignKey("sync_channels.id"), nullable=True)
    sync_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    storage_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SyncChannelModel(BaseModel):
    """Storage location kept in line with documents of one space; see ``domain.sync.channel``."""

    __tablename__ = "sync_channels"
    __table_args__ = (
        CheckConstraint("storage_kind IN ('github', 's3')", name="sync_channels_storage_kind_check"),
        CheckConstraint(
            "visibility IN ('internal_public', 'internal_group', 'internal_private', "
            "'client_private', 'client_public')",
            name="sync_channels_visibility_check",
        ),
        CheckConstraint("jsonb_typeof(location) = 'object'", name="sync_channels_location_object"),
        CheckConstraint("jsonb_typeof(exclusions) = 'array'", name="sync_channels_exclusions_array"),
        CheckConstraint("generation > 0", name="sync_channels_generation_positive"),
        Index("idx_sync_channels_space", "space_id"),
    )

    space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="RESTRICT"), nullable=False
    )
    storage_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    location: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    folder: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    exclusions: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    # Same lifetime as the channel's documents, whose owner and group cascade as well.
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"))
    responsible_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    generation: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")


class SyncRunModel(BaseModel):
    """One reconciliation of a channel; see ``domain.sync.run``."""

    __tablename__ = "sync_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'partial', 'failed', 'cancelled', 'rejected')",
            name="sync_runs_status_check",
        ),
        CheckConstraint("trigger IN ('manual', 'webhook', 'schedule')", name="sync_runs_trigger_check"),
        CheckConstraint("jsonb_typeof(outcomes) = 'array'", name="sync_runs_outcomes_array"),
        Index("idx_sync_runs_channel", "channel_id", "id"),
        Index(
            "ux_sync_runs_delivery",
            "channel_id",
            "delivery_id",
            unique=True,
            postgresql_where="delivery_id IS NOT NULL",
        ),
    )

    channel_id: Mapped[int] = mapped_column(
        ForeignKey("sync_channels.id", ondelete="CASCADE"), nullable=False
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)
    delivery_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    storage_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcomes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConfigParameterModel(BaseModel):
    """Dynamic configuration parameters — hot-reloadable without restart.

    WARNING: This table is NOT suitable for storing secrets (JWT keys, DB passwords,
    API keys, etc.). Values are returned in plaintext via GET /admin/config, logged
    in audit_log_config_change, and visible to any admin user. Use .env / environment
    variables for sensitive configuration.
    """

    __tablename__ = "config_parameters"
    __table_args__ = (
        # (key, domain_key) with NULLS NOT DISTINCT: NULLs-distinct let two
        # concurrent seeding processes create duplicate rows, which made every
        # get_by_key raise MultipleResultsFound forever.
        UniqueConstraint(
            "key",
            "domain_key",
            name="ux_config_parameters_key_domain",
            postgresql_nulls_not_distinct=True,
        ),
        Index("idx_config_parameters_category", "category"),
        Index("idx_config_parameters_domain_key", "domain_key", postgresql_where="domain_key IS NOT NULL"),
    )

    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), nullable=False, default="str")
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    min_value: Mapped[float | None] = mapped_column(nullable=True)
    max_value: Mapped[float | None] = mapped_column(nullable=True)
    allowed_values: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON, nullable=True)
    domain_key: Mapped[str | None] = mapped_column(String(32), nullable=True)


class ApiKeyModel(BaseModel):
    __tablename__ = "api_keys"
    __table_args__ = (Index("idx_api_keys_user_id", "user_id"),)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BackgroundJobModel(BaseModel):
    __tablename__ = "background_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name="background_jobs_status_check",
        ),
        Index(
            "ux_background_jobs_active_dedup",
            "deduplication_key",
            unique=True,
            postgresql_where=text("deduplication_key IS NOT NULL AND status IN ('pending', 'running')"),
            sqlite_where=text("deduplication_key IS NOT NULL AND status IN ('pending', 'running')"),
        ),
        Index("idx_background_jobs_status", "status"),
        Index("idx_background_jobs_job_type", "job_type"),
        Index("idx_background_jobs_creation_date", "creation_date"),
        Index("idx_background_jobs_heartbeat", "heartbeat_at"),
        Index(
            "idx_background_jobs_relay",
            "heartbeat_at",
            "id",
            postgresql_where=text("status = 'pending' AND dispatch_payload IS NOT NULL"),
            sqlite_where=text("status = 'pending' AND dispatch_payload IS NOT NULL"),
        ),
    )

    payload_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    deduplication_key: Mapped[str | None] = mapped_column(String(128))
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    attempt_id: Mapped[str | None] = mapped_column(String(36))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dispatch_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    progress: Mapped[dict[str, str] | None] = mapped_column(
        jsonb_type(none_as_null=True).with_variant(JSON(none_as_null=True), "sqlite")
    )
    dispatch_payload: Mapped[dict[str, object] | None] = mapped_column(
        jsonb_type(none_as_null=True).with_variant(JSON(none_as_null=True), "sqlite")
    )
    # W3C trace context of the submitting request; outside the task payload contract.
    trace_context: Mapped[dict[str, str] | None] = mapped_column(
        jsonb_type(none_as_null=True).with_variant(JSON(none_as_null=True), "sqlite")
    )
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=BackgroundJobStatus.PENDING.value)
    related_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ChunkModel(BaseModel):
    """Chunk text stored in Postgres for exact substring search (pg_trgm).

    Chunks live primarily in Qdrant (vectors). This table stores the raw text
    for Ctrl+F style search via trigram GIN index.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", "id", name="ux_chunks_document_chunk_index_id"),
        UniqueConstraint("document_id", "chunk_index", name="ux_chunks_document_index"),
        Index("ix_chunks_visibility", "visibility"),
        Index("ix_chunks_owner_id", "owner_id"),
        Index("ix_chunks_group_id", "group_id"),
        Index("ix_chunks_space_id", "space_id"),
        Index("idx_chunks_content_hash", "content_hash"),
        Index(
            "ix_chunks_content_trgm",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
        Index("ix_chunks_edited_by", "edited_by", postgresql_where="edited_by IS NOT NULL"),
        CheckConstraint(
            "jsonb_typeof(context_metadata) = 'object'", name="chk_chunks_context_metadata_object"
        ),
        CheckConstraint("jsonb_typeof(pack_metadata) = 'object'", name="chk_chunks_pack_metadata_object"),
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    doc_domain: Mapped[str] = mapped_column(String(32), nullable=False, default=DocDomain.GENERAL.value)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="SET NULL"))
    # Copy of the document's space: a search selection, not an access condition.
    space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="RESTRICT"), nullable=False
    )
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    edited_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    manual: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    content_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    domain_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    context_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    # Owned and interpreted by the domain pack; the core stores and carries it.
    pack_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    heading_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    doc_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True)


class DocumentStagedRevisionModel(LinkedBaseModel):
    """Hidden revision of a document's chunks, published once all its points are indexed.

    The revision's parent nodes wait here as JSON; its chunks wait in ``chunk_staging``.
    """

    __tablename__ = "document_staged_revisions"

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    token: Mapped[str] = mapped_column(String(32), nullable=False)
    nodes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    creation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class StagedChunkModel(LinkedBaseModel):
    """Chunk of a staged revision; same columns as ``chunks`` and ids from its sequence.

    Never read by search: its points are hidden in Qdrant until the revision is published.
    """

    __tablename__ = "chunk_staging"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="ux_chunk_staging_document_index"),
        CheckConstraint(
            "jsonb_typeof(context_metadata) = 'object'", name="chk_chunk_staging_context_metadata_object"
        ),
        CheckConstraint(
            "jsonb_typeof(pack_metadata) = 'object'", name="chk_chunk_staging_pack_metadata_object"
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, server_default=text("nextval('chunks_id_seq'::regclass)")
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document_staged_revisions.document_id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    doc_domain: Mapped[str] = mapped_column(String(32), nullable=False, default=DocDomain.GENERAL.value)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="SET NULL"))
    space_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="RESTRICT"), nullable=False
    )
    content_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    domain_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    context_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    pack_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    heading_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    doc_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    creation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentNodeModel(BaseModel):
    """Parent scope of chunks (article, point, section) for parent-child retrieval.

    Holds no text: content is read from ``chunks`` under the chunk ACL. Never query it
    without proving access to a chunk of the same document.
    """

    __tablename__ = "document_nodes"
    __table_args__ = (UniqueConstraint("document_id", "node_id", name="ux_document_nodes_document_node"),)

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    node_id: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_start: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_end: Mapped[int] = mapped_column(Integer, nullable=False)
    pages: Mapped[list[int]] = mapped_column(JSONB, nullable=False, default=list)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(16), nullable=False)


rag_config_snapshots = Table(
    "rag_config_snapshots",
    BaseModel.metadata,
    Column("hash", String(64), primary_key=True),
    Column("data", JSONB, nullable=False),
    Column("creation_date", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


class ChatLogModel(BaseModel):
    """Persistent Q&A log for quality tracking."""

    __tablename__ = "chat_logs"
    __table_args__ = (
        Index("idx_chat_logs_user_id", "user_id"),
        Index("idx_chat_logs_creation_date", "creation_date"),
        Index("idx_chat_logs_domain", "domain"),
        Index("idx_chat_logs_outcome", "outcome"),
        Index("idx_chat_logs_message_id", "message_id", unique=True),
        Index("idx_chat_logs_config_hash", "config_hash"),
        Index("idx_chat_logs_space_date", "space_id", "creation_date"),
        Index("idx_chat_logs_date_outcome", "creation_date", "outcome"),
    )

    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    breadth: Mapped[str | None] = mapped_column(String(20), nullable=True)
    domain: Mapped[str | None] = mapped_column(String(20), nullable=True)
    message_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    space_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_spaces.id", ondelete="RESTRICT"), nullable=False
    )
    pack_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    depth_requested: Mapped[str | None] = mapped_column(String(20), nullable=True)
    as_of_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    outcome: Mapped[str] = mapped_column(String(30), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    condensed_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    sub_queries: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    refined_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    sources_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidates_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    top_rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    aux_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieval_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rerank_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generation_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config_hash: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("rag_config_snapshots.hash", ondelete="RESTRICT"), nullable=True
    )
    config_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    evidence_profile: Mapped[str | None] = mapped_column(String(200), nullable=True)
    answer_style: Mapped[str | None] = mapped_column(String(20), nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BenchmarkQuestionModel(BaseModel):
    """Benchmark test question — database source of truth for all evaluations."""

    __tablename__ = "benchmark_questions"
    __table_args__ = (
        CheckConstraint(
            "annotations IS NULL OR jsonb_typeof(annotations) = 'object'",
            name="chk_benchmark_questions_annotations_object",
        ),
        Index("idx_benchmark_questions_dataset", "dataset"),
        Index("idx_benchmark_questions_is_active", "is_active"),
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    case_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    variant_of: Mapped[str | None] = mapped_column(String(100), nullable=True)
    question_style: Mapped[str | None] = mapped_column(String(100), nullable=True)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotations: Mapped[dict[str, Any] | None] = mapped_column(
        jsonb_type(none_as_null=True),
        nullable=True,
    )
    history: Mapped[list[dict[str, str]] | None] = mapped_column(
        jsonb_type(none_as_null=True),
        nullable=True,
    )
    as_of_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, server_default="belarus_law-main")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class BenchmarkSweepModel(BaseModel):
    """Parameter sweep — multi-config automated search."""

    __tablename__ = "benchmark_sweeps"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name="benchmark_sweeps_status_check",
        ),
        Index("idx_benchmark_sweeps_status", "status"),
        CheckConstraint(
            f"evaluation_mode IN ('{SweepEvaluationMode.FAST.value}', '{SweepEvaluationMode.FULL.value}')",
            name="benchmark_sweeps_evaluation_mode_check",
        ),
        CheckConstraint(
            f"evaluation_mode != '{SweepEvaluationMode.FULL.value}' "
            f"OR strategy = '{BenchmarkStrategy.GRID.value}'",
            name="benchmark_sweeps_full_grid_check",
        ),
        CheckConstraint(
            "llm_replay_mode IN (" + ", ".join(f"'{mode.value}'" for mode in LLMReplayMode) + ")",
            name="benchmark_sweeps_llm_replay_mode_check",
        ),
        Index(
            BENCHMARK_ACTIVE_SWEEP_INDEX,
            text("(1)"),
            unique=True,
            postgresql_where=text(
                f"status IN ('{BenchmarkSweepStatus.PENDING.value}', '{BenchmarkSweepStatus.RUNNING.value}')"
            ),
        ),
    )

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=BenchmarkSweepStatus.PENDING.value
    )
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    search_space: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    objective_weights: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, server_default="belarus_law-main")
    space_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_spaces.id", ondelete="RESTRICT", name="fk_benchmark_sweeps_space_id"),
        nullable=True,
    )
    top_n_llm: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    judge_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evaluation_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=SweepEvaluationMode.FAST.value
    )
    llm_replay_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=LLMReplayMode.OFF.value
    )
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("background_jobs.id", ondelete="SET NULL"), nullable=True
    )
    total_configs: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    evaluated_configs: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    best_run_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "benchmark_runs.id", ondelete="SET NULL", name="fk_benchmark_sweeps_best_run", use_alter=True
        ),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class BenchmarkRunModel(BaseModel):
    """Single benchmark run — config snapshot + aggregated metrics."""

    __tablename__ = "benchmark_runs"
    __table_args__ = (
        Index("idx_benchmark_runs_sweep_id", "sweep_id"),
        Index("idx_benchmark_runs_dataset", "dataset"),
        UniqueConstraint("job_id", name="uq_benchmark_runs_job_id"),
        UniqueConstraint("legacy_hash", name="uq_benchmark_runs_legacy_hash"),
    )

    sweep_id: Mapped[int | None] = mapped_column(
        ForeignKey("benchmark_sweeps.id", ondelete="SET NULL"), nullable=True
    )
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    summary_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    llm_evaluated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, server_default="belarus_law-main")
    per_question_results: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    review_rows: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True, deferred=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    legacy_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifacts: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict, server_default="{}")


class VectorStoreOutboxModel(BaseModel):
    """Outbox for applying changes to Qdrant after Postgres transaction commits.

    Part of the Transactional Outbox pattern ensuring Postgres ↔ Qdrant consistency.
    """

    __tablename__ = "vector_store_outbox"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'in_progress', 'done', 'failed', 'dead_letter')",
            name="vector_store_outbox_status_check",
        ),
        CheckConstraint(
            "operation IN ('upsert_chunks', 'delete_by_document', 'delete_chunks', "
            "'update_metadata', 'set_document_id', 'rename_document', 'publish_revision')",
            name="vector_store_outbox_operation_check",
        ),
        Index(
            "idx_outbox_dispatch",
            "status",
            "next_attempt_at",
            postgresql_where="status IN ('pending', 'failed')",
        ),
        Index("idx_outbox_aggregate", "aggregate_type", "aggregate_id"),
    )

    operation: Mapped[str] = mapped_column(String(30), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(20), nullable=False, default="document")
    aggregate_id: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=OutboxStatus.PENDING.value)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IngestionRegistryModel(BaseModel):
    """Tracks which files have been indexed — replaces JSON-file registry."""

    __tablename__ = "ingestion_registry"
    __table_args__ = (Index("idx_ingestion_registry_key", "ingestion_key", unique=True),)

    ingestion_key: Mapped[str] = mapped_column(Text, nullable=False)

    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    source: Mapped[str] = mapped_column(String(1000), nullable=False, server_default="")
    chunks: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    chars: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BenchmarkCheckpointModel(LinkedBaseModel):
    """A committed snapshot of paid work, independent of the worker filesystem."""

    __tablename__ = "benchmark_checkpoints"

    sweep_id: Mapped[int] = mapped_column(
        ForeignKey("benchmark_sweeps.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


chat_log_traces = Table(
    "chat_log_traces",
    BaseModel.metadata,
    Column("chat_log_id", Integer, ForeignKey("chat_logs.id", ondelete="CASCADE"), primary_key=True),
    Column("schema_version", Integer, nullable=False),
    Column("data", JSONB, nullable=False),
    CheckConstraint("octet_length(data::text) <= 65536", name="ck_chat_log_trace_size"),
)


message_feedback = Table(
    "message_feedback",
    BaseModel.metadata,
    Column("id", Integer, primary_key=True),
    Column("message_id", Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("rating", String(8), nullable=False),
    Column("reasons", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("comment", Text, nullable=True),
    Column("creation_date", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("update_date", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("message_id", "user_id", name="uq_message_feedback_message_user"),
    CheckConstraint("rating IN ('up', 'down')", name="ck_message_feedback_rating"),
    CheckConstraint("rating = 'down' OR reasons = '[]'::jsonb", name="ck_message_feedback_reasons_down"),
)


chat_log_reviews = Table(
    "chat_log_reviews",
    BaseModel.metadata,
    Column("chat_log_id", Integer, ForeignKey("chat_logs.id", ondelete="CASCADE"), primary_key=True),
    Column("status", String(30), nullable=False),
    Column("note", Text, nullable=True),
    Column("reviewer_id", Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    Column(
        "benchmark_question_id",
        Integer,
        ForeignKey("benchmark_questions.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("update_date", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(
        "status IN ('new', 'benchmark_candidate', 'in_benchmark', 'content_gap', "
        "'pipeline_issue', 'wont_fix')",
        name="ck_chat_log_reviews_status",
    ),
    Index("idx_chat_log_reviews_status", "status"),
)


benchmark_dataset_policies = Table(
    "benchmark_dataset_policies",
    BaseModel.metadata,
    Column("dataset", String(100), primary_key=True),
    Column("thresholds", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("noise", JSONB, nullable=True),
    Column("calibration", JSONB, nullable=True),
    Column("updated_by", Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    Column("update_date", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


benchmark_judge_calibrations = Table(
    "benchmark_judge_calibrations",
    BaseModel.metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset", String(100), nullable=False),
    # History outlives the run it was measured on.
    Column("run_id", Integer, ForeignKey("benchmark_runs.id", ondelete="SET NULL"), nullable=True),
    Column("judge_model", String(200), nullable=True),
    Column("judge_rubric_version", String(100), nullable=True),
    Column("metrics", JSONB, nullable=False),
    Column("created_by", Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    Column("measured_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("idx_benchmark_judge_calibrations_dataset", "dataset", text("measured_at DESC")),
)


benchmark_human_labels = Table(
    "benchmark_human_labels",
    BaseModel.metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", Integer, ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False),
    Column("question_key", String(200), nullable=False),
    Column("metric", String(20), nullable=False),
    Column("score", Float, nullable=False),
    Column("labeler_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("note", Text, nullable=True),
    Column("update_date", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("metric IN ('correctness', 'faithfulness')", name="ck_benchmark_human_labels_metric"),
    CheckConstraint("score >= 0 AND score <= 10", name="ck_benchmark_human_labels_score"),
    UniqueConstraint("run_id", "question_key", "metric", "labeler_id", name="uq_benchmark_human_labels_case"),
)


# Recorded LLM responses reused by benchmark runs (infrastructure.ml.replay).
llm_replay_entries = Table(
    "llm_replay_entries",
    BaseModel.metadata,
    Column("key", String(64), primary_key=True),
    Column("model", String(255), nullable=True),
    Column("status_code", SmallInteger, nullable=False),
    Column("headers", Text, nullable=False),
    Column("body", LargeBinary, nullable=False),
    Column("input_tokens", Integer, nullable=True),
    Column("output_tokens", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("last_used_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("idx_llm_replay_entries_last_used_at", "last_used_at"),
)
