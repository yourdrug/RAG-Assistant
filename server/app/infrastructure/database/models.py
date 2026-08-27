"""SQLAlchemy ORM models — mapped to existing PostgreSQL tables.

Uses shared BaseModel (int PK + creation_date) and LinkedBaseModel (M2M join tables).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
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
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)

    conversation = relationship("ConversationModel", back_populates="messages")


class GroupModel(BaseModel):
    __tablename__ = "groups"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    members = relationship("UserModel", secondary="user_groups", back_populates="groups")


class UserGroupModel(LinkedBaseModel):
    __tablename__ = "user_groups"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)


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
        CheckConstraint(
            "doc_domain IN ('legal', 'general')",
            name="documents_doc_domain_check",
        ),
        Index("idx_documents_owner", "owner_id"),
        Index("idx_documents_group", "group_id"),
        Index("idx_documents_visibility", "visibility"),
        Index("idx_documents_status", "status"),
        Index("idx_documents_doc_domain", "doc_domain"),
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
    doc_domain: Mapped[str] = mapped_column(String(16), nullable=False, default="general")
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="file")
    has_manual_edits: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    error_message: Mapped[str | None] = mapped_column(Text)
    warning_message: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float)
    chunks: Mapped[int | None] = mapped_column(Integer)
    chars: Mapped[int | None] = mapped_column(Integer)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime)


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
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)


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
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ChunkModel(BaseModel):
    """Chunk text stored in Postgres for exact substring search (pg_trgm).

    Chunks live primarily in Qdrant (vectors). This table stores the raw text
    for Ctrl+F style search via trigram GIN index.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(
            "doc_domain IN ('legal', 'general')",
            name="chunks_doc_domain_check",
        ),
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    doc_domain: Mapped[str] = mapped_column(String(16), nullable=False, default="general")
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id", ondelete="SET NULL"))
    edited_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    edited_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    manual: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")


class ChatLogModel(BaseModel):
    """Persistent Q&A log for quality tracking."""

    __tablename__ = "chat_logs"
    __table_args__ = (
        Index("idx_chat_logs_user_id", "user_id"),
        Index("idx_chat_logs_creation_date", "creation_date"),
        Index("idx_chat_logs_domain", "domain"),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    breadth: Mapped[str | None] = mapped_column(String(20), nullable=True)
    domain: Mapped[str | None] = mapped_column(String(20), nullable=True)
    retrieval_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reranker_score: Mapped[float | None] = mapped_column(nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BenchmarkQuestionModel(BaseModel):
    """Benchmark test question — source of truth replaces test_questions.json."""

    __tablename__ = "benchmark_questions"
    __table_args__ = (
        Index("idx_benchmark_questions_dataset", "dataset"),
        Index("idx_benchmark_questions_is_active", "is_active"),
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, server_default="main")
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
    )

    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    search_space: Mapped[dict] = mapped_column(JSON, nullable=False)
    objective_weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, server_default="main")
    top_n_llm: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("background_jobs.id", ondelete="SET NULL"), nullable=True
    )
    total_configs: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    evaluated_configs: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    best_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BenchmarkRunModel(BaseModel):
    """Single benchmark run — config snapshot + aggregated metrics."""

    __tablename__ = "benchmark_runs"
    __table_args__ = (
        Index("idx_benchmark_runs_sweep_id", "sweep_id"),
        Index("idx_benchmark_runs_dataset", "dataset"),
    )

    sweep_id: Mapped[int | None] = mapped_column(
        ForeignKey("benchmark_sweeps.id", ondelete="SET NULL"), nullable=True
    )
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    summary_metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    llm_evaluated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, server_default="main")
    per_question_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
