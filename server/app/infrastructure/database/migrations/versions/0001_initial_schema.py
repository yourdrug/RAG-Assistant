"""Initial schema for empty databases; replaces the historical migration chain.

Existing installations require the separately coordinated database reset.
The shared pg_trgm extension is retained on downgrade.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def create_tables() -> None:
    op.create_table(
        "background_jobs",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("job_type", sa.VARCHAR(length=50), nullable=False),
        sa.Column(
            "status",
            sa.VARCHAR(length=20),
            server_default=sa.text("'pending'::character varying"),
            nullable=False,
        ),
        sa.Column("related_id", sa.INTEGER(), nullable=True),
        sa.Column("request_id", sa.VARCHAR(length=32), nullable=True),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error_message", sa.TEXT(), nullable=True),
        sa.Column("heartbeat_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed')", name="background_jobs_status_check"
        ),
        sa.PrimaryKeyConstraint("id", name="background_jobs_pkey"),
    )
    op.create_table(
        "config_parameters",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("key", sa.VARCHAR(length=100), nullable=False),
        sa.Column("value", sa.TEXT(), nullable=False),
        sa.Column(
            "value_type",
            sa.VARCHAR(length=20),
            server_default=sa.text("'str'::character varying"),
            nullable=False,
        ),
        sa.Column("category", sa.VARCHAR(length=50), nullable=False),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("min_value", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column("max_value", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column("allowed_values", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("domain_key", sa.VARCHAR(length=32), nullable=True),
        sa.PrimaryKeyConstraint("id", name="config_parameters_pkey"),
        sa.UniqueConstraint(
            "key", "domain_key", name="ux_config_parameters_key_domain", postgresql_nulls_not_distinct=True
        ),
    )
    op.create_table(
        "corpus_control",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("phase", sa.TEXT(), nullable=False),
        sa.Column("write_epoch", sa.BIGINT(), server_default=sa.text("'0'::bigint"), nullable=False),
        sa.Column("active", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("candidate", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("digest", sa.TEXT(), nullable=True),
        sa.Column("owner", sa.TEXT(), nullable=True),
        sa.Column("error", sa.TEXT(), nullable=True),
        sa.Column(
            "updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(
            "phase = ANY (ARRAY['active'::text, 'preparing'::text, 'committing'::text, 'failed'::text])",
            name="corpus_control_phase_check",
        ),
        sa.CheckConstraint("id = 1", name="corpus_control_id_check"),
        sa.PrimaryKeyConstraint("id", name="corpus_control_pkey"),
    )
    op.create_table(
        "corpus_processes",
        sa.Column("process_id", sa.TEXT(), nullable=False),
        sa.Column("role", sa.TEXT(), nullable=False),
        sa.Column("prepared_generation", sa.TEXT(), nullable=True),
        sa.Column("applied_generation", sa.TEXT(), nullable=True),
        sa.Column("error", sa.TEXT(), nullable=True),
        sa.Column(
            "heartbeat_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role = ANY (ARRAY['api'::text, 'worker'::text, 'cli'::text])", name="corpus_processes_role_check"
        ),
        sa.PrimaryKeyConstraint("process_id", name="corpus_processes_pkey"),
    )
    op.create_table(
        "cost_rates",
        sa.Column("provider", sa.TEXT(), nullable=False),
        sa.Column("model", sa.TEXT(), nullable=False),
        sa.Column("version", sa.INTEGER(), nullable=False),
        sa.Column("unit", sa.TEXT(), nullable=False),
        sa.Column("input_usd_per_million", sa.NUMERIC(precision=24, scale=12), nullable=False),
        sa.Column("output_usd_per_million", sa.NUMERIC(precision=24, scale=12), nullable=False),
        sa.Column("context_tokens", sa.INTEGER(), nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(
            "input_usd_per_million >= 0::numeric AND output_usd_per_million >= 0::numeric",
            name="cost_rates_check1",
        ),
        sa.CheckConstraint("version > 0 AND context_tokens > 0", name="cost_rates_check"),
        sa.PrimaryKeyConstraint("provider", "model", "version", name="cost_rates_pkey"),
    )
    op.create_table(
        "groups",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("name", sa.VARCHAR(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id", name="groups_pkey"),
        sa.UniqueConstraint("name", name="groups_name_key"),
    )
    op.create_table(
        "ingestion_registry",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("filename", sa.VARCHAR(length=500), nullable=False),
        sa.Column(
            "file_hash",
            sa.VARCHAR(length=255),
            server_default=sa.text("''::character varying"),
            nullable=False,
        ),
        sa.Column(
            "source", sa.VARCHAR(length=1000), server_default=sa.text("''::character varying"), nullable=False
        ),
        sa.Column("chunks", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("chars", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("indexed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("ingestion_key", sa.TEXT(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="ingestion_registry_pkey"),
    )
    op.create_table(
        "regulatory_acts",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("act_type", sa.VARCHAR(length=32), nullable=False),
        sa.Column("act_number", sa.VARCHAR(length=100), nullable=True),
        sa.Column("title", sa.TEXT(), nullable=False),
        sa.Column("issuing_authority", sa.VARCHAR(length=255), nullable=True),
        sa.Column("act_date", sa.DATE(), nullable=True),
        sa.Column(
            "visibility_scope",
            sa.VARCHAR(length=100),
            server_default=sa.text("'internal_public'::character varying"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="regulatory_acts_pkey"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("email", sa.VARCHAR(length=255), nullable=False),
        sa.Column("hashed_password", sa.VARCHAR(length=255), nullable=False),
        sa.Column(
            "role", sa.VARCHAR(length=16), server_default=sa.text("'user'::character varying"), nullable=False
        ),
        sa.Column(
            "kind",
            sa.VARCHAR(length=16),
            server_default=sa.text("'internal'::character varying"),
            nullable=False,
        ),
        sa.Column("is_active", sa.BOOLEAN(), server_default=sa.text("true"), nullable=False),
        sa.CheckConstraint(
            "NOT (kind::text = 'client'::text AND (role IN ('admin', 'curator')))",
            name="chk_client_not_privileged",
        ),
        sa.CheckConstraint("kind IN ('internal', 'client')", name="users_kind_check"),
        sa.CheckConstraint("role IN ('admin', 'curator', 'user')", name="users_role_check"),
        sa.PrimaryKeyConstraint("id", name="users_pkey"),
        sa.UniqueConstraint("email", name="users_email_key"),
    )
    op.create_table(
        "vector_store_outbox",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("operation", sa.VARCHAR(length=30), nullable=False),
        sa.Column(
            "aggregate_type",
            sa.VARCHAR(length=20),
            server_default=sa.text("'document'::character varying"),
            nullable=False,
        ),
        sa.Column("aggregate_id", sa.INTEGER(), nullable=False),
        sa.Column("payload", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "status",
            sa.VARCHAR(length=20),
            server_default=sa.text("'pending'::character varying"),
            nullable=False,
        ),
        sa.Column("attempts", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_attempts", sa.INTEGER(), server_default=sa.text("8"), nullable=False),
        sa.Column(
            "next_attempt_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_error", sa.TEXT(), nullable=True),
        sa.Column("locked_by", sa.VARCHAR(length=64), nullable=True),
        sa.Column("locked_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "operation IN ('upsert_chunks', 'delete_by_document', 'delete_chunks', 'update_metadata', 'set_document_id')",
            name="vector_store_outbox_operation_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'in_progress', 'done', 'failed', 'dead_letter')",
            name="vector_store_outbox_status_check",
        ),
        sa.PrimaryKeyConstraint("id", name="vector_store_outbox_pkey"),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.INTEGER(), nullable=False),
        sa.Column("key_hash", sa.VARCHAR(length=64), nullable=False),
        sa.Column("key_prefix", sa.VARCHAR(length=20), nullable=False),
        sa.Column("name", sa.VARCHAR(length=255), nullable=True),
        sa.Column("revoked_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_used_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="api_keys_user_id_fkey", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="api_keys_pkey"),
        sa.UniqueConstraint("key_hash", name="api_keys_key_hash_key"),
    )
    op.create_table(
        "benchmark_questions",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("question", sa.TEXT(), nullable=False),
        sa.Column("expected_answer", sa.TEXT(), nullable=True),
        sa.Column("source_hint", sa.TEXT(), nullable=True),
        sa.Column("tags", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "dataset",
            sa.VARCHAR(length=100),
            server_default=sa.text("'main'::character varying"),
            nullable=False,
        ),
        sa.Column("is_active", sa.BOOLEAN(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_by", sa.INTEGER(), nullable=True),
        sa.Column("notes", sa.TEXT(), nullable=True),
        sa.Column("annotations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "annotations IS NULL OR jsonb_typeof(annotations) = 'object'::text",
            name="chk_benchmark_questions_annotations_object",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name="benchmark_questions_created_by_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_questions_pkey"),
    )
    op.create_table(
        "benchmark_sweeps",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.VARCHAR(length=20),
            server_default=sa.text("'pending'::character varying"),
            nullable=False,
        ),
        sa.Column("strategy", sa.VARCHAR(length=50), nullable=False),
        sa.Column("search_space", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("objective_weights", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "dataset",
            sa.VARCHAR(length=100),
            server_default=sa.text("'main'::character varying"),
            nullable=False,
        ),
        sa.Column("top_n_llm", sa.INTEGER(), server_default=sa.text("3"), nullable=False),
        sa.Column("job_id", sa.INTEGER(), nullable=True),
        sa.Column("total_configs", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("evaluated_configs", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("best_run_id", sa.INTEGER(), nullable=True),
        sa.Column("version", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("judge_model", sa.VARCHAR(length=255), nullable=True),
        sa.Column(
            "evaluation_mode",
            sa.VARCHAR(length=20),
            server_default=sa.text("'fast'::character varying"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "evaluation_mode::text <> 'full'::text OR strategy::text = 'grid'::text",
            name="benchmark_sweeps_full_grid_check",
        ),
        sa.CheckConstraint(
            "evaluation_mode IN ('fast', 'full')", name="benchmark_sweeps_evaluation_mode_check"
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name="benchmark_sweeps_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["background_jobs.id"], name="benchmark_sweeps_job_id_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_sweeps_pkey"),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.INTEGER(), nullable=False),
        sa.Column("summary", sa.TEXT(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="conversations_user_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="conversations_pkey"),
    )
    op.create_table(
        "cost_usage",
        sa.Column("operation_id", sa.TEXT(), nullable=False),
        sa.Column("principal", sa.TEXT(), nullable=False),
        sa.Column("provider", sa.TEXT(), nullable=False),
        sa.Column("model", sa.TEXT(), nullable=False),
        sa.Column("operation", sa.TEXT(), nullable=False),
        sa.Column("provider_response_id", sa.TEXT(), nullable=True),
        sa.Column("reconciliation_evidence", sa.TEXT(), nullable=True),
        sa.Column("rate_version", sa.INTEGER(), nullable=False),
        sa.Column("reserved_units", sa.BIGINT(), nullable=False),
        sa.Column("actual_units", sa.BIGINT(), nullable=True),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("settled_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("request_fingerprint", sa.TEXT(), nullable=True),
        sa.Column("billable_units", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("cost_source", sa.TEXT(), nullable=True),
        sa.Column("dispatch_state", sa.TEXT(), server_default=sa.text("'reserved'::text"), nullable=False),
        sa.CheckConstraint(
            "dispatch_state = ANY (ARRAY['reserved'::text, 'sending'::text, 'settled'::text])",
            name="ck_cost_usage_dispatch",
        ),
        sa.CheckConstraint("(actual_units IS NULL) = (settled_at IS NULL)", name="cost_usage_check1"),
        sa.CheckConstraint(
            "reserved_units >= 0 AND (actual_units IS NULL OR actual_units >= 0)", name="cost_usage_check"
        ),
        sa.ForeignKeyConstraint(
            ["provider", "model", "rate_version"],
            ["cost_rates.provider", "cost_rates.model", "cost_rates.version"],
            name="cost_usage_provider_model_rate_version_fkey",
        ),
        sa.PrimaryKeyConstraint("operation_id", name="cost_usage_pkey"),
    )
    op.create_table(
        "curator_group_assignments",
        sa.Column("curator_id", sa.INTEGER(), nullable=False),
        sa.Column("group_id", sa.INTEGER(), nullable=False),
        sa.Column("assigned_by", sa.INTEGER(), nullable=True),
        sa.Column(
            "assigned_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["assigned_by"],
            ["users.id"],
            name="curator_group_assignments_assigned_by_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["curator_id"], ["users.id"], name="curator_group_assignments_curator_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="curator_group_assignments_group_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("curator_id", "group_id", name="curator_group_assignments_pkey"),
    )
    op.create_table(
        "curator_user_assignments",
        sa.Column("curator_id", sa.INTEGER(), nullable=False),
        sa.Column("target_user_id", sa.INTEGER(), nullable=False),
        sa.Column("assigned_by", sa.INTEGER(), nullable=True),
        sa.Column(
            "assigned_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("curator_id <> target_user_id", name="chk_curator_not_self_assign"),
        sa.ForeignKeyConstraint(
            ["assigned_by"],
            ["users.id"],
            name="curator_user_assignments_assigned_by_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["curator_id"], ["users.id"], name="curator_user_assignments_curator_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_user_id"],
            ["users.id"],
            name="curator_user_assignments_target_user_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("curator_id", "target_user_id", name="curator_user_assignments_pkey"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("filename", sa.VARCHAR(length=255), nullable=False),
        sa.Column("source_path", sa.TEXT(), server_default=sa.text("''::text"), nullable=False),
        sa.Column("visibility", sa.VARCHAR(length=20), nullable=False),
        sa.Column("owner_id", sa.INTEGER(), nullable=True),
        sa.Column("group_id", sa.INTEGER(), nullable=True),
        sa.Column(
            "status",
            sa.VARCHAR(length=16),
            server_default=sa.text("'pending'::character varying"),
            nullable=False,
        ),
        sa.Column("error_message", sa.TEXT(), nullable=True),
        sa.Column("warning_message", sa.TEXT(), nullable=True),
        sa.Column("chunks", sa.INTEGER(), nullable=True),
        sa.Column("chars", sa.INTEGER(), nullable=True),
        sa.Column("indexed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "doc_domain",
            sa.VARCHAR(length=32),
            server_default=sa.text("'general'::character varying"),
            nullable=False,
        ),
        sa.Column(
            "source_type",
            sa.VARCHAR(length=16),
            server_default=sa.text("'file'::character varying"),
            nullable=False,
        ),
        sa.Column("has_manual_edits", sa.BOOLEAN(), server_default=sa.text("false"), nullable=False),
        sa.Column("quality_score", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column("version", sa.INTEGER(), server_default=sa.text("0"), nullable=False),
        sa.Column("version_group_id", sa.INTEGER(), nullable=True),
        sa.Column("ingestion_key", sa.TEXT(), nullable=True),
        sa.CheckConstraint("source_type IN ('file', 'manual')", name="documents_source_type_check"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'indexing', 'done', 'failed')", name="documents_status_check"
        ),
        sa.CheckConstraint(
            "visibility IN ('internal_public', 'internal_group', 'internal_private', 'client_private')",
            name="documents_visibility_check",
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="documents_group_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="documents_owner_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_group_id"], ["documents.id"], name="fk_documents_version_group", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="documents_pkey"),
    )
    op.create_table(
        "user_groups",
        sa.Column("user_id", sa.INTEGER(), nullable=False),
        sa.Column("group_id", sa.INTEGER(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="user_groups_group_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="user_groups_user_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "group_id", name="user_groups_pkey"),
    )
    op.create_table(
        "act_versions",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("act_id", sa.INTEGER(), nullable=True),
        sa.Column("document_id", sa.INTEGER(), nullable=False),
        sa.Column("effective_from", sa.DATE(), nullable=True),
        sa.Column("effective_to", sa.DATE(), nullable=True),
        sa.Column("is_current", sa.BOOLEAN(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "date_source",
            sa.VARCHAR(length=32),
            server_default=sa.text("'extracted'::character varying"),
            nullable=False,
        ),
        sa.Column("date_confidence", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column("verified_by", sa.INTEGER(), nullable=True),
        sa.Column("verified_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["act_id"], ["regulatory_acts.id"], name="act_versions_act_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name="act_versions_document_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["verified_by"], ["users.id"], name="act_versions_verified_by_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="act_versions_pkey"),
    )
    op.create_table(
        "benchmark_checkpoints",
        sa.Column("sweep_id", sa.INTEGER(), nullable=False),
        sa.Column("key", sa.VARCHAR(length=512), nullable=False),
        sa.Column("payload", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["sweep_id"],
            ["benchmark_sweeps.id"],
            name="benchmark_checkpoints_sweep_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("sweep_id", "key", name="benchmark_checkpoints_pkey"),
    )
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("sweep_id", sa.INTEGER(), nullable=True),
        sa.Column("config_json", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("summary_metrics", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "duration_sec",
            sa.DOUBLE_PRECISION(precision=53),
            server_default=sa.text("'0'::double precision"),
            nullable=False,
        ),
        sa.Column("llm_evaluated", sa.BOOLEAN(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "dataset",
            sa.VARCHAR(length=100),
            server_default=sa.text("'main'::character varying"),
            nullable=False,
        ),
        sa.Column("per_question_results", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("filename", sa.VARCHAR(length=255), nullable=True),
        sa.Column("job_id", sa.INTEGER(), nullable=True),
        sa.Column("legacy_hash", sa.VARCHAR(length=64), nullable=True),
        sa.Column(
            "artifacts",
            postgresql.JSON(astext_type=sa.Text()),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["sweep_id"], ["benchmark_sweeps.id"], name="benchmark_runs_sweep_id_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_runs_pkey"),
        sa.UniqueConstraint("job_id", name="uq_benchmark_runs_job_id"),
        sa.UniqueConstraint("legacy_hash", name="uq_benchmark_runs_legacy_hash"),
    )
    op.create_table(
        "chat_logs",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.INTEGER(), nullable=True),
        sa.Column("conversation_id", sa.INTEGER(), nullable=True),
        sa.Column("question", sa.TEXT(), nullable=False),
        sa.Column("answer", sa.TEXT(), nullable=False),
        sa.Column("sources", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("latency_ms", sa.INTEGER(), nullable=True),
        sa.Column("model_used", sa.VARCHAR(length=100), nullable=True),
        sa.Column("breadth", sa.VARCHAR(length=20), nullable=True),
        sa.Column("domain", sa.VARCHAR(length=20), nullable=True),
        sa.Column("retrieval_count", sa.INTEGER(), nullable=True),
        sa.Column("reranker_score", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column("input_tokens", sa.INTEGER(), nullable=True),
        sa.Column("output_tokens", sa.INTEGER(), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="chat_logs_conversation_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="chat_logs_user_id_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="chat_logs_pkey"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("conversation_id", sa.INTEGER(), nullable=True),
        sa.Column("role", sa.VARCHAR(length=16), nullable=False),
        sa.Column("content", sa.TEXT(), nullable=False),
        sa.Column("sources", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="messages_role_check"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="messages_conversation_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="messages_pkey"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.INTEGER(), nullable=False),
        sa.Column("chunk_index", sa.INTEGER(), nullable=False),
        sa.Column("content", sa.TEXT(), nullable=False),
        sa.Column(
            "filename",
            sa.VARCHAR(length=255),
            server_default=sa.text("''::character varying"),
            nullable=False,
        ),
        sa.Column("visibility", sa.VARCHAR(length=20), nullable=False),
        sa.Column("owner_id", sa.INTEGER(), nullable=True),
        sa.Column("group_id", sa.INTEGER(), nullable=True),
        sa.Column(
            "doc_domain",
            sa.VARCHAR(length=32),
            server_default=sa.text("'general'::character varying"),
            nullable=False,
        ),
        sa.Column("edited_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("edited_by", sa.INTEGER(), nullable=True),
        sa.Column("manual", sa.BOOLEAN(), server_default=sa.text("false"), nullable=False),
        sa.Column("content_hash", sa.VARCHAR(length=16), nullable=True),
        sa.Column("domain_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("effective_from", sa.DATE(), nullable=True),
        sa.Column("effective_to", sa.DATE(), nullable=True),
        sa.Column("is_current", sa.BOOLEAN(), server_default=sa.text("true"), nullable=False),
        sa.Column("act_version_id", sa.INTEGER(), nullable=True),
        sa.Column("section", sa.TEXT(), nullable=True),
        sa.Column("heading", sa.VARCHAR(length=500), nullable=True),
        sa.Column("heading_level", sa.INTEGER(), nullable=True),
        sa.Column("content_type", sa.VARCHAR(length=32), nullable=True),
        sa.Column("doc_title", sa.VARCHAR(length=500), nullable=True),
        sa.Column("doc_type", sa.VARCHAR(length=64), nullable=True),
        sa.Column(
            "context_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(context_metadata) = 'object'::text", name="chk_chunks_context_metadata_object"
        ),
        sa.ForeignKeyConstraint(
            ["act_version_id"], ["act_versions.id"], name="chunks_act_version_id_fkey", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name="chunks_document_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["edited_by"], ["users.id"], name="chunks_edited_by_fkey", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="chunks_group_id_fkey", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name="chunks_owner_id_fkey", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="chunks_pkey"),
        sa.UniqueConstraint("document_id", "chunk_index", "id", name="ux_chunks_document_chunk_index_id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="ux_chunks_document_index"),
    )


def create_runtime_indexes() -> None:
    op.create_index("idx_background_jobs_creation_date", "background_jobs", ["creation_date"], unique=False)
    op.create_index("idx_background_jobs_heartbeat", "background_jobs", ["heartbeat_at"], unique=False)
    op.create_index("idx_background_jobs_job_type", "background_jobs", ["job_type"], unique=False)
    op.create_index("idx_background_jobs_status", "background_jobs", ["status"], unique=False)
    op.create_index("idx_config_parameters_category", "config_parameters", ["category"], unique=False)
    op.create_index(
        "idx_config_parameters_domain_key",
        "config_parameters",
        ["domain_key"],
        unique=False,
        postgresql_where="(domain_key IS NOT NULL)",
    )
    op.create_index("idx_ingestion_registry_key", "ingestion_registry", ["ingestion_key"], unique=True)
    op.create_index(
        "ux_regulatory_acts_identity",
        "regulatory_acts",
        [
            "act_type",
            "act_number",
            "visibility_scope",
            sa.literal_column("COALESCE(act_date, '0001-01-01'::date)"),
        ],
        unique=True,
    )
    op.create_index(
        "idx_outbox_aggregate", "vector_store_outbox", ["aggregate_type", "aggregate_id"], unique=False
    )
    op.create_index(
        "idx_outbox_dispatch",
        "vector_store_outbox",
        ["status", "next_attempt_at"],
        unique=False,
        postgresql_where="status IN ('pending', 'failed')",
    )
    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"], unique=False)
    op.create_index("idx_benchmark_questions_dataset", "benchmark_questions", ["dataset"], unique=False)
    op.create_index("idx_benchmark_questions_is_active", "benchmark_questions", ["is_active"], unique=False)
    op.create_index("idx_benchmark_sweeps_status", "benchmark_sweeps", ["status"], unique=False)
    op.create_index(
        "uq_benchmark_sweeps_one_active",
        "benchmark_sweeps",
        [sa.literal_column("(1)")],
        unique=True,
        postgresql_where="status IN ('pending', 'running')",
    )
    op.create_index("idx_conversations_user", "conversations", ["user_id"], unique=False)
    op.create_index("ix_cost_usage_principal_window", "cost_usage", ["principal", "settled_at"], unique=False)
    op.create_index(
        "idx_curator_user_assignments_target", "curator_user_assignments", ["target_user_id"], unique=False
    )


def create_document_indexes() -> None:
    op.create_index("idx_documents_doc_domain", "documents", ["doc_domain"], unique=False)
    op.create_index("idx_documents_group", "documents", ["group_id"], unique=False)
    op.create_index("idx_documents_owner", "documents", ["owner_id"], unique=False)
    op.create_index("idx_documents_source_type", "documents", ["source_type"], unique=False)
    op.create_index("idx_documents_status", "documents", ["status"], unique=False)
    op.create_index(
        "idx_documents_version_group",
        "documents",
        ["version_group_id"],
        unique=False,
        postgresql_where="(version_group_id IS NOT NULL)",
    )
    op.create_index("idx_documents_visibility", "documents", ["visibility"], unique=False)
    op.create_index(
        "ux_documents_active_slot",
        "documents",
        [sa.literal_column("COALESCE(owner_id, 0)"), sa.literal_column("COALESCE(group_id, 0)"), "filename"],
        unique=True,
        postgresql_where="ingestion_key IS NULL AND status IN ('pending', 'processing', 'indexing', 'done', 'failed')",
    )
    op.create_index(
        "ux_documents_ingestion_slot",
        "documents",
        [
            "ingestion_key",
            sa.literal_column("COALESCE(owner_id, 0)"),
            sa.literal_column("COALESCE(group_id, 0)"),
        ],
        unique=True,
        postgresql_where="ingestion_key IS NOT NULL AND status IN ('pending', 'processing', 'indexing', 'done', 'failed')",
    )
    op.create_index("idx_act_versions_act_id", "act_versions", ["act_id"], unique=False)
    op.create_index(
        "ux_act_versions_one_current", "act_versions", ["act_id"], unique=True, postgresql_where="is_current"
    )
    op.create_index("idx_benchmark_runs_dataset", "benchmark_runs", ["dataset"], unique=False)
    op.create_index("idx_benchmark_runs_sweep_id", "benchmark_runs", ["sweep_id"], unique=False)
    op.create_index("idx_chat_logs_creation_date", "chat_logs", ["creation_date"], unique=False)
    op.create_index("idx_chat_logs_domain", "chat_logs", ["domain"], unique=False)
    op.create_index("idx_chat_logs_user_id", "chat_logs", ["user_id"], unique=False)
    op.create_index("idx_messages_conv", "messages", ["conversation_id"], unique=False)
    op.create_index(
        "idx_chunks_act_version_id",
        "chunks",
        ["act_version_id"],
        unique=False,
        postgresql_where="(act_version_id IS NOT NULL)",
    )
    op.create_index("idx_chunks_content_hash", "chunks", ["content_hash"], unique=False)
    op.create_index(
        "ix_chunks_content_trgm",
        "chunks",
        ["content"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"content": "public.gin_trgm_ops"},
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_chunks_edited_by",
        "chunks",
        ["edited_by"],
        unique=False,
        postgresql_where="(edited_by IS NOT NULL)",
    )
    op.create_index("ix_chunks_group_id", "chunks", ["group_id"], unique=False)
    op.create_index("ix_chunks_owner_id", "chunks", ["owner_id"], unique=False)
    op.create_index("ix_chunks_visibility", "chunks", ["visibility"], unique=False)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")
    create_tables()
    create_runtime_indexes()
    create_document_indexes()
    op.create_foreign_key(
        "fk_benchmark_sweeps_best_run",
        "benchmark_sweeps",
        "benchmark_runs",
        ["best_run_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_benchmark_sweeps_best_run", "benchmark_sweeps", type_="foreignkey")
    op.drop_table("chunks")
    op.drop_table("messages")
    op.drop_table("chat_logs")
    op.drop_table("benchmark_runs")
    op.drop_table("benchmark_checkpoints")
    op.drop_table("act_versions")
    op.drop_table("user_groups")
    op.drop_table("documents")
    op.drop_table("curator_user_assignments")
    op.drop_table("curator_group_assignments")
    op.drop_table("cost_usage")
    op.drop_table("conversations")
    op.drop_table("benchmark_sweeps")
    op.drop_table("benchmark_questions")
    op.drop_table("api_keys")
    op.drop_table("vector_store_outbox")
    op.drop_table("users")
    op.drop_table("regulatory_acts")
    op.drop_table("ingestion_registry")
    op.drop_table("groups")
    op.drop_table("cost_rates")
    op.drop_table("corpus_processes")
    op.drop_table("corpus_control")
    op.drop_table("config_parameters")
    op.drop_table("background_jobs")
