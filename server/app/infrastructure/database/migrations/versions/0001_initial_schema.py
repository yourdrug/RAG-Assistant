"""Initial schema for empty databases; replaces the historical migration chain.

Existing installations require a database reset. The shared pg_trgm extension
is retained on downgrade.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:  # noqa: PLR0915 - one generated baseline
    """Create the schema and the starting knowledge space."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")
    op.create_table(
        "background_jobs",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('background_jobs_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("job_type", sa.VARCHAR(length=50), autoincrement=False, nullable=False),
        sa.Column(
            "status",
            sa.VARCHAR(length=20),
            server_default=sa.text("'pending'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("related_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("request_id", sa.VARCHAR(length=32), autoincrement=False, nullable=True),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("error_message", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("heartbeat_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("dispatch_payload", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column(
            "payload_version", sa.INTEGER(), server_default=sa.text("1"), autoincrement=False, nullable=False
        ),
        sa.Column("deduplication_key", sa.VARCHAR(length=128), autoincrement=False, nullable=True),
        sa.Column("attempt", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False),
        sa.Column("attempt_id", sa.VARCHAR(length=36), autoincrement=False, nullable=True),
        sa.Column("lease_until", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("next_attempt_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column(
            "dispatch_attempts",
            sa.INTEGER(),
            server_default=sa.text("0"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("progress", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name="background_jobs_status_check",
        ),
        sa.PrimaryKeyConstraint("id", name="background_jobs_pkey"),
        postgresql_ignore_search_path=False,
    )
    op.create_index("idx_background_jobs_creation_date", "background_jobs", ["creation_date"], unique=False)
    op.create_index("idx_background_jobs_heartbeat", "background_jobs", ["heartbeat_at"], unique=False)
    op.create_index("idx_background_jobs_job_type", "background_jobs", ["job_type"], unique=False)
    op.create_index(
        "idx_background_jobs_relay",
        "background_jobs",
        ["heartbeat_at", "id"],
        unique=False,
        postgresql_where="(((status)::text = 'pending'::text) AND (dispatch_payload IS NOT NULL))",
    )
    op.create_index("idx_background_jobs_status", "background_jobs", ["status"], unique=False)
    op.create_index(
        "ux_background_jobs_active_dedup",
        "background_jobs",
        ["deduplication_key"],
        unique=True,
        postgresql_where="((deduplication_key IS NOT NULL) AND (status IN ('pending', 'running')))",
    )
    op.create_table(
        "config_parameters",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('config_parameters_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("key", sa.VARCHAR(length=100), autoincrement=False, nullable=False),
        sa.Column("value", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "value_type",
            sa.VARCHAR(length=20),
            server_default=sa.text("'str'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("category", sa.VARCHAR(length=50), autoincrement=False, nullable=False),
        sa.Column("description", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("min_value", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("max_value", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("allowed_values", postgresql.JSON, autoincrement=False, nullable=True),
        sa.Column("domain_key", sa.VARCHAR(length=32), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint("id", name="config_parameters_pkey"),
        sa.UniqueConstraint(
            "key", "domain_key", name="ux_config_parameters_key_domain", postgresql_nulls_not_distinct=True
        ),
    )
    op.create_index("idx_config_parameters_category", "config_parameters", ["category"], unique=False)
    op.create_index(
        "idx_config_parameters_domain_key",
        "config_parameters",
        ["domain_key"],
        unique=False,
        postgresql_where="(domain_key IS NOT NULL)",
    )
    op.create_table(
        "corpus_control",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('corpus_control_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("phase", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "write_epoch",
            sa.BIGINT(),
            server_default=sa.text("'0'::bigint"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("active", postgresql.JSONB, autoincrement=False, nullable=False),
        sa.Column("candidate", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("digest", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("owner", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("error", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
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
        sa.Column("process_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("role", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("prepared_generation", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("applied_generation", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("error", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column(
            "heartbeat_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.CheckConstraint(
            "role = ANY (ARRAY['api'::text, 'worker'::text, 'cli'::text])", name="corpus_processes_role_check"
        ),
        sa.PrimaryKeyConstraint("process_id", name="corpus_processes_pkey"),
    )
    op.create_table(
        "groups",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('groups_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("name", sa.VARCHAR(length=100), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name="groups_pkey"),
        sa.UniqueConstraint("name", name="groups_name_key"),
        postgresql_ignore_search_path=False,
    )
    op.create_table(
        "ingestion_registry",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('ingestion_registry_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("filename", sa.VARCHAR(length=500), autoincrement=False, nullable=False),
        sa.Column(
            "file_hash",
            sa.VARCHAR(length=255),
            server_default=sa.text("''::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.VARCHAR(length=1000),
            server_default=sa.text("''::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("chunks", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False),
        sa.Column("chars", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False),
        sa.Column("indexed_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("ingestion_key", sa.TEXT(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name="ingestion_registry_pkey"),
    )
    op.create_index("idx_ingestion_registry_key", "ingestion_registry", ["ingestion_key"], unique=True)
    op.create_table(
        "knowledge_spaces",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('knowledge_spaces_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("name", sa.VARCHAR(length=100), autoincrement=False, nullable=False),
        sa.Column("pack_key", sa.VARCHAR(length=64), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name="knowledge_spaces_pkey"),
        sa.UniqueConstraint("name", name="knowledge_spaces_name_key"),
        postgresql_ignore_search_path=False,
    )
    op.create_table(
        "rag_config_snapshots",
        sa.Column("hash", sa.VARCHAR(length=64), autoincrement=False, nullable=False),
        sa.Column("data", postgresql.JSONB, autoincrement=False, nullable=False),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("hash", name="rag_config_snapshots_pkey"),
    )
    op.create_table(
        "regulatory_acts",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('regulatory_acts_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("act_type", sa.VARCHAR(length=32), autoincrement=False, nullable=False),
        sa.Column("act_number", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("title", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("issuing_authority", sa.VARCHAR(length=255), autoincrement=False, nullable=True),
        sa.Column("act_date", sa.DATE(), autoincrement=False, nullable=True),
        sa.Column(
            "visibility_scope",
            sa.VARCHAR(length=100),
            server_default=sa.text("'internal_public'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="regulatory_acts_pkey"),
    )
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
    op.create_table(
        "vector_store_outbox",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('vector_store_outbox_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("operation", sa.VARCHAR(length=30), autoincrement=False, nullable=False),
        sa.Column(
            "aggregate_type",
            sa.VARCHAR(length=20),
            server_default=sa.text("'document'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("aggregate_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("payload", postgresql.JSON, autoincrement=False, nullable=False),
        sa.Column(
            "status",
            sa.VARCHAR(length=20),
            server_default=sa.text("'pending'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("attempts", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False),
        sa.Column(
            "max_attempts", sa.INTEGER(), server_default=sa.text("8"), autoincrement=False, nullable=False
        ),
        sa.Column(
            "next_attempt_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("last_error", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("locked_by", sa.VARCHAR(length=64), autoincrement=False, nullable=True),
        sa.Column("locked_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("completed_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.CheckConstraint(
            "operation IN ('upsert_chunks', 'delete_by_document', 'delete_chunks', 'update_metadata', "
            "'set_document_id')",
            name="vector_store_outbox_operation_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'in_progress', 'done', 'failed', 'dead_letter')",
            name="vector_store_outbox_status_check",
        ),
        sa.PrimaryKeyConstraint("id", name="vector_store_outbox_pkey"),
    )
    op.create_index(
        "idx_outbox_aggregate", "vector_store_outbox", ["aggregate_type", "aggregate_id"], unique=False
    )
    op.create_index(
        "idx_outbox_dispatch",
        "vector_store_outbox",
        ["status", "next_attempt_at"],
        unique=False,
        postgresql_where="(status IN ('pending', 'failed'))",
    )
    op.create_table(
        "benchmark_sweeps",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('benchmark_sweeps_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.VARCHAR(length=20),
            server_default=sa.text("'pending'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("strategy", sa.VARCHAR(length=50), autoincrement=False, nullable=False),
        sa.Column("search_space", postgresql.JSON, autoincrement=False, nullable=False),
        sa.Column("objective_weights", postgresql.JSON, autoincrement=False, nullable=False),
        sa.Column(
            "dataset",
            sa.VARCHAR(length=100),
            server_default=sa.text("'main'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "top_n_llm", sa.INTEGER(), server_default=sa.text("3"), autoincrement=False, nullable=False
        ),
        sa.Column("job_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "total_configs", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False
        ),
        sa.Column(
            "evaluated_configs",
            sa.INTEGER(),
            server_default=sa.text("0"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("best_run_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("version", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False),
        sa.Column("judge_model", sa.VARCHAR(length=255), autoincrement=False, nullable=True),
        sa.Column(
            "evaluation_mode",
            sa.VARCHAR(length=20),
            server_default=sa.text("'fast'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.CheckConstraint(
            "evaluation_mode::text <> 'full'::text OR strategy::text = 'grid'::text",
            name="benchmark_sweeps_full_grid_check",
        ),
        sa.CheckConstraint(
            "evaluation_mode IN ('fast', 'full')",
            name="benchmark_sweeps_evaluation_mode_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'failed', 'cancelled')",
            name="benchmark_sweeps_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["background_jobs.id"], name="benchmark_sweeps_job_id_fkey", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="fk_benchmark_sweeps_space_id", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_sweeps_pkey"),
    )
    op.create_index("idx_benchmark_sweeps_status", "benchmark_sweeps", ["status"], unique=False)
    op.create_index(
        "uq_benchmark_sweeps_one_active",
        "benchmark_sweeps",
        [sa.literal_column("(1)")],
        unique=True,
        postgresql_where="(status IN ('pending', 'running'))",
    )
    op.create_table(
        "knowledge_space_upload_groups",
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("group_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["groups.id"],
            name="knowledge_space_upload_groups_group_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["space_id"],
            ["knowledge_spaces.id"],
            name="knowledge_space_upload_groups_space_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("space_id", "group_id", name="knowledge_space_upload_groups_pkey"),
    )
    op.create_table(
        "users",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('users_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("email", sa.VARCHAR(length=255), autoincrement=False, nullable=False),
        sa.Column("hashed_password", sa.VARCHAR(length=255), autoincrement=False, nullable=False),
        sa.Column(
            "role",
            sa.VARCHAR(length=16),
            server_default=sa.text("'user'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "kind",
            sa.VARCHAR(length=16),
            server_default=sa.text("'internal'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "is_active", sa.BOOLEAN(), server_default=sa.text("true"), autoincrement=False, nullable=False
        ),
        sa.Column("client_space_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.CheckConstraint(
            "NOT (kind::text = 'client'::text AND (role IN ('admin', 'curator')))",
            name="chk_client_not_privileged",
        ),
        sa.CheckConstraint(
            "client_space_id IS NULL OR kind::text = 'client'::text", name="chk_client_space_kind"
        ),
        sa.CheckConstraint(
            "kind IN ('internal', 'client')",
            name="users_kind_check",
        ),
        sa.CheckConstraint(
            "role IN ('admin', 'curator', 'user')",
            name="users_role_check",
        ),
        sa.ForeignKeyConstraint(
            ["client_space_id"], ["knowledge_spaces.id"], name="fk_users_client_space_id", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="users_pkey"),
        sa.UniqueConstraint("email", name="users_email_key"),
        postgresql_ignore_search_path=False,
    )
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('api_keys_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("key_hash", sa.VARCHAR(length=64), autoincrement=False, nullable=False),
        sa.Column("key_prefix", sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column("name", sa.VARCHAR(length=255), autoincrement=False, nullable=True),
        sa.Column("revoked_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("last_used_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="api_keys_user_id_fkey", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="api_keys_pkey"),
        sa.UniqueConstraint("key_hash", name="api_keys_key_hash_key"),
    )
    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"], unique=False)
    op.create_table(
        "benchmark_checkpoints",
        sa.Column("sweep_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("key", sa.VARCHAR(length=512), autoincrement=False, nullable=False),
        sa.Column("payload", postgresql.JSON, autoincrement=False, nullable=False),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
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
        "benchmark_questions",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('benchmark_questions_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("question", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("expected_answer", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("source_hint", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("tags", postgresql.JSON, autoincrement=False, nullable=True),
        sa.Column(
            "dataset",
            sa.VARCHAR(length=100),
            server_default=sa.text("'main'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "is_active", sa.BOOLEAN(), server_default=sa.text("true"), autoincrement=False, nullable=False
        ),
        sa.Column("created_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("notes", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("annotations", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("history", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("as_of_date", sa.DATE(), autoincrement=False, nullable=True),
        sa.Column("case_id", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("variant_of", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("question_style", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.CheckConstraint(
            "annotations IS NULL OR jsonb_typeof(annotations) = 'object'::text",
            name="chk_benchmark_questions_annotations_object",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name="benchmark_questions_created_by_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_questions_pkey"),
    )
    op.create_index("idx_benchmark_questions_dataset", "benchmark_questions", ["dataset"], unique=False)
    op.create_index("idx_benchmark_questions_is_active", "benchmark_questions", ["is_active"], unique=False)
    op.create_table(
        "benchmark_runs",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('benchmark_runs_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("sweep_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("config_json", postgresql.JSON, autoincrement=False, nullable=False),
        sa.Column("summary_metrics", postgresql.JSON, autoincrement=False, nullable=False),
        sa.Column(
            "duration_sec",
            sa.DOUBLE_PRECISION(precision=53),
            server_default=sa.text("'0'::double precision"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "llm_evaluated",
            sa.BOOLEAN(),
            server_default=sa.text("false"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "dataset",
            sa.VARCHAR(length=100),
            server_default=sa.text("'main'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("per_question_results", postgresql.JSON, autoincrement=False, nullable=True),
        sa.Column("filename", sa.VARCHAR(length=255), autoincrement=False, nullable=True),
        sa.Column("job_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("legacy_hash", sa.VARCHAR(length=64), autoincrement=False, nullable=True),
        sa.Column(
            "artifacts",
            postgresql.JSON,
            server_default=sa.text("'{}'::json"),
            autoincrement=False,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["sweep_id"], ["benchmark_sweeps.id"], name="benchmark_runs_sweep_id_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="benchmark_runs_pkey"),
        sa.UniqueConstraint("job_id", name="uq_benchmark_runs_job_id"),
        sa.UniqueConstraint("legacy_hash", name="uq_benchmark_runs_legacy_hash"),
        postgresql_ignore_search_path=False,
    )
    op.create_index("idx_benchmark_runs_dataset", "benchmark_runs", ["dataset"], unique=False)
    op.create_index("idx_benchmark_runs_sweep_id", "benchmark_runs", ["sweep_id"], unique=False)
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('conversations_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("summary", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="conversations_space_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="conversations_user_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="conversations_pkey"),
    )
    op.create_index("idx_conversations_user", "conversations", ["user_id"], unique=False)
    op.create_table(
        "curator_group_assignments",
        sa.Column("curator_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("group_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("assigned_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "assigned_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
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
        sa.Column("curator_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("target_user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("assigned_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "assigned_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
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
    op.create_index(
        "idx_curator_user_assignments_target", "curator_user_assignments", ["target_user_id"], unique=False
    )
    op.create_table(
        "documents",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('documents_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("filename", sa.VARCHAR(length=255), autoincrement=False, nullable=False),
        sa.Column(
            "source_path", sa.TEXT(), server_default=sa.text("''::text"), autoincrement=False, nullable=False
        ),
        sa.Column("visibility", sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column("owner_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("group_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "status",
            sa.VARCHAR(length=16),
            server_default=sa.text("'pending'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("error_message", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("warning_message", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("chunks", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("chars", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("indexed_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column(
            "doc_domain",
            sa.VARCHAR(length=32),
            server_default=sa.text("'general'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "source_type",
            sa.VARCHAR(length=16),
            server_default=sa.text("'file'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "has_manual_edits",
            sa.BOOLEAN(),
            server_default=sa.text("false"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("quality_score", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("version", sa.INTEGER(), server_default=sa.text("0"), autoincrement=False, nullable=False),
        sa.Column("version_group_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("ingestion_key", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.CheckConstraint(
            "source_type IN ('file', 'manual')",
            name="documents_source_type_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'indexing', 'done', 'failed')",
            name="documents_status_check",
        ),
        sa.CheckConstraint(
            "visibility IN ('internal_public', 'internal_group', 'internal_private', 'client_private', "
            "'client_public')",
            name="documents_visibility_check",
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="documents_group_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="documents_owner_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="fk_documents_space_id", ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["version_group_id"], ["documents.id"], name="fk_documents_version_group", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="documents_pkey"),
        postgresql_ignore_search_path=False,
    )
    op.create_index("idx_documents_doc_domain", "documents", ["doc_domain"], unique=False)
    op.create_index("idx_documents_group", "documents", ["group_id"], unique=False)
    op.create_index("idx_documents_owner", "documents", ["owner_id"], unique=False)
    op.create_index("idx_documents_source_type", "documents", ["source_type"], unique=False)
    op.create_index("idx_documents_space", "documents", ["space_id"], unique=False)
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
        [
            "space_id",
            sa.literal_column("COALESCE(owner_id, 0)"),
            sa.literal_column("COALESCE(group_id, 0)"),
            "filename",
        ],
        unique=True,
        postgresql_where="((ingestion_key IS NULL) AND (status IN ('pending', 'processing', 'indexing', "
        "'done', 'failed')))",
    )
    op.create_index(
        "ux_documents_ingestion_slot",
        "documents",
        [
            "space_id",
            "ingestion_key",
            sa.literal_column("COALESCE(owner_id, 0)"),
            sa.literal_column("COALESCE(group_id, 0)"),
        ],
        unique=True,
        postgresql_where="((ingestion_key IS NOT NULL) AND (status IN ('pending', 'processing', "
        "'indexing', 'done', 'failed')))",
    )
    op.create_table(
        "user_groups",
        sa.Column("user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("group_id", sa.INTEGER(), autoincrement=False, nullable=False),
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
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('act_versions_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("act_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("document_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("effective_from", sa.DATE(), autoincrement=False, nullable=True),
        sa.Column("effective_to", sa.DATE(), autoincrement=False, nullable=True),
        sa.Column(
            "is_current", sa.BOOLEAN(), server_default=sa.text("true"), autoincrement=False, nullable=False
        ),
        sa.Column(
            "date_source",
            sa.VARCHAR(length=32),
            server_default=sa.text("'extracted'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("date_confidence", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("verified_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("verified_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
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
        sa.UniqueConstraint("document_id", name="ux_act_versions_document_id"),
    )
    op.create_index("idx_act_versions_act_id", "act_versions", ["act_id"], unique=False)
    op.create_index(
        "ux_act_versions_one_current", "act_versions", ["act_id"], unique=True, postgresql_where="is_current"
    )
    op.create_table(
        "chunks",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('chunks_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("document_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("chunk_index", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("content", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "filename",
            sa.VARCHAR(length=255),
            server_default=sa.text("''::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("visibility", sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column("owner_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("group_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "doc_domain",
            sa.VARCHAR(length=32),
            server_default=sa.text("'general'::character varying"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("edited_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("edited_by", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "manual", sa.BOOLEAN(), server_default=sa.text("false"), autoincrement=False, nullable=False
        ),
        sa.Column("content_hash", sa.VARCHAR(length=16), autoincrement=False, nullable=True),
        sa.Column("domain_metadata", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("section", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("heading", sa.VARCHAR(length=500), autoincrement=False, nullable=True),
        sa.Column("heading_level", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("content_type", sa.VARCHAR(length=32), autoincrement=False, nullable=True),
        sa.Column("doc_title", sa.VARCHAR(length=500), autoincrement=False, nullable=True),
        sa.Column("doc_type", sa.VARCHAR(length=64), autoincrement=False, nullable=True),
        sa.Column(
            "context_metadata",
            postgresql.JSONB,
            server_default=sa.text("'{}'::jsonb"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("embed_text", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column(
            "pack_metadata",
            postgresql.JSONB,
            server_default=sa.text("'{}'::jsonb"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.CheckConstraint(
            "jsonb_typeof(context_metadata) = 'object'::text", name="chk_chunks_context_metadata_object"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(pack_metadata) = 'object'::text", name="chk_chunks_pack_metadata_object"
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
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="fk_chunks_space_id", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="chunks_pkey"),
        sa.UniqueConstraint("document_id", "chunk_index", "id", name="ux_chunks_document_chunk_index_id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="ux_chunks_document_index"),
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
    op.create_index("ix_chunks_space_id", "chunks", ["space_id"], unique=False)
    op.create_index("ix_chunks_visibility", "chunks", ["visibility"], unique=False)
    op.create_table(
        "document_nodes",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('document_nodes_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("document_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("node_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("parent_id", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("kind", sa.VARCHAR(length=32), autoincrement=False, nullable=False),
        sa.Column("label", sa.VARCHAR(length=500), autoincrement=False, nullable=False),
        sa.Column("number", sa.VARCHAR(length=64), autoincrement=False, nullable=True),
        sa.Column("depth", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("chunk_start", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("chunk_end", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "pages",
            postgresql.JSONB,
            server_default=sa.text("'[]'::jsonb"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("char_count", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("text_hash", sa.VARCHAR(length=16), autoincrement=False, nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name="document_nodes_document_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="document_nodes_pkey"),
        sa.UniqueConstraint("document_id", "node_id", name="ux_document_nodes_document_node"),
    )
    op.create_table(
        "messages",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('messages_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("conversation_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("role", sa.VARCHAR(length=16), autoincrement=False, nullable=False),
        sa.Column("content", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("sources", postgresql.JSON, autoincrement=False, nullable=True),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name="messages_role_check",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="messages_conversation_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="messages_pkey"),
    )
    op.create_index("idx_messages_conv", "messages", ["conversation_id"], unique=False)
    op.create_table(
        "chat_logs",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('chat_logs_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("user_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("conversation_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("question", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("answer", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("sources", postgresql.JSON, autoincrement=False, nullable=True),
        sa.Column("latency_ms", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("model_used", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("breadth", sa.VARCHAR(length=20), autoincrement=False, nullable=True),
        sa.Column("domain", sa.VARCHAR(length=20), autoincrement=False, nullable=True),
        sa.Column("input_tokens", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("output_tokens", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("message_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("space_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("pack_key", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("request_id", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("depth_requested", sa.VARCHAR(length=20), autoincrement=False, nullable=True),
        sa.Column("as_of_date", sa.DATE(), autoincrement=False, nullable=True),
        sa.Column("outcome", sa.VARCHAR(length=30), autoincrement=False, nullable=False),
        sa.Column("error_code", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("condensed_query", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("sub_queries", postgresql.JSONB, autoincrement=False, nullable=True),
        sa.Column("refined_query", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("confidence", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("sources_count", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("candidates_count", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("top_rerank_score", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("aux_tokens", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("ttft_ms", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("retrieval_ms", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("rerank_ms", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("generation_ms", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("config_hash", sa.VARCHAR(length=64), autoincrement=False, nullable=True),
        sa.Column("config_version", sa.VARCHAR(length=100), autoincrement=False, nullable=True),
        sa.Column("evidence_profile", sa.VARCHAR(length=200), autoincrement=False, nullable=True),
        sa.ForeignKeyConstraint(
            ["config_hash"],
            ["rag_config_snapshots.hash"],
            name="fk_chat_logs_config_hash",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="chat_logs_conversation_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["messages.id"], name="fk_chat_logs_message_id", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["space_id"], ["knowledge_spaces.id"], name="fk_chat_logs_space_id", ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="chat_logs_user_id_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="chat_logs_pkey"),
    )
    op.create_index("idx_chat_logs_config_hash", "chat_logs", ["config_hash"], unique=False)
    op.create_index("idx_chat_logs_creation_date", "chat_logs", ["creation_date"], unique=False)
    op.create_index("idx_chat_logs_date_outcome", "chat_logs", ["creation_date", "outcome"], unique=False)
    op.create_index("idx_chat_logs_domain", "chat_logs", ["domain"], unique=False)
    op.create_index("idx_chat_logs_message_id", "chat_logs", ["message_id"], unique=True)
    op.create_index("idx_chat_logs_outcome", "chat_logs", ["outcome"], unique=False)
    op.create_index("idx_chat_logs_space_date", "chat_logs", ["space_id", "creation_date"], unique=False)
    op.create_index("idx_chat_logs_user_id", "chat_logs", ["user_id"], unique=False)
    op.create_table(
        "message_feedback",
        sa.Column(
            "id",
            sa.INTEGER(),
            server_default=sa.text("nextval('message_feedback_id_seq'::regclass)"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("message_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("user_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("rating", sa.VARCHAR(length=8), autoincrement=False, nullable=False),
        sa.Column(
            "reasons",
            postgresql.JSONB,
            server_default=sa.text("'[]'::jsonb"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("comment", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column(
            "creation_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "update_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.CheckConstraint(
            "rating::text = 'down'::text OR reasons = '[]'::jsonb", name="ck_message_feedback_reasons_down"
        ),
        sa.CheckConstraint(
            "rating IN ('up', 'down')",
            name="ck_message_feedback_rating",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["messages.id"], name="message_feedback_message_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="message_feedback_user_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="message_feedback_pkey"),
        sa.UniqueConstraint("message_id", "user_id", name="uq_message_feedback_message_user"),
    )
    op.create_table(
        "chat_log_reviews",
        sa.Column("chat_log_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("status", sa.VARCHAR(length=30), autoincrement=False, nullable=False),
        sa.Column("note", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("reviewer_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("benchmark_question_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "update_date",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('new', 'benchmark_candidate', 'in_benchmark', 'content_gap', 'pipeline_issue', "
            "'wont_fix')",
            name="ck_chat_log_reviews_status",
        ),
        sa.ForeignKeyConstraint(
            ["benchmark_question_id"],
            ["benchmark_questions.id"],
            name="chat_log_reviews_benchmark_question_id_fkey",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["chat_log_id"], ["chat_logs.id"], name="chat_log_reviews_chat_log_id_fkey", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"], ["users.id"], name="chat_log_reviews_reviewer_id_fkey", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("chat_log_id", name="chat_log_reviews_pkey"),
    )
    op.create_index("idx_chat_log_reviews_status", "chat_log_reviews", ["status"], unique=False)
    op.create_table(
        "chat_log_traces",
        sa.Column("chat_log_id", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("schema_version", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("data", postgresql.JSONB, autoincrement=False, nullable=False),
        sa.CheckConstraint("octet_length(data::text) <= 65536", name="ck_chat_log_trace_size"),
        sa.ForeignKeyConstraint(
            ["chat_log_id"], ["chat_logs.id"], name="chat_log_traces_chat_log_id_fkey", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("chat_log_id", name="chat_log_traces_pkey"),
    )
    # benchmark_sweeps and benchmark_runs reference each other.
    op.create_foreign_key(
        "fk_benchmark_sweeps_best_run",
        "benchmark_sweeps",
        "benchmark_runs",
        ["best_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.execute(
        "INSERT INTO knowledge_spaces (creation_date, name, pack_key) VALUES (NOW(), 'Общее', 'generic')"
    )


def downgrade() -> None:  # noqa: PLR0915 - one generated baseline
    """Drop the schema."""
    op.drop_constraint("fk_benchmark_sweeps_best_run", "benchmark_sweeps", type_="foreignkey")
    op.drop_table("chat_log_traces")
    op.drop_index("idx_chat_log_reviews_status", table_name="chat_log_reviews")
    op.drop_table("chat_log_reviews")
    op.drop_table("message_feedback")
    op.drop_index("idx_chat_logs_user_id", table_name="chat_logs")
    op.drop_index("idx_chat_logs_space_date", table_name="chat_logs")
    op.drop_index("idx_chat_logs_outcome", table_name="chat_logs")
    op.drop_index("idx_chat_logs_message_id", table_name="chat_logs")
    op.drop_index("idx_chat_logs_domain", table_name="chat_logs")
    op.drop_index("idx_chat_logs_date_outcome", table_name="chat_logs")
    op.drop_index("idx_chat_logs_creation_date", table_name="chat_logs")
    op.drop_index("idx_chat_logs_config_hash", table_name="chat_logs")
    op.drop_table("chat_logs")
    op.drop_index("idx_messages_conv", table_name="messages")
    op.drop_table("messages")
    op.drop_table("document_nodes")
    op.drop_index("ix_chunks_visibility", table_name="chunks")
    op.drop_index("ix_chunks_space_id", table_name="chunks")
    op.drop_index("ix_chunks_owner_id", table_name="chunks")
    op.drop_index("ix_chunks_group_id", table_name="chunks")
    op.drop_index("ix_chunks_edited_by", table_name="chunks", postgresql_where="(edited_by IS NOT NULL)")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index(
        "ix_chunks_content_trgm",
        table_name="chunks",
        postgresql_using="gin",
        postgresql_ops={"content": "public.gin_trgm_ops"},
    )
    op.drop_index("idx_chunks_content_hash", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ux_act_versions_one_current", table_name="act_versions", postgresql_where="is_current")
    op.drop_index("idx_act_versions_act_id", table_name="act_versions")
    op.drop_table("act_versions")
    op.drop_table("user_groups")
    op.drop_index(
        "ux_documents_ingestion_slot",
        table_name="documents",
        postgresql_where="((ingestion_key IS NOT NULL) AND (status IN ('pending', 'processing', "
        "'indexing', 'done', 'failed')))",
    )
    op.drop_index(
        "ux_documents_active_slot",
        table_name="documents",
        postgresql_where="((ingestion_key IS NULL) AND (status IN ('pending', 'processing', 'indexing', "
        "'done', 'failed')))",
    )
    op.drop_index("idx_documents_visibility", table_name="documents")
    op.drop_index(
        "idx_documents_version_group",
        table_name="documents",
        postgresql_where="(version_group_id IS NOT NULL)",
    )
    op.drop_index("idx_documents_status", table_name="documents")
    op.drop_index("idx_documents_space", table_name="documents")
    op.drop_index("idx_documents_source_type", table_name="documents")
    op.drop_index("idx_documents_owner", table_name="documents")
    op.drop_index("idx_documents_group", table_name="documents")
    op.drop_index("idx_documents_doc_domain", table_name="documents")
    op.drop_table("documents")
    op.drop_index("idx_curator_user_assignments_target", table_name="curator_user_assignments")
    op.drop_table("curator_user_assignments")
    op.drop_table("curator_group_assignments")
    op.drop_index("idx_conversations_user", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index("idx_benchmark_runs_sweep_id", table_name="benchmark_runs")
    op.drop_index("idx_benchmark_runs_dataset", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
    op.drop_index("idx_benchmark_questions_is_active", table_name="benchmark_questions")
    op.drop_index("idx_benchmark_questions_dataset", table_name="benchmark_questions")
    op.drop_table("benchmark_questions")
    op.drop_table("benchmark_checkpoints")
    op.drop_index("idx_api_keys_user_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("knowledge_space_upload_groups")
    op.drop_index(
        "uq_benchmark_sweeps_one_active",
        table_name="benchmark_sweeps",
        postgresql_where="(status IN ('pending', 'running'))",
    )
    op.drop_index("idx_benchmark_sweeps_status", table_name="benchmark_sweeps")
    op.drop_table("benchmark_sweeps")
    op.drop_index(
        "idx_outbox_dispatch",
        table_name="vector_store_outbox",
        postgresql_where="(status IN ('pending', 'failed'))",
    )
    op.drop_index("idx_outbox_aggregate", table_name="vector_store_outbox")
    op.drop_table("vector_store_outbox")
    op.drop_index("ux_regulatory_acts_identity", table_name="regulatory_acts")
    op.drop_table("regulatory_acts")
    op.drop_table("rag_config_snapshots")
    op.drop_table("knowledge_spaces")
    op.drop_index("idx_ingestion_registry_key", table_name="ingestion_registry")
    op.drop_table("ingestion_registry")
    op.drop_table("groups")
    op.drop_table("corpus_processes")
    op.drop_table("corpus_control")
    op.drop_index(
        "idx_config_parameters_domain_key",
        table_name="config_parameters",
        postgresql_where="(domain_key IS NOT NULL)",
    )
    op.drop_index("idx_config_parameters_category", table_name="config_parameters")
    op.drop_table("config_parameters")
    op.drop_index(
        "ux_background_jobs_active_dedup",
        table_name="background_jobs",
        postgresql_where="((deduplication_key IS NOT NULL) AND (status IN ('pending', 'running')))",
    )
    op.drop_index("idx_background_jobs_status", table_name="background_jobs")
    op.drop_index(
        "idx_background_jobs_relay",
        table_name="background_jobs",
        postgresql_where="(((status)::text = 'pending'::text) AND (dispatch_payload IS NOT NULL))",
    )
    op.drop_index("idx_background_jobs_job_type", table_name="background_jobs")
    op.drop_index("idx_background_jobs_heartbeat", table_name="background_jobs")
    op.drop_index("idx_background_jobs_creation_date", table_name="background_jobs")
    op.drop_table("background_jobs")
