"""Identify acts by signing date and visibility scope.

Revision ID: e8f9a0b1c2d3
Revises: a1b2c3d4e5f8
"""

import sqlalchemy as sa
from alembic import op

revision = "e8f9a0b1c2d3"
down_revision = "a1b2c3d4e5f8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("regulatory_acts", sa.Column("act_date", sa.Date(), nullable=True))
    op.add_column(
        "regulatory_acts",
        sa.Column("visibility_scope", sa.String(100), nullable=False, server_default="internal_public"),
    )
    # Preserve homogeneous chains. Mixed legacy chains cannot be assigned to
    # a single scope safely, so new uploads never automatically join them.
    op.execute("""
        UPDATE regulatory_acts a SET visibility_scope = scopes.scope
        FROM (
            SELECT av.act_id,
                CASE WHEN COUNT(DISTINCT
                    CASE WHEN d.visibility = 'internal_group'
                        THEN d.visibility || ':' || COALESCE(d.group_id::text, 'None')
                    WHEN d.visibility IN ('internal_private', 'client_private')
                        THEN d.visibility || ':' || COALESCE(d.owner_id::text, 'None')
                    ELSE d.visibility END) = 1
                THEN MIN(CASE WHEN d.visibility = 'internal_group'
                    THEN d.visibility || ':' || COALESCE(d.group_id::text, 'None')
                    WHEN d.visibility IN ('internal_private', 'client_private')
                    THEN d.visibility || ':' || COALESCE(d.owner_id::text, 'None')
                    ELSE d.visibility END)
                ELSE 'legacy' END AS scope
            FROM act_versions av JOIN documents d ON d.id = av.document_id
            WHERE av.act_id IS NOT NULL GROUP BY av.act_id
        ) scopes WHERE a.id = scopes.act_id
    """)
    op.execute("ALTER TABLE regulatory_acts DROP CONSTRAINT IF EXISTS ux_regulatory_acts_type_number")
    op.execute("DROP INDEX IF EXISTS ux_regulatory_acts_type_number")
    op.create_index(
        "ux_regulatory_acts_identity",
        "regulatory_acts",
        ["act_type", "act_number", "visibility_scope", sa.text("COALESCE(act_date, DATE '0001-01-01')")],
        unique=True,
    )
    _repair_version_metadata()


def _repair_version_metadata() -> None:
    # Split mixed legacy chains using the documents' authoritative ACL. Never
    # let a private upload keep a public edition noncurrent after deployment.
    op.execute("""
        CREATE TEMP TABLE rag_act_scope_repair ON COMMIT DROP AS
        SELECT av.id AS version_id, av.act_id,
            CASE WHEN d.visibility = 'internal_group'
                THEN d.visibility || ':' || COALESCE(d.group_id::text, 'None')
            WHEN d.visibility IN ('internal_private', 'client_private')
                THEN d.visibility || ':' || COALESCE(d.owner_id::text, 'None')
            ELSE d.visibility END AS scope
        FROM act_versions av JOIN documents d ON d.id = av.document_id
    """)
    op.execute("""
        INSERT INTO regulatory_acts (act_type, act_number, title, issuing_authority, visibility_scope)
        SELECT DISTINCT a.act_type, a.act_number, a.title, a.issuing_authority, s.scope
        FROM regulatory_acts a JOIN rag_act_scope_repair s ON s.act_id = a.id
        WHERE a.visibility_scope = 'legacy' AND a.act_number IS NOT NULL
        ON CONFLICT DO NOTHING
    """)
    op.execute("""
        UPDATE act_versions av SET act_id = target.id
        FROM rag_act_scope_repair s, regulatory_acts original, regulatory_acts target
        WHERE av.id = s.version_id AND av.act_id = original.id
          AND original.visibility_scope = 'legacy'
          AND original.act_number IS NOT NULL
          AND target.act_type = original.act_type
          AND target.act_number IS NOT DISTINCT FROM original.act_number
          AND target.visibility_scope = s.scope
          AND target.act_date IS NULL
    """)
    op.execute("""
        UPDATE act_versions av SET act_id = NULL, is_current = TRUE
        FROM regulatory_acts a
        WHERE av.act_id = a.id AND a.visibility_scope = 'legacy' AND a.act_number IS NULL
    """)
    # Untrusted candidates were previously persisted as filterable dates. They
    # belong in manual review, never in an inferred historical interval.
    op.execute("""
        UPDATE act_versions SET effective_from = NULL, effective_to = NULL
        WHERE date_source = 'extracted'
    """)
    # Close older intervals by the next known effective date, preserving any
    # manually supplied earlier end date (including an intentional gap).
    op.execute("""
        WITH bounds AS (
            SELECT v.id, MIN(later.effective_from) AS next_date
            FROM act_versions v JOIN act_versions later ON later.act_id = v.act_id
              AND (later.effective_from > v.effective_from
                   OR (later.effective_from = v.effective_from AND later.id > v.id))
            GROUP BY v.id
        )
        UPDATE act_versions v SET effective_to = bounds.next_date FROM bounds
        WHERE v.id = bounds.id AND (v.effective_to IS NULL OR v.effective_to > bounds.next_date)
    """)
    # Unset first to respect the one-current-version constraint during repair.
    op.execute("UPDATE act_versions SET is_current = FALSE WHERE act_id IS NOT NULL")
    op.execute("""
        WITH ranked AS (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY act_id ORDER BY effective_from DESC NULLS LAST, id DESC
            ) AS position
            FROM act_versions WHERE act_id IS NOT NULL
              AND (effective_from IS NULL OR effective_from <= CURRENT_DATE)
              AND (effective_to IS NULL OR effective_to > CURRENT_DATE)
        )
        UPDATE act_versions SET is_current = TRUE FROM ranked
        WHERE act_versions.id = ranked.id AND ranked.position = 1
    """)
    op.execute("""
        UPDATE chunks c SET effective_from = v.effective_from,
            effective_to = v.effective_to, is_current = v.is_current
        FROM act_versions v WHERE c.act_version_id = v.id
    """)
    # Existing vectors and answer caches must reflect the repaired SQL metadata.
    # The outbox worker updates payloads and invalidates affected answers without
    # recomputing embeddings.
    op.execute("""
        INSERT INTO vector_store_outbox
            (operation, aggregate_type, aggregate_id, payload, status, attempts,
             max_attempts, next_attempt_at)
        SELECT 'update_metadata', 'act_version', id,
            json_build_object('act_version_id', id, 'act_id', act_id, 'is_current', is_current,
                'effective_from', effective_from, 'effective_to', effective_to),
            'pending', 0, 8, NOW()
        FROM act_versions
    """)


def downgrade() -> None:
    # The old uniqueness constraint intentionally fails if scoped/date-specific
    # identities now share a number: downgrade must not silently destroy data.
    op.create_unique_constraint(
        "ux_regulatory_acts_type_number", "regulatory_acts", ["act_type", "act_number"]
    )
    op.drop_index("ux_regulatory_acts_identity", table_name="regulatory_acts")
    op.drop_column("regulatory_acts", "visibility_scope")
    op.drop_column("regulatory_acts", "act_date")
