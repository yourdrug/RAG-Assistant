"""Reserve the filename slot throughout document indexing.

Duplicates are reported without changing data; resolve them explicitly before retrying.
The table lock closes the gap between the preflight and index replacement.
"""

from alembic import op

revision = "n4e5f6g7h8i9"
down_revision = "m3d4e5f6g7h8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE")
    op.execute(
        """
        DO $$
        DECLARE conflicting_ids integer[];
        BEGIN
            SELECT array_agg(id ORDER BY id) INTO conflicting_ids
            FROM documents
            WHERE status IN ('pending', 'processing', 'indexing', 'done', 'failed')
            GROUP BY COALESCE(owner_id, 0), COALESCE(group_id, 0), filename
            HAVING count(*) > 1
            LIMIT 1;
            IF conflicting_ids IS NOT NULL THEN
                RAISE EXCEPTION 'Cannot reserve indexing filename slots: duplicate document IDs %',
                    conflicting_ids
                    USING HINT = 'Resolve duplicate filename slots explicitly, then retry migration.';
            END IF;
        END $$
        """
    )
    op.execute("DROP INDEX ux_documents_active_slot")
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_active_slot ON documents "
        "(COALESCE(owner_id, 0), COALESCE(group_id, 0), filename) "
        "WHERE status IN ('pending', 'processing', 'indexing', 'done', 'failed')"
    )


def downgrade() -> None:
    op.execute("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE")
    op.execute("DROP INDEX ux_documents_active_slot")
    op.execute(
        "CREATE UNIQUE INDEX ux_documents_active_slot ON documents "
        "(COALESCE(owner_id, 0), COALESCE(group_id, 0), filename) "
        "WHERE status IN ('pending', 'processing', 'done', 'failed')"
    )
