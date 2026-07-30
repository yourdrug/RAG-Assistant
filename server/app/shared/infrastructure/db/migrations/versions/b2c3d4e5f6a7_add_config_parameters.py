"""add config_parameters table for dynamic config without restart

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-30 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop table if it exists from a failed previous run
    op.execute("DROP TABLE IF EXISTS config_parameters CASCADE")

    op.create_table(
        "config_parameters",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "creation_date", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(length=20), server_default="str", nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("min_value", sa.Float(), nullable=True),
        sa.Column("max_value", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )

    op.create_index("idx_config_parameters_category", "config_parameters", ["category"])

    # Seed default values for dynamic parameters
    op.execute("""
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value) VALUES
        ('retriever_fetch_k',    '25',  'int',   'rag',       'Candidates from Qdrant before rerank (narrow)', 1, 100),
        ('retriever_top_k',      '6',   'int',   'rag',       'Chunks after rerank (narrow)',                  1, 50),
        ('retriever_fetch_k_broad','40','int',   'rag',       'Candidates from Qdrant before rerank (broad)',  1, 200),
        ('retriever_top_k_broad','10',  'int',   'rag',       'Chunks after rerank (broad)',                   1, 50),
        ('history_window',       '8',   'int',   'rag',       'History messages sent to LLM',                  0, 50),
        ('chunk_size',           '900', 'int',   'rag',       'Chunk size for document splitting',            100, 5000),
        ('chunk_overlap',        '150', 'int',   'rag',       'Overlap between chunks',                         0, 1000),
        ('hybrid_enabled',       'true','bool',  'hybrid',    'Enable BM25 + dense hybrid search',           NULL, NULL),
        ('bm25_fetch_k',         '25',  'int',   'hybrid',    'BM25 candidates before RRF',                    1, 100),
        ('rrf_k',                '60',  'int',   'hybrid',    'RRF constant',                                  1, 200),
        ('dense_weight',         '1.0', 'float', 'hybrid',    'Dense weight in RRF',                          0.0, 10.0),
        ('sparse_weight',        '1.0', 'float', 'hybrid',    'Sparse weight in RRF',                         0.0, 10.0),
        ('embed_batch_size',     '32',  'int',   'ingestion', 'Embedding batch size',                           1, 256)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS config_parameters")
