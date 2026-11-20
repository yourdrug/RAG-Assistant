"""Persist a compact projection for review, leaving full results in their existing column."""

from alembic import op
import sqlalchemy as sa


revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_runs", sa.Column("review_rows", sa.JSON(), nullable=True))
    op.execute("""
        CREATE FUNCTION benchmark_review_rows(results json) RETURNS json
        LANGUAGE sql IMMUTABLE SET search_path FROM CURRENT AS $$
            SELECT CASE WHEN json_typeof(results) = 'array' THEN COALESCE((
                SELECT json_agg(json_build_object(
                    'id', row.id, 'case_id', row.case_id, 'question', row.question,
                    'run', COALESCE(item.value->'run', '1'::json),
                    'outcome', row.outcome, 'answer_type', row.answer_type,
                    'correctness', row.correctness, 'faithfulness', row.faithfulness,
                    'generator_metrics', row.generator_metrics,
                    'judge_rubric_version', row.judge_rubric_version,
                    'hit_rate', COALESCE(row.hit_rate, row.retriever_metrics->'hit_rate'),
                    'evidence_metrics', COALESCE(row.evidence_metrics, '{}'::json),
                    'evidence', json_build_object('completion', row.evidence->'completion'),
                    'annotation_validation', json_build_object('status', row.annotation_validation->'status'),
                    'annotations', json_build_object(
                        'answer_type', row.annotations->'answer_type',
                        'expected_refusal', row.annotations->'expected_refusal',
                        'expected_fragments',
                        NULLIF((row.annotations->'expected_fragments')::jsonb, 'null'::jsonb) <> '[]'::jsonb,
                        'required_facts',
                        NULLIF((row.annotations->'required_facts')::jsonb, 'null'::jsonb) <> '[]'::jsonb
                    )
                ) ORDER BY item.ordinality)
                FROM json_array_elements(
                    CASE WHEN json_typeof(results) = 'array' THEN results ELSE '[]'::json END
                ) WITH ORDINALITY AS item(value, ordinality)
                CROSS JOIN LATERAL json_to_record(
                    CASE WHEN json_typeof(item.value) = 'object' THEN item.value ELSE '{}'::json END
                ) AS row(
                    id json, case_id json, question text, outcome text, answer_type text,
                    correctness json, faithfulness json, generator_metrics json, judge_rubric_version text,
                    hit_rate json, retriever_metrics json, evidence_metrics json, evidence json,
                    annotation_validation json, annotations json
                )
            ), '[]'::json) END
        $$
    """)
    op.execute("""
        UPDATE benchmark_runs SET review_rows = benchmark_review_rows(per_question_results)
        WHERE json_typeof(per_question_results) = 'array'
    """)
    op.execute("""
        CREATE FUNCTION sync_benchmark_review_rows() RETURNS trigger
        LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
        BEGIN
            NEW.review_rows := benchmark_review_rows(NEW.per_question_results);
            RETURN NEW;
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER benchmark_review_rows_sync
        BEFORE INSERT OR UPDATE OF per_question_results ON benchmark_runs
        FOR EACH ROW EXECUTE FUNCTION sync_benchmark_review_rows()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER benchmark_review_rows_sync ON benchmark_runs")
    op.execute("DROP FUNCTION sync_benchmark_review_rows()")
    op.execute("DROP FUNCTION benchmark_review_rows(json)")
    op.drop_column("benchmark_runs", "review_rows")
