"""add LLM, OCR, Storage dynamic config parameters

Revision ID: d4e5f6a7b8c9
Revises: 8e19a1cfb758
Create Date: 2026-08-06 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "8e19a1cfb758"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO config_parameters (key, value, value_type, category, description, min_value, max_value) VALUES
        -- LLM
        ('llm_model',              'qwen2.5:14b', 'str',   'llm',    'LLM model name (Ollama)',                  NULL,  NULL),
        ('llm_temperature',        '0.1',          'float', 'llm',    'LLM temperature',                         0.0,   2.0),
        ('llm_top_p',              '0.9',          'float', 'llm',    'LLM top_p (nucleus sampling)',             0.0,   1.0),
        ('llm_num_ctx_narrow',     '8192',         'int',   'llm',    'Context window size (narrow questions)',   1024,  131072),
        ('llm_num_ctx_broad',      '16384',        'int',   'llm',    'Context window size (broad questions)',    1024,  131072),
        ('llm_num_predict_narrow', '400',          'int',   'llm',    'Max tokens to generate (narrow)',          64,    8192),
        ('llm_num_predict_broad',  '2048',         'int',   'llm',    'Max tokens to generate (broad)',           64,    32768),
        -- OCR
        ('ocr_enabled',            'true',         'bool',  'ocr',    'Enable OCR for scanned PDF pages',        NULL,  NULL),
        ('ocr_engine',             'paddleocr',    'str',   'ocr',    'OCR engine: paddleocr | surya | auto',     NULL,  NULL),
        ('ocr_dpi',                '300',          'int',   'ocr',    'DPI for PDF page rendering',              72,    600),
        ('ocr_lang_surya',         '["ru","en"]',  'str',   'ocr',    'Surya OCR languages (JSON array)',        NULL,  NULL),
        ('ocr_lang_paddle',        'ru',           'str',   'ocr',    'PaddleOCR language (requires restart of OCR engine)', NULL, NULL),
        -- Storage
        ('file_backend',           'local',        'str',   'storage','File backend: local | s3',                 NULL,  NULL),
        ('s3_endpoint',            'http://minio:9000', 'str', 'storage','S3/MinIO endpoint URL',                  NULL,  NULL),
        ('s3_bucket',              'rag-documents','str',   'storage','S3 bucket name',                           NULL,  NULL),
        ('s3_access_key',          'minioadmin',   'str',   'storage','S3 access key',                            NULL,  NULL),
        ('s3_secret_key',          'minioadmin',   'str',   'storage','S3 secret key',                            NULL,  NULL),
        ('s3_region',              'us-east-1',    'str',   'storage','S3 region',                                NULL,  NULL),
        ('data_dir',               '/code/project/data', 'str', 'storage','Root data directory (local mode)',     NULL,  NULL)
    ON CONFLICT (key) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM config_parameters WHERE key IN (
            'llm_model', 'llm_temperature', 'llm_top_p',
            'llm_num_ctx_narrow', 'llm_num_ctx_broad',
            'llm_num_predict_narrow', 'llm_num_predict_broad',
            'ocr_enabled', 'ocr_engine', 'ocr_dpi', 'ocr_lang_surya', 'ocr_lang_paddle',
            'file_backend', 's3_endpoint', 's3_bucket', 's3_access_key', 's3_secret_key', 's3_region',
            'data_dir'
        )
    """)
