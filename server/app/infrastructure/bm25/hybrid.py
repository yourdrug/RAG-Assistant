"""BM25 sparse retrieval and Reciprocal Rank Fusion (RRF) merge.

.. deprecated::
    This module is kept for backward compatibility. Import from the specific
    submodules instead:

    - ``infrastructure.bm25.bm25_index.BM25Index``
    - ``infrastructure.bm25.persistence.save_bm25_index_to_s3``
    - ``infrastructure.bm25.rrf.rrf_merge``
    - ``domain.utils.content_hash``
"""

from infrastructure.bm25._stemmer import stem_token  # noqa: F401
from infrastructure.bm25._tokenizer import tokenize, tokenize_raw  # noqa: F401
from infrastructure.bm25.bm25_index import BM25Index  # noqa: F401
from infrastructure.bm25.persistence import (  # noqa: F401
    BM25_S3_KEY,
    load_bm25_index,
    load_bm25_index_from_s3,
    load_bm25_index_from_s3_sync,
    save_bm25_index,
    save_bm25_index_to_s3,
)
from domain.utils import rrf_merge  # noqa: F401

# Re-export from domain for backward compatibility
from domain.utils import content_hash  # noqa: F401
