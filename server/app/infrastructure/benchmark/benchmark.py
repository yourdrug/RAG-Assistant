"""Backward-compatible re-exports from focused sub-modules.

All ``from infrastructure.benchmark.benchmark import X`` continue to work.
New code should import from the specific sub-modules directly.
"""

from infrastructure.benchmark.judge import (  # noqa: F401
    get_rag_answer,
    get_rag_answer_with_usage,
    judge_answer,
    judge_answer_async,
)
from infrastructure.benchmark.metrics import (  # noqa: F401
    _extract_source_name,
    _safe_avg,
    compute_context_precision_recall,
    compute_retriever_metrics,
    compute_summary_metrics,
)
from infrastructure.benchmark.persistence import (  # noqa: F401
    _sanitize_model_name,
    log_question_result,
    log_summary,
    save_results,
)
from infrastructure.benchmark.retrieval import (  # noqa: F401
    _apply_rerank_filters,
    build_llm,
    retrieve_with_scores_hybrid,
)
from infrastructure.benchmark.runner import (  # noqa: F401
    EXAMPLE_QUESTIONS,
    load_questions,
    run_benchmark_async,
)
