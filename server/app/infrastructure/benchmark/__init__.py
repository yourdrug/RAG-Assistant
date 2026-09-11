"""Benchmarking — RAG quality benchmark, history tracking, and parameter sweep.

Re-exports from focused sub-modules for backward compatibility.
All ``from infrastructure.benchmark.benchmark import X`` continue to work.
"""

# History & sweep (unchanged)
from infrastructure.benchmark.benchmark_history import (  # noqa: F401
    compare_runs,
    get_last_baseline,
    load_history,
    save_summary_to_history,
)
from infrastructure.benchmark.benchmark_history_adapter import BenchmarkHistoryAdapter  # noqa: F401
from infrastructure.benchmark.sweep_engine import SweepEngine  # noqa: F401

# Retrieval
from infrastructure.benchmark.retrieval import (  # noqa: F401
    build_llm,
    retrieve_with_scores_hybrid,
)

# Judge
from infrastructure.benchmark.judge import (  # noqa: F401
    get_rag_answer,
    judge_answer,
    judge_answer_async,
)

# Metrics
from infrastructure.benchmark.metrics import (  # noqa: F401
    compute_context_precision_recall,
    compute_retriever_metrics,
    compute_summary_metrics,
    _safe_avg,
)

# Persistence
from infrastructure.benchmark.persistence import (  # noqa: F401
    log_question_result,
    log_summary,
    save_results,
    _sanitize_model_name,
)

# Runner
from infrastructure.benchmark.runner import (  # noqa: F401
    load_questions,
    run_benchmark,
    run_benchmark_async,
)

__all__ = [
    "BenchmarkHistoryAdapter",
    "SweepEngine",
    "compare_runs",
    "get_last_baseline",
    "load_history",
    "save_summary_to_history",
    "build_llm",
    "retrieve_with_scores_hybrid",
    "get_rag_answer",
    "judge_answer",
    "judge_answer_async",
    "compute_context_precision_recall",
    "compute_retriever_metrics",
    "compute_summary_metrics",
    "_safe_avg",
    "log_question_result",
    "log_summary",
    "save_results",
    "_sanitize_model_name",
    "load_questions",
    "run_benchmark",
    "run_benchmark_async",
]
