"""Framework-agnostic RAG policy with a stable public import path.

Responsibilities:
- classification: question breadth/domain, references and conditional cases;
- sanitization: escaping untrusted prompt content;
- prompt_blocks: canonical instruction wording;
- prompts: system instructions and user-role context assembly;
- retrieval: candidate limits, hybrid weights and context budgets.

Import public functions from this package; implementations depend only on the
standard library and other domain modules.
"""

from .classification import (
    COMPOUND_PATTERNS,
    classify_query_domain,
    classify_question_breadth,
    has_exact_reference,
    is_out_of_domain,
    needs_decomposition,
    should_enumerate_cases,
)
from .prompts import (
    DECOMPOSITION_ASSESSMENT_SYSTEM,
    SUFFICIENCY_ASSESSMENT_SYSTEM,
    build_context_message,
    build_system_prompt,
)
from .retrieval import (
    RetrievalParams,
    compute_context_budget,
    compute_retrieval_params,
    select_final_top_k,
)
from .sanitization import sanitize_for_prompt

# Retained for existing consumers that inspect the canonical refusal rule.
from .prompt_blocks import _CRITICAL_RULES_BLOCK as _CRITICAL_RULES_BLOCK

__all__ = [
    "COMPOUND_PATTERNS",
    "classify_query_domain",
    "classify_question_breadth",
    "has_exact_reference",
    "is_out_of_domain",
    "needs_decomposition",
    "should_enumerate_cases",
    "DECOMPOSITION_ASSESSMENT_SYSTEM",
    "SUFFICIENCY_ASSESSMENT_SYSTEM",
    "build_context_message",
    "build_system_prompt",
    "RetrievalParams",
    "compute_context_budget",
    "compute_retrieval_params",
    "select_final_top_k",
    "sanitize_for_prompt",
]
