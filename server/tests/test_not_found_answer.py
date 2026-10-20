"""The fallback response is shared by prompts, guardrails and metrics."""

from domain.services.rag_policy.prompt_blocks import NOT_FOUND_ANSWER
from domain.services.rag_policy.prompts import build_system_prompt
from domain.value_objects.not_found_patterns import NOT_FOUND_PATTERNS
from infrastructure.ml.guardrails.output_scanner import BLOCKED_OUTPUT_MESSAGE


def test_not_found_answer_is_used_verbatim_in_prompt() -> None:
    assert NOT_FOUND_ANSWER == "Информация не найдена в документах."
    assert build_system_prompt().count(f'"{NOT_FOUND_ANSWER}"') == 3
    assert BLOCKED_OUTPUT_MESSAGE == NOT_FOUND_ANSWER


def test_not_found_answer_is_recognized_by_metrics() -> None:
    assert any(pattern in NOT_FOUND_ANSWER.lower() for pattern in NOT_FOUND_PATTERNS)
