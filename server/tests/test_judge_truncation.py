"""Exercise actual Instructor parsing of provider length-limit responses."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import httpx
import instructor
from instructor.core.exceptions import IncompleteOutputException
from openai import OpenAI
import pytest

from config import settings
from infrastructure.benchmark.judging.judge import (
    JUDGE_EXPANDED_OUTPUT_TOKENS,
    JUDGE_OUTPUT_TOKENS,
    judge_with_structured_output,
)


if TYPE_CHECKING:
    from infrastructure.ml.clients.managed_sync_instructor import ManagedSyncInstructor


@pytest.mark.parametrize("still_truncated", [False, True])
@pytest.mark.parametrize(
    "budgets_pair", [(JUDGE_OUTPUT_TOKENS, JUDGE_EXPANDED_OUTPUT_TOKENS), (100, 200), (100, 100)]
)
def test_truncated_judge_retries_once_with_larger_budget(
    still_truncated: bool, budgets_pair: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(settings, "benchmark_judge_initial_tokens", budgets_pair[0])
    monkeypatch.setattr(settings, "benchmark_judge_retry_tokens", budgets_pair[1])
    budgets = []

    def respond(request: httpx.Request) -> httpx.Response:
        budgets.append(json.loads(request.content)["max_tokens"])
        truncated = len(budgets) == 1 or still_truncated
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"score":8,"reason":"ok"}'},
                        "finish_reason": "length" if truncated else "stop",
                    }
                ],
            },
        )

    with OpenAI(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond))
    ) as raw:
        client = instructor.from_openai(raw, mode=instructor.Mode.MD_JSON)

        if still_truncated:
            with pytest.raises(IncompleteOutputException):
                judge_with_structured_output(cast("ManagedSyncInstructor", client), "score", "test")
        else:
            assert (
                judge_with_structured_output(cast("ManagedSyncInstructor", client), "score", "test").score
                == 8
            )
    assert budgets == list(budgets_pair)
