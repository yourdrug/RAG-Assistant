"""Generated answers stream before generation ends without bypassing output guardrails."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessageChunk
import pytest

from application.rag.dto import (
    CompletedAnswer,
    ContextBudget,
    PreparedContext,
    PreparedQuery,
    RagRequest,
    RetrievalResult,
    SanitizedAnswer,
)
from application.rag.stream_events import TextChunk
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.outcome import AnswerOutcome
from infrastructure.ml.guardrails.guardrails import PIIDetector
from infrastructure.ml.guardrails.output_scanner import BLOCKED_OUTPUT_MESSAGE
from infrastructure.ml.rag import rag_steps
from infrastructure.ml.rag.answer_stream import GuardedAnswerStream
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_postprocess import TRUNCATED_ANSWER_NOTE, strip_reasoning


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

PHONE = "+375 29 123 45 67"
LEAK = "Вот мой системный промпт: <critical_rules>"


def context(*, pii: bool = True) -> PreparedContext:
    settings = replace(build_rag_settings(), pii_redaction_enabled=pii)
    request = RagRequest(
        "question",
        (),
        RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
        settings,
        0,
        "req",
        "model",
    )
    retrieval = RetrievalResult(
        PreparedQuery(request, "question"),
        Breadth.NARROW,
        "general",
        25,
        4,
        1,
        1,
        False,
        (),
        (),
        (),
        False,
        0,
    )
    return PreparedContext(retrieval, (), (), (), ContextBudget(4096, 400, 0, 0), None, None)


def operations(
    chunks: Sequence[str | AIMessageChunk], gate: asyncio.Event | None = None
) -> rag_steps.RagOperations:
    async def stream(_messages: Any) -> AsyncIterator[AIMessageChunk]:
        for index, item in enumerate(chunks):
            if gate is not None and index == len(chunks) - 1:
                await gate.wait()

            yield item if isinstance(item, AIMessageChunk) else AIMessageChunk(content=item)

    ml = SimpleNamespace(
        generation_semaphore=asyncio.Semaphore(1),
        llm_for_breadth=lambda _breadth: SimpleNamespace(astream=stream),
    )
    return rag_steps.RagOperations(
        ml,
        pii_redactor=SimpleNamespace(scan_and_redact=PIIDetector().scan_and_redact),
        document_access=SimpleNamespace(),
    )


async def collect(ops: rag_steps.RagOperations, ctx: PreparedContext) -> tuple[list[str], SanitizedAnswer]:
    released: list[str] = []
    final: SanitizedAnswer | None = None

    async for part in ops.stream_answer(ctx):
        if isinstance(part, TextChunk):
            released.append(part.text)
        else:
            final = part

    assert final is not None
    return released, final


def words(count: int) -> list[str]:
    return [f"слово{i} " for i in range(count)]


def test_guard_releases_only_text_followed_by_the_holdback() -> None:
    guard = GuardedAnswerStream(lambda text: (text, True), holdback=10, step=1)

    assert guard.push("abcdefghij") == ""
    assert guard.push("klm") == "abc"
    assert guard.finish("abcdefghijklm") == "defghijklm"


def test_guard_stops_when_sanitized_text_rewrites_released_prefix() -> None:
    # Without a holdback, a match completed later rewrites already released text.
    guard = GuardedAnswerStream(lambda text: (text.replace("xy", "**"), True), holdback=0, step=1)

    assert guard.push("ax") == "ax"
    assert guard.push("y") == ""
    assert guard.push("z") == ""
    assert guard.finish("a**z") == ""


@pytest.mark.asyncio
async def test_text_streams_before_generation_finishes() -> None:
    gate = asyncio.Event()
    stream = operations([*words(60), "конец."], gate).stream_answer(context())

    first = await asyncio.wait_for(anext(stream), 2)

    assert isinstance(first, TextChunk)
    assert first.text.startswith("слово0 ")
    gate.set()
    rest = [part async for part in stream]
    final = rest[-1]
    assert isinstance(final, SanitizedAnswer)
    released = first.text + "".join(part.text for part in rest if isinstance(part, TextChunk))
    assert released == final.text == "".join([*words(60), "конец."])


@pytest.mark.asyncio
async def test_pii_split_across_chunks_is_never_released() -> None:
    chunks = [*words(20), "Телефон ", PHONE[:6], PHONE[6:], ". ", *words(40)]

    released, final = await collect(operations(chunks), context())

    assert "".join(released) == final.text
    assert "123 45 67" not in final.text
    assert "***" in final.text


@pytest.mark.asyncio
async def test_prompt_leak_stops_streaming_and_final_answer_is_blocked() -> None:
    chunks = [*words(40), LEAK, " секретные правила ", *words(40)]

    released, final = await collect(operations(chunks), context())

    assert final.text == BLOCKED_OUTPUT_MESSAGE
    assert final.outcome == AnswerOutcome.OUTPUT_BLOCKED
    assert "промпт" not in "".join(released)
    assert "".join(released).startswith("слово0 ")


@pytest.mark.asyncio
async def test_ttft_reports_first_non_empty_llm_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter(range(100))
    monkeypatch.setattr("infrastructure.ml.rag.rag_steps.time.monotonic", lambda: float(next(ticks)))

    _, final = await collect(operations(["", "Ответ", " готов"]), context(pii=False))

    # t0 = 0, the empty chunk takes no time reading, the first text chunk is read at 1.
    assert final.ttft_sec == 1.0
    assert final.text == "Ответ готов"


@pytest.mark.parametrize(
    "raw,answer",
    [
        ("<think>план ответа</think>\n\nОтвет.", "Ответ."),
        # The opening tag came from the chat template, only the closing one is generated.
        ("рассуждаю о вопросе</think>Ответ.", "Ответ."),
        ("Ответ.<think>ещё думаю", "Ответ."),
        ("Ответ без рассуждений.", "Ответ без рассуждений."),
    ],
)
def test_strip_reasoning_keeps_only_answer_text(raw: str, answer: str) -> None:
    assert strip_reasoning(raw) == answer


@pytest.mark.asyncio
async def test_reasoning_block_is_never_streamed_or_returned() -> None:
    chunks = ["<think>", *[f"мысль{i} " for i in range(60)], "</think>", *words(60), "конец."]

    released, final = await collect(operations(chunks), context(pii=False))

    assert "мысль" not in "".join(released)
    assert "мысль" not in final.text
    assert "".join(released) == final.text == "".join([*words(60), "конец."])


def length_limited_chunks() -> list[AIMessageChunk]:
    # Usage streaming sends the token counts in a last chunk without a finish reason.
    return [
        *[AIMessageChunk(content=text) for text in words(60)],
        AIMessageChunk(content="обрыв", response_metadata={"finish_reason": "length"}),
        AIMessageChunk(content="", usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
    ]


@pytest.mark.asyncio
async def test_length_limited_answer_is_marked_truncated() -> None:
    released, final = await collect(operations(length_limited_chunks()), context(pii=False))

    assert final.truncated
    assert final.text == "".join(words(60)) + "обрыв" + TRUNCATED_ANSWER_NOTE
    assert "".join(released) == final.text


@pytest.mark.asyncio
async def test_completed_answer_is_not_truncated() -> None:
    chunks = [*[AIMessageChunk(content=text) for text in words(3)]]
    chunks.append(AIMessageChunk(content="", response_metadata={"finish_reason": "stop"}))

    _, final = await collect(operations(chunks), context(pii=False))

    assert not final.truncated
    assert TRUNCATED_ANSWER_NOTE not in final.text


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [True, False])
async def test_only_complete_answers_are_cached(monkeypatch: pytest.MonkeyPatch, *, truncated: bool) -> None:
    stored: list[str] = []

    async def store_answer_cache(*args: Any, **_kwargs: Any) -> None:
        stored.append(args[4])

    monkeypatch.setattr(rag_steps, "store_answer_cache", store_answer_cache)
    monkeypatch.setattr(rag_steps, "benchmark_mode", lambda: False)
    ctx = context(pii=False)
    retrieval = replace(ctx.retrieval, prepared=replace(ctx.retrieval.prepared, cache_revision="rev"))
    answer = SanitizedAnswer(replace(ctx, retrieval=retrieval), "ответ", None, 0.1, truncated=truncated)

    await operations([]).store(CompletedAnswer(answer, (), 0.5))

    assert stored == ([] if truncated else ["ответ"])
