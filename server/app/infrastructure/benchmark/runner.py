"""Benchmark runner — main benchmark orchestration.

Runs the full RAG pipeline via RagService.invoke() with cache disabled,
then evaluates quality via LLM judge.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from config import _settings_overrides, get_setting, settings
from domain.exceptions import BenchmarkQuestionsNotFound

logger = logging.getLogger("default")

EXAMPLE_QUESTIONS = [
    {
        "id": "q1",
        "question": "Какие товары подлежат обязательной маркировке?",
        "expected_answer": None,
        "source_hint": None,
    },
    {
        "id": "q2",
        "question": "Каков порядок электронного документооборота?",
        "expected_answer": None,
        "source_hint": "электронном документе",
    },
]


def load_questions(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.warning("Файл %s не найден — создаю пример test_questions.json", path)
        example_path = Path(path)
        example_path.write_text(json.dumps(EXAMPLE_QUESTIONS, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Отредактируй %s и запусти снова.", path)
        raise BenchmarkQuestionsNotFound(path)

    data = json.loads(p.read_text(encoding="utf-8"))
    logger.info("Загружено вопросов: %d", len(data))
    return data


def _compute_retriever_metrics_from_sources(
    sources: list[dict],
    source_hint: str | None,
) -> dict:
    """Compute retriever metrics from RagResult.sources format.

    Sources from RagService are list[dict] with keys: source, max_score, pages, etc.
    """
    scores = [s.get("max_score", 0.0) for s in sources]
    avg_sim = sum(scores) / len(scores) if scores else 0.0

    if source_hint is None:
        return {
            "hit_rate": None,
            "mrr": None,
            "avg_similarity": round(avg_sim, 4),
            "retrieved_sources": [s.get("source", "?") for s in sources],
        }

    hit_rate = 0
    mrr = 0.0
    for rank, src in enumerate(sources, 1):
        filename = src.get("source", "")
        if source_hint.lower() in filename.lower():
            hit_rate = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "mrr": round(mrr, 4),
        "avg_similarity": round(avg_sim, 4),
        "retrieved_sources": [s.get("source", "?") for s in sources],
    }


async def run_benchmark_async(
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    max_concurrent: int | None = None,
    seed: int | None = None,
    n_runs: int = 1,
    rag_service=None,
    fetch_k: int | None = None,
):
    """Async benchmark using the full RAG pipeline via RagService.invoke().

    If rag_service is None, falls back to legacy standalone pipeline.
    Cache is disabled during benchmark runs.
    """
    if max_concurrent is None:
        max_concurrent = settings.benchmark_max_concurrent

    from domain.value_objects.chat_context import ChatContext
    from infrastructure.benchmark.judge import judge_answer_async
    from infrastructure.benchmark.metrics import (
        _estimate_cost_usd,
        compute_context_precision_recall,
    )
    from infrastructure.benchmark.persistence import (
        log_question_result,
        log_summary,
        save_results,
    )

    logger.info(
        "RAG Benchmark (async, max_concurrent=%d, pipeline=%s)",
        max_concurrent,
        "RagService" if rag_service else "standalone",
    )
    logger.info("  questions : %s", questions_path)
    logger.info("  top_k     : %d", top_k)
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)

    questions = load_questions(questions_path)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Benchmark context — admin user with full internal access.
    # ACL filter is built from admin visibility conditions to ensure
    # benchmark retrieval does not leak CLIENT_PRIVATE documents.
    bench_ctx = ChatContext(user_id=0, user_kind="internal", user_role="admin")

    from domain.services.access_control import get_visibility_conditions
    from domain.value_objects.roles import UserKind, UserRole
    from infrastructure.repositories.vector.acl import build_qdrant_filter

    bench_user_dict = {"id": 0, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}
    access_filter = build_qdrant_filter(
        user=bench_user_dict,
        group_ids=[],
    )
    visibility_conditions = get_visibility_conditions(
        user_kind=UserKind.INTERNAL,
        user_id=0,
        group_ids=[],
        for_list=False,
        user_role=UserRole.ADMIN,
    )

    all_results: list[dict] = []

    for run_idx in range(1, n_runs + 1):
        if n_runs > 1:
            logger.info("=== Run %d/%d ===", run_idx, n_runs)

        async def _process_question(idx: int, q: dict, _run_idx=run_idx) -> dict:
            async with semaphore:
                t_start = time.time()

                if rag_service is not None:
                    # ── Full pipeline path via RagService ──────────────
                    token = _settings_overrides.set({"cache_enabled": False})
                    try:
                        rag_result = await rag_service.invoke(
                            question=q["question"],
                            history=[],
                            ctx=bench_ctx,
                        )
                    finally:
                        _settings_overrides.reset(token)

                    answer = rag_result.answer
                    input_tokens = rag_result.input_tokens
                    output_tokens = rag_result.output_tokens
                    ttft_sec = rag_result.ttft_sec
                    breadth = rag_result.breadth
                    domain = rag_result.domain

                    retriever_metrics = _compute_retriever_metrics_from_sources(
                        rag_result.sources, q.get("source_hint")
                    )

                    # Build context_for_judge from sources
                    context_for_judge = "\n\n---\n\n".join(
                        s.get("content", "") for s in rag_result.sources if s.get("content")
                    )
                    if not context_for_judge:
                        context_for_judge = "\n\n---\n\n".join(
                            s.get("source", "") for s in rag_result.sources
                        )

                else:
                    # ── Legacy standalone path (fallback) ──────────────
                    from infrastructure.benchmark.judge import get_rag_answer_with_usage
                    from infrastructure.benchmark.metrics import compute_retriever_metrics
                    from infrastructure.benchmark.retrieval import build_llm, retrieve_with_scores_hybrid

                    rag_llm = build_llm(
                        settings.llm_model, settings.ollama_base_url, provider=settings.llm_provider
                    )
                    fetch_k_val = fetch_k or int(get_setting("rag.retriever_fetch_k"))
                    docs_with_scores = await asyncio.to_thread(
                        retrieve_with_scores_hybrid,
                        q["question"],
                        top_k,
                        fetch_k_val,
                        access_filter=access_filter,
                        visibility_conditions=visibility_conditions,
                        user_id=bench_ctx.user_id,
                        user_group_ids=bench_ctx.user_group_ids,
                    )
                    answer, rag_response = await asyncio.to_thread(
                        get_rag_answer_with_usage, rag_llm, docs_with_scores, q["question"]
                    )
                    input_tokens, output_tokens = _extract_usage_from_response(rag_response)
                    ttft_sec = None
                    breadth = None
                    domain = None

                    retriever_metrics = compute_retriever_metrics(docs_with_scores, q.get("source_hint"))
                    context_for_judge = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)

                # ── Judge scoring (same for both paths) ──────────────
                generator_metrics = await judge_answer_async(
                    question=q["question"],
                    answer=answer,
                    context=context_for_judge,
                    expected_answer=q.get("expected_answer"),
                )

                context_metrics = (
                    await asyncio.to_thread(
                        compute_context_precision_recall,
                        q["question"],
                        answer,
                        context_override=context_for_judge,
                    )
                    if context_for_judge
                    else {
                        "context_precision": None,
                        "context_recall": None,
                    }
                )

                latency = time.time() - t_start
                cost_usd = _estimate_cost_usd(settings.llm_model, input_tokens or 0, output_tokens or 0)

                result = {
                    "id": q.get("id", str(idx)),
                    "question": q["question"],
                    "answer": answer,
                    "expected_answer": q.get("expected_answer"),
                    "source_hint": q.get("source_hint"),
                    "retriever_metrics": retriever_metrics,
                    "generator_metrics": generator_metrics,
                    "context_metrics": context_metrics,
                    "latency_sec": round(latency, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost_usd, 6),
                    "ttft_sec": ttft_sec,
                    "breadth": breadth,
                    "domain": domain,
                    "run": _run_idx,
                }
                log_question_result(idx, len(questions), q, result)
                return result

        logger.info("Запускаю тесты...")
        total_start = time.time()

        tasks = [_process_question(idx, q) for idx, q in enumerate(questions, 1)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - total_start
        log_summary(list(results), total_time)
        all_results.extend(results)

    save_results(all_results, out_dir, model_name=settings.llm_model, run_id="all" if n_runs > 1 else "")


def _extract_usage_from_response(response) -> tuple[int | None, int | None]:
    """Extract input/output token counts from a LangChain LLM response."""
    try:
        metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = metadata.get("token_usage", {})
        if token_usage:
            return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")
        usage = metadata.get("usage", {})
        if usage:
            return usage.get("prompt_tokens"), usage.get("completion_tokens")
    except Exception:
        logger.debug("Failed to extract token usage from LLM response", exc_info=True)
    return None, None
