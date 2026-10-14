"""Adapt full and standalone RAG execution to the same benchmark answer contract."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
import logging
from typing import Protocol

from application.ports.chat_rag_port import ChatRAGPort
from config import _settings_overrides, get_setting, settings
from domain.value_objects.chat_context import ChatContext
from domain.services.access_control import get_visibility_conditions
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.repositories.vector.acl import build_qdrant_filter
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.benchmark import judge, metrics, retrieval
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.clients.managed_llm import close_llm_pools
from infrastructure.ml.config.client_switch import (
    prepare_clients,
    dispose_clients,
    LLM_CLIENT_KEYS,
    EMBEDDING_CLIENT_KEYS,
    RERANK_CLIENT_KEYS,
)
from infrastructure.ml.config.endpoint_policy import validate_endpoints, validate_pools
from domain.exceptions import ValidationError
from infrastructure.storage.file_storage import LazyStorage, finish_owned_task
from infrastructure.bm25.persistence import BM25LoadError, load_configured_bm25_snapshot

from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    snapshot_documents,
)

from infrastructure.benchmark.case_metrics import (
    compute_retriever_metrics_from_sources,
    extract_usage_from_response,
)
from infrastructure.ml.usage_capture import active_llm_usage


log = logging.getLogger("default")
ML_REFRESH_MAX_ATTEMPTS = 3


async def run_owned_sync(function, *args, **kwargs):
    """Wait for synchronous work before releasing the clients it still uses."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as exc:
        try:
            await finish_owned_task(task)
        finally:
            raise exc


@dataclass(frozen=True)
class BenchmarkAnswer:
    answer: str
    context: str
    retriever_metrics: dict
    input_tokens: int | None
    output_tokens: int | None
    ttft_sec: float | None = None
    breadth: str | None = None
    domain: str | None = None
    evidence: BenchmarkEvidence | None = None
    rag_latency_sec: float | None = None
    llm_usage: list[dict] | None = None


class BenchmarkAnswerGenerator(Protocol):
    async def generate(self, question: dict, ctx: ChatContext) -> BenchmarkAnswer: ...


class RagBenchmarkGenerator:
    def __init__(self, rag: ChatRAGPort, top_k: int, fetch_k: int | None) -> None:
        self._rag = rag
        self._top_k = top_k
        self._fetch_k = fetch_k

    async def generate(self, question: dict, ctx: ChatContext) -> BenchmarkAnswer:
        overrides = {
            **(_settings_overrides.get() or {}),
            "cache_enabled": False,
            "retriever_top_k": self._top_k,
        }
        if self._fetch_k is not None:
            overrides["retriever_fetch_k"] = self._fetch_k
        evidence = BenchmarkEvidence()
        evidence_token = active_evidence.set(evidence)
        token = _settings_overrides.set(overrides)
        records = active_llm_usage.get()
        if records is None:
            records = []
        usage_token = active_llm_usage.set(records)
        started = time.monotonic()
        try:
            result = await self._rag.invoke(question=question["question"], history=[], ctx=ctx)
            elapsed = time.monotonic() - started
        finally:
            active_llm_usage.reset(usage_token)
            _settings_overrides.reset(token)
            active_evidence.reset(evidence_token)
        context = "\n\n---\n\n".join(s.get("content", "") for s in result.sources if s.get("content"))
        if evidence.context is not None:
            context = evidence.context
        return BenchmarkAnswer(
            answer=result.answer,
            context=context,
            retriever_metrics=compute_retriever_metrics_from_sources(
                result.sources, question.get("source_hint")
            ),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            ttft_sec=result.ttft_sec,
            breadth=result.breadth,
            domain=result.domain,
            evidence=evidence,
            rag_latency_sec=elapsed,
            llm_usage=records,
        )


class StandaloneBenchmarkGenerator:
    def __init__(self, top_k: int, fetch_k: int | None, ctx: ChatContext) -> None:
        self._storage = LazyStorage()
        self._clients = MLClientRegistry(bm25_loader=partial(load_configured_bm25_snapshot, self._storage))
        self._ml_settings = settings.model_copy()
        self._ml_refresh_lock = asyncio.Lock()
        self._top_k = top_k
        self._fetch_k = fetch_k
        self._access_filter = build_qdrant_filter(
            user={"id": ctx.user_id, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}, group_ids=[]
        )
        self._visibility_conditions = get_visibility_conditions(
            user_kind=UserKind.INTERNAL,
            user_id=ctx.user_id,
            group_ids=[],
            for_list=False,
            user_role=UserRole.ADMIN,
        )

    async def refresh_ml_clients(self) -> None:
        client_keys = LLM_CLIENT_KEYS | EMBEDDING_CLIENT_KEYS | RERANK_CLIENT_KEYS
        async with self._ml_refresh_lock:
            for _ in range(ML_REFRESH_MAX_ATTEMPTS):
                candidate = settings.model_copy()
                keys = {
                    key for key in client_keys if getattr(candidate, key) != getattr(self._ml_settings, key)
                }
                if not keys:
                    return
                validate_endpoints(candidate)
                validate_pools(candidate)
                prepared = await prepare_clients(
                    self._clients, self._ml_settings, candidate, keys, applied=True
                )
                if any(getattr(settings, key) != getattr(candidate, key) for key in client_keys):
                    await dispose_clients(prepared)
                    continue
                self._clients.install_clients(prepared)
                self._ml_settings = candidate
                return
        raise ValidationError("ML configuration changed during benchmark preparation; retry the operation")

    async def generate(self, question: dict, ctx: ChatContext) -> BenchmarkAnswer:
        await self.refresh_ml_clients()
        llm = retrieval.build_llm(
            settings.llm_model, settings.ollama_base_url, provider=settings.llm_provider
        )
        try:
            try:
                snapshot = await self._clients.ensure_bm25_loaded()
            except BM25LoadError:
                log.warning("BM25 snapshot unavailable; using dense search", exc_info=True)
                snapshot = None
            docs = await run_owned_sync(
                retrieval.retrieve_with_scores_hybrid,
                question["question"],
                self._top_k,
                self._fetch_k or int(get_setting("rag.retriever_fetch_k")),
                access_filter=self._access_filter,
                visibility_conditions=self._visibility_conditions,
                user_id=ctx.user_id,
                user_group_ids=ctx.user_group_ids,
                ml_clients=self._clients,
                bm25_index=snapshot,
            )
            context, selected = format_docs_with_selection(docs, max_context_tokens=4000)
            answer, response = await run_owned_sync(
                judge.get_rag_answer_with_usage, llm, docs, question["question"]
            )
            input_tokens, output_tokens = extract_usage_from_response(response)
            return BenchmarkAnswer(
                answer=answer,
                context=context,
                evidence=BenchmarkEvidence(snapshot_documents(docs), snapshot_documents(selected), context),
                retriever_metrics=metrics.compute_retriever_metrics(docs, question.get("source_hint")),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        finally:
            await close_llm_pools(llm)

    async def close(self) -> None:
        try:
            await self._clients.close()
        finally:
            await self._storage.aclose()
