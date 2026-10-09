"""Compare original and manually decomposed retrieval without changing runtime flags.

Queries are frozen before retrieval; no gold answers enter search. Both arms use
identical ACL/temporal filters, fetch_k, reranker and final context budget. A fair
comparison of automatic decomposition must be a separate experiment.
"""

import argparse
import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path

from langchain_core.documents import Document

from domain.services import get_visibility_conditions
from domain.utils import content_hash
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_retrieval import run_hybrid_search
from infrastructure.ml.rag.rag_reranking import rerank_documents
from infrastructure.repositories.vector.acl import build_qdrant_filter, with_temporal_filter

logger = logging.getLogger("default")
QUERIES = {
    149: [
        "Структура сведений EDI-провайдеров ссылка на классификатор дополнительной таможенной информации",
        "Структура ЭТТН ссылка на классификатор дополнительной таможенной информации",
        "Какой орган утверждает классификатор дополнительной таможенной информации?",
    ],
    159: [
        "Кто такой EDI-провайдер и какую роль играет в передаче сведений в ПК СПТ постановление МНС N 11?",
        "Кто такой EDI-провайдер и какую роль играет в передаче сведений в ПК СПТ постановление МНС N 13?",
    ],
    160: [
        "Требования к электронной цифровой подписи Закон N 113-З",
        "Требования к ЭЦП постановление МНС N 11 представление сведений EDI-провайдерами",
    ],
}


def record(doc, score=None):
    return {"content": doc.page_content, "metadata": doc.metadata, "score": score}


async def compare(cases, user_id):
    registry = MLClientRegistry()
    settings = build_rag_settings()
    settings = replace(settings, retriever=replace(settings.retriever, fetch_k=30, top_k=8))
    user = ChatContext(user_id, UserKind.INTERNAL, UserRole.ADMIN).to_user_context()
    conditions = get_visibility_conditions(
        UserKind.INTERNAL, user_id, [], for_list=False, user_role=UserRole.ADMIN
    )
    access_filter = with_temporal_filter(build_qdrant_filter(user, []), None)
    report = {
        "experiment": "manual_act_queries",
        "acl_principal": {"user_id": user_id, "role": UserRole.ADMIN.value},
        "fetch_k_per_query": 30,
        "top_k": 8,
        "max_context_tokens": 6000,
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "automatic_decomposition": False,
        "cases": [],
    }
    try:
        for case in cases:
            output = {"id": case["id"], "question": case["question"], "arms": {}}
            for arm, queries in {"original": [case["question"]], "per_act": QUERIES[case["id"]]}.items():
                by_hash = {}
                per_query = []
                for query in queries:
                    candidates = await run_hybrid_search(
                        query,
                        30,
                        access_filter,
                        settings,
                        registry,
                        dense_weight=0.5,
                        sparse_weight=0.5,
                        visibility_conditions=conditions,
                        user_id=user_id,
                        user_group_ids=[],
                    )
                    per_query.append({"query": query, "candidates": [record(d) for d in candidates]})
                    for doc in candidates:
                        by_hash[content_hash(doc.page_content)] = doc
                ranked = await rerank_documents(
                    case["question"],
                    list(by_hash.values()),
                    top_n=8,
                    reranker=registry.reranker(),
                    min_score=settings.rerank.min_score,
                    score_gap_ratio=settings.rerank.score_gap_ratio,
                )
                context, selected = format_docs_with_selection(ranked, max_context_tokens=6000)
                output["arms"][arm] = {
                    "queries": per_query,
                    "merged_count": len(by_hash),
                    "ranked": [record(d, float(s)) for d, s in ranked],
                    "context": context,
                    "selected": [record(d, float(s)) for d, s in selected],
                }
                logger.info("Retrieved %s %s candidates=%s", case["id"], arm, len(by_hash))
            report["cases"].append(output)
    finally:
        await registry.close()
    return report


async def balance_saved_results(report):
    """Rerank saved per-query candidates, then reserve equal slots per part."""
    registry = MLClientRegistry()
    try:
        for case in report["cases"]:
            parts = case["arms"]["per_act"]["queries"]
            quota = 8 // len(parts)
            merged = {}
            rankings = []
            for part in parts:
                docs = [
                    Document(page_content=d["content"], metadata=d["metadata"]) for d in part["candidates"]
                ]
                ranked = await rerank_documents(
                    part["query"], docs, top_n=quota, reranker=registry.reranker()
                )
                rankings.append({"query": part["query"], "ranked": [record(d, float(s)) for d, s in ranked]})
                for doc, score in ranked:
                    merged.setdefault(content_hash(doc.page_content), (doc, score))
            # Fill unused/deduplicated slots from the original global ranking.
            for item in case["arms"]["per_act"]["ranked"]:
                if len(merged) >= 8:
                    break
                doc = Document(page_content=item["content"], metadata=item["metadata"])
                merged.setdefault(content_hash(doc.page_content), (doc, item["score"]))
            context, selected = format_docs_with_selection(list(merged.values()), max_context_tokens=6000)
            case["arms"]["per_act_balanced"] = {
                "quota_per_part": quota,
                "part_rankings": rankings,
                "context": context,
                "selected": [record(d, float(s)) for d, s in selected],
            }
    finally:
        await registry.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--from-results", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if args.from_results:
        result = asyncio.run(balance_saved_results(data))
    else:
        if args.user_id is None:
            parser.error("--user-id is required for live retrieval")
        result = asyncio.run(compare(data["cases"], args.user_id))
    logger.warning(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
