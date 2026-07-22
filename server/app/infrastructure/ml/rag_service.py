"""RAG Service — infrastructure implementation of the rag_service protocol used by chat use cases."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from config import settings

from infrastructure.acl import build_qdrant_filter
from infrastructure.clients import get_llm, get_reranker, get_vector_store
from infrastructure.ml.rag import (
    build_prompt,
    extract_sources,
    format_docs,
    history_to_messages,
    rerank_documents,
)

log = logging.getLogger("default")


class RagService:
    async def stream(
        self,
        question: str,
        history: list,
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> AsyncIterator[str]:
        user = {"id": user_id, "kind": user_kind}
        access_filter = build_qdrant_filter(user, user_group_ids, assigned_client_ids)

        retriever = get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_fetch_k, "filter": access_filter},
        )
        prompt = build_prompt()

        candidates = retriever.invoke(question)
        docs = rerank_documents(question, candidates, top_n=settings.retriever_top_k, reranker=get_reranker())

        context = format_docs(docs)
        sources = extract_sources(docs)

        history_dicts = []
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                history_dicts.append(
                    {
                        "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                        "content": msg.content,
                    }
                )
            elif isinstance(msg, dict):
                history_dicts.append(msg)
        history_messages = history_to_messages(history_dicts)

        messages = prompt.format_messages(
            context=context,
            history=history_messages,
            question=question,
        )

        async for chunk in get_llm().astream(messages):
            text = chunk.content
            if text:
                yield text

        yield f"\n__sources__:{json.dumps(sources, ensure_ascii=False)}"

    async def invoke(
        self,
        question: str,
        history: list,
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> tuple[str, list[dict]]:
        answer_parts: list[str] = []
        sources: list[dict] = []

        async for chunk in self.stream(
            question, history, user_id, user_kind, user_group_ids, assigned_client_ids
        ):
            if chunk.startswith("\n__sources__:"):
                sources = json.loads(chunk.replace("\n__sources__:", ""))
            else:
                answer_parts.append(chunk)

        return "".join(answer_parts), sources
