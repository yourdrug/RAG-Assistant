"""
services/chat_service.py — Chat business logic.
"""

import json
import logging
from collections.abc import AsyncIterator

from config import settings
from domain.rag import (
    build_prompt,
    extract_sources,
    format_docs,
    history_to_messages,
    rerank_documents,
)
from infrastructure.acl import build_qdrant_filter
from infrastructure.clients import get_llm, get_reranker, get_vector_store
from infrastructure.database import (
    get_history,
    get_or_create_conversation,
    save_message,
)
from sqlalchemy.orm import Session

log = logging.getLogger("default")


class ChatService:
    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user: dict,
        db: Session,
    ) -> AsyncIterator[str]:
        conv_id = get_or_create_conversation(db, conversation_id, user["id"])
        save_message(db, conv_id, "user", question)

        history = get_history(db, conv_id, window=settings.history_window)
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        full_answer = ""
        sources: list[dict] = []

        async for chunk in self._rag_stream(question, history, user, db):
            if chunk.startswith("\n__sources__:"):
                sources = json.loads(chunk.replace("\n__sources__:", ""))
            else:
                full_answer += chunk
                yield chunk

        save_message(db, conv_id, "assistant", full_answer, sources=sources)
        yield f"\n__meta__:{json.dumps({'conversation_id': conv_id, 'sources': sources}, ensure_ascii=False)}"

    async def sync_chat(
        self,
        question: str,
        conversation_id: int | None,
        user: dict,
        db: Session,
    ) -> tuple[str, list[dict], int]:
        conv_id = get_or_create_conversation(db, conversation_id, user["id"])
        save_message(db, conv_id, "user", question)

        history = get_history(db, conv_id, window=settings.history_window)
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        answer, sources = await self._rag_invoke(question, history, user, db)
        save_message(db, conv_id, "assistant", answer, sources=sources)

        return answer, sources, conv_id

    async def _rag_stream(
        self,
        question: str,
        history: list[dict],
        current_user: dict,
        db: Session,
    ) -> AsyncIterator[str]:
        access_filter = build_qdrant_filter(current_user, db)

        retriever = get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_fetch_k, "filter": access_filter},
        )
        prompt = build_prompt()

        candidates = retriever.invoke(question)
        docs = rerank_documents(question, candidates, top_n=settings.retriever_top_k, reranker=get_reranker())

        context = format_docs(docs)
        sources = extract_sources(docs)
        history_messages = history_to_messages(history)

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

    async def _rag_invoke(
        self,
        question: str,
        history: list[dict],
        current_user: dict,
        db: Session,
    ) -> tuple[str, list[dict]]:
        answer_parts: list[str] = []
        sources: list[dict] = []

        async for chunk in self._rag_stream(question, history, current_user, db):
            if chunk.startswith("\n__sources__:"):
                sources = json.loads(chunk.replace("\n__sources__:", ""))
            else:
                answer_parts.append(chunk)

        return "".join(answer_parts), sources
