"""Convert LangChain documents and messages to and from stage DTOs."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from application.rag.dto import PromptMessage, ScoredDocument
from domain.knowledge.chunk import Chunk


if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def chunk_value(doc: LCDocument) -> Chunk:
    return Chunk(content=doc.page_content, metadata=deepcopy(doc.metadata))


def scored_values(docs: list[tuple[LCDocument, float]]) -> tuple[ScoredDocument, ...]:
    return tuple(ScoredDocument(chunk_value(doc), float(score)) for doc, score in docs)


def lc_documents(docs: tuple[ScoredDocument, ...]) -> list[tuple[LCDocument, float]]:
    return [
        (LCDocument(page_content=d.chunk.content, metadata=deepcopy(d.chunk.metadata)), d.score) for d in docs
    ]


def message_values(messages: list[Any]) -> tuple[PromptMessage, ...]:
    return tuple(PromptMessage(m.type, m.content) for m in messages)


def lc_messages(messages: tuple[PromptMessage, ...]) -> list[BaseMessage]:
    classes = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
    return [classes[m.role](content=m.content) for m in messages]
