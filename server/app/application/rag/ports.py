"""Stage contracts for application-owned RAG orchestration."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from application.rag.dto import (
    CompletedAnswer,
    PreparedContext,
    PreparedQuery,
    RagRequest,
    RetrievalResult,
    SanitizedAnswer,
    Sufficiency,
    TerminalResponse,
)
from domain.rag.context import RagContext


CorpusLease = Callable[[], AbstractAsyncContextManager[None]]


class RequestPreparation(Protocol):
    def request(self, question: str, history: list[object], ctx: RagContext) -> RagRequest: ...
    async def condense(self, request: RagRequest) -> PreparedQuery: ...


class InputSafety(Protocol):
    async def scan(self, query: PreparedQuery) -> TerminalResponse | None: ...


class AnswerCache(Protocol):
    async def lookup(self, query: PreparedQuery) -> PreparedQuery | TerminalResponse: ...
    async def store(self, answer: CompletedAnswer) -> None: ...


class Retrieval(Protocol):
    async def retrieve(
        self, query: PreparedQuery, previous: RetrievalResult | None = None
    ) -> RetrievalResult: ...


class EvidenceGate(Protocol):
    async def reject_ood(self, result: RetrievalResult) -> TerminalResponse | None: ...
    async def relevant(self, result: RetrievalResult) -> bool: ...
    async def assess(self, result: RetrievalResult) -> Sufficiency: ...
    async def reject(self, result: RetrievalResult) -> TerminalResponse: ...
    def retry(self, refinement: str) -> None: ...


class ContextPreparation(Protocol):
    def build(self, result: RetrievalResult) -> PreparedContext: ...


class Generation(Protocol):
    async def generate(self, context: PreparedContext) -> SanitizedAnswer: ...


class Finalization(Protocol):
    def finalize(self, answer: SanitizedAnswer) -> CompletedAnswer: ...
