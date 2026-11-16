"""Answer a question with the RAG rules of the pack of the space it is asked in."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.spaces.ports import DomainPackNotLoadedError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from application.rag.service import RagService
    from application.rag.stream_events import StreamEvent
    from domain.rag.context import RagContext
    from domain.rag.result import RagResult


class SpaceRequiredError(ValueError):
    def __init__(self) -> None:
        super().__init__("A question is asked in one knowledge space")


class SpaceRagRouter:
    """One RAG service per loaded pack; the question's space chooses it."""

    def __init__(self, services: Mapping[str, RagService]) -> None:
        self._services = dict(services)

    def service(self, ctx: RagContext) -> RagService:
        if ctx.space is None:
            raise SpaceRequiredError

        service = self._services.get(ctx.space.pack_key)

        if service is None:
            raise DomainPackNotLoadedError(ctx.space.pack_key)

        return service

    def stream(self, question: str, history: Sequence[object], ctx: RagContext) -> AsyncIterator[StreamEvent]:
        return self.service(ctx).stream(question, history, ctx)

    async def invoke(self, question: str, history: Sequence[object], ctx: RagContext) -> RagResult:
        return await self.service(ctx).invoke(question, history, ctx)
