"""Review journal entries: triage status, conversation context and transfer to benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from typing import TYPE_CHECKING, Any, Literal

from domain.benchmark.question import BenchmarkQuestion, validate_history
from domain.chat.message import MessageRole
from domain.shared.exceptions import EntityNotFoundError, ValidationError


if TYPE_CHECKING:
    from datetime import date

    from application.chat.ports import ChatSettingsPort, PIIRedactorPort
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.chat.chat_log import ChatLog
    from domain.chat.message import Message
    from domain.chat.review import ChatLogReview, ReviewStatus

DEFAULT_CONTEXT_WINDOW = 8
JOURNAL_TAG = "from-journal"
MAX_SOURCE_HINT_CHARS = 1000


@dataclass(frozen=True)
class BenchmarkDraft:
    question: str
    history: list[dict[str, str]]
    as_of_date: date | None
    source_hint: str | None
    datasets: list[str]
    benchmark_question_id: int | None = None


@dataclass(frozen=True)
class BenchmarkTransfer:
    dataset: str
    question: str
    history: list[dict[str, str]] = field(default_factory=list)
    expected_answer: str | None = None
    source_hint: str | None = None
    as_of_date: date | Literal[""] | None = None
    annotations: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    is_active: bool = False


def source_hint(entry: ChatLog) -> str | None:
    names = dict.fromkeys(str(source["source"]) for source in entry.sources if source.get("source"))
    return "; ".join(names)[:MAX_SOURCE_HINT_CHARS] or None


class ChatLogReviewService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        pii_redactor: PIIRedactorPort | None = None,
        chat_settings: ChatSettingsPort | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.pii_redactor = pii_redactor
        self.chat_settings = chat_settings

    @property
    def context_window(self) -> int:
        """The same number of turns the answer itself could see."""
        return self.chat_settings.history_window if self.chat_settings is not None else DEFAULT_CONTEXT_WINDOW

    def redact(self, text: str) -> str:
        return self.pii_redactor.redact(text) if self.pii_redactor is not None else text

    def redact_annotation_value(self, value: object) -> object:
        if isinstance(value, str):
            return self.redact(value)

        if isinstance(value, list):
            return [self.redact_annotation_value(item) for item in value]

        if isinstance(value, dict):
            return {key: self.redact_annotation_value(item) for key, item in value.items()}

        return value

    async def set_status(
        self,
        log_id: int,
        status: ReviewStatus,
        note: str | None,
        reviewer_id: int,
        user_ids: tuple[int, ...] | None,
    ) -> ChatLogReview:
        async with self.uow_factory.create(master=True) as uow:
            await self.visible_entry(uow, log_id, user_ids)
            review = await uow.chat_logs.lock_review(log_id)
            review.set_status(status, note, reviewer_id)
            await uow.chat_logs.save_review(review)
            return review

    async def context(self, log_id: int, user_ids: tuple[int, ...] | None) -> list[Message]:
        async with self.uow_factory.create() as uow:
            entry = await self.visible_entry(uow, log_id, user_ids)
            return await self.previous_turns(uow, entry)

    async def draft(self, log_id: int, user_ids: tuple[int, ...] | None) -> BenchmarkDraft:
        async with self.uow_factory.create() as uow:
            entry = await self.visible_entry(uow, log_id, user_ids)
            turns = await self.previous_turns(uow, entry)
            reviews = await uow.chat_logs.list_reviews([log_id])
            return BenchmarkDraft(
                question=self.redact(entry.question),
                history=self.benchmark_history(turns),
                as_of_date=entry.as_of_date,
                source_hint=source_hint(entry),
                datasets=await self.datasets(uow, entry),
                benchmark_question_id=reviews[0].benchmark_question_id if reviews else None,
            )

    async def transfer(
        self, log_id: int, body: BenchmarkTransfer, reviewer_id: int, user_ids: tuple[int, ...] | None
    ) -> BenchmarkQuestion:
        question_text = self.redact(body.question).strip()

        if not question_text:
            msg = "question must not be empty"
            raise ValidationError(msg, field="question")

        expected_answer = self.redact(body.expected_answer or "").strip()

        if body.is_active and not expected_answer:
            msg = "An expected answer is required to create an active benchmark question"
            raise ValidationError(msg, field="expected_answer")

        # The form is editable: redact what comes back, not only what was offered.
        history = validate_history(
            [{"role": item["role"], "content": self.redact(item["content"])} for item in body.history]
        )
        annotations = (
            {key: self.redact_annotation_value(value) for key, value in body.annotations.items()}
            if body.annotations is not None
            else None
        )

        async with self.uow_factory.create(master=True) as uow:
            entry = await self.visible_entry(uow, log_id, user_ids)

            if body.dataset not in await self.datasets(uow, entry):
                msg = "dataset must be a benchmark dataset of the entry's pack"
                raise ValidationError(msg, field="dataset")

            # Locked first: a concurrent transfer waits here and then sees the question link.
            review = await uow.chat_logs.lock_review(log_id)

            review.ensure_not_transferred()
            notes = self.redact(body.notes or "").strip()
            sources = source_hint(entry) if body.source_hint is None else body.source_hint
            tags = dict.fromkeys([JOURNAL_TAG, *(self.redact(tag).strip() for tag in body.tags)])
            question = await uow.benchmark_questions.create(
                BenchmarkQuestion(
                    question=question_text,
                    expected_answer=expected_answer or None,
                    history=history or None,
                    as_of_date=entry.as_of_date if body.as_of_date is None else body.as_of_date or None,
                    source_hint=self.redact(sources or "").strip() or None,
                    annotations=annotations,
                    dataset=body.dataset,
                    is_active=body.is_active,
                    created_by=reviewer_id,
                    tags=[tag for tag in tags if tag],
                    notes=f"chat_log:{log_id}\n{notes}" if notes else f"chat_log:{log_id}",
                )
            )

            if question.id is None:
                msg = "Saved benchmark question has no id"
                raise RuntimeError(msg)

            review.mark_in_benchmark(question.id, reviewer_id)
            await uow.chat_logs.save_review(review)
            return question

    @staticmethod
    async def visible_entry(uow: UnitOfWork, log_id: int, user_ids: tuple[int, ...] | None) -> ChatLog:
        entry = await uow.chat_logs.get_by_id(log_id, user_ids)

        if entry is None:
            msg = "Chat log"
            raise EntityNotFoundError(msg, log_id)

        return entry

    @staticmethod
    async def datasets(uow: UnitOfWork, entry: ChatLog) -> list[str]:
        if not entry.pack_key:
            return []

        quick = f"{entry.pack_key}-quick"
        return [
            f"{entry.pack_key}-main",
            *([quick] if quick in await uow.benchmark_questions.get_datasets() else []),
        ]

    async def previous_turns(self, uow: UnitOfWork, entry: ChatLog) -> list[Message]:
        """Turns before this attempt, read only through a journal entry already in scope."""
        if entry.conversation_id is None:
            return []

        if entry.message_id is not None:
            # The answer's own question directly precedes the answer message.
            messages = await uow.messages.list_page(
                entry.conversation_id, limit=self.context_window + 1, before_id=entry.message_id
            )

            if messages and messages[-1].role == MessageRole.USER:
                messages = messages[:-1]

            return messages[-self.context_window :]

        # Errors and cancellations save no messages: everything earlier is context.
        # The date bound is applied before the limit, so later turns cannot push it out.
        return await uow.messages.list_page(
            entry.conversation_id, limit=self.context_window, created_before=entry.creation_date
        )

    def benchmark_history(self, turns: list[Message]) -> list[dict[str, str]]:
        return [
            {
                "role": turn.role.value,
                # User turns are stored HTML-escaped; a benchmark question holds plain text.
                "content": self.redact(
                    html.unescape(turn.content) if turn.role == MessageRole.USER else turn.content
                ),
            }
            for turn in turns
        ]
