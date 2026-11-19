"""Stable task contracts shared by Redis producers, durable relay and workers."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from domain.benchmark.llm_replay import LLMReplayMode
from domain.knowledge.enums import DocumentVisibility
from domain.runtime.jobs import JobType
from infrastructure.worker.document_deletion import delete_document
from infrastructure.worker.queues import (
    BATCH_QUEUE,
    DOCUMENTS_QUEUE,
    MAINTENANCE_QUEUE,
)
from infrastructure.worker.sweep import run_sweep_task
from infrastructure.worker.tasks import process_document, run_benchmark, run_full_ingest, run_single_ingest


__all__ = ["BATCH_QUEUE", "DOCUMENTS_QUEUE", "MAINTENANCE_QUEUE", "TASKS", "tasks_for_queue"]


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cost_principal: str | None = None


class DocumentPayload(Payload):
    document_id: PositiveInt
    storage_key: str
    filename: str
    visibility: str
    owner_id: int | None
    group_id: int | None
    replace_id: int | None
    doc_domain: str | None = None
    principal_id: int | None = None


class DeletionPayload(Payload):
    document_id: PositiveInt
    user_id: PositiveInt


class IngestPayload(Payload):
    resolved_dir: str
    reset: bool
    domain: str
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value
    group_id: int | None = None
    client_id: int | None = None
    space_id: int | None = None


class FileIngestPayload(Payload):
    resolved: str
    domain: str
    force: bool = False
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value
    group_id: int | None = None
    client_id: int | None = None
    space_id: int | None = None


class BenchmarkPayload(Payload):
    space_id: PositiveInt | None = None
    dataset: str
    top_k: PositiveInt
    judge_model: str
    out_dir: str | None = None
    # Hash of the questions at enqueue time; the worker refuses a different set.
    dataset_hash: str | None = None
    # Stored as its string value in the job row; strict mode would reject the string.
    llm_replay_mode: Annotated[LLMReplayMode, Field(strict=False)] = LLMReplayMode.OFF
    # Chat answer-length selector to benchmark; None is "Авто".
    depth: Literal["narrow", "broad"] | None = None


class SweepPayload(Payload):
    sweep_id: PositiveInt


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    handler: Callable[..., Coroutine[Any, Any, None]]
    payload: type[Payload]
    timeout: int
    queue: str
    version: int = 1

    def validate(self, payload: dict[str, object], version: int) -> dict[str, Any]:
        if version != self.version:
            message = f"Unsupported payload version {version} for {self.name}"
            raise ValueError(message)

        return self.payload.model_validate(payload).model_dump(exclude_unset=True)


TASKS = {
    JobType.DOCUMENT_PROCESSING: TaskDefinition(
        "process_document", process_document, DocumentPayload, 1800, DOCUMENTS_QUEUE
    ),
    JobType.DOCUMENT_DELETION: TaskDefinition(
        "delete_document", delete_document, DeletionPayload, 1800, DOCUMENTS_QUEUE
    ),
    JobType.INGEST: TaskDefinition("run_full_ingest", run_full_ingest, IngestPayload, 7200, BATCH_QUEUE),
    JobType.INGEST_FILE: TaskDefinition(
        "run_single_ingest", run_single_ingest, FileIngestPayload, 1800, BATCH_QUEUE
    ),
    JobType.BENCHMARK: TaskDefinition("run_benchmark", run_benchmark, BenchmarkPayload, 3600, BATCH_QUEUE),
    JobType.SWEEP: TaskDefinition("run_sweep", run_sweep_task, SweepPayload, 21600, BATCH_QUEUE),
}


def tasks_for_queue(queue: str) -> list[TaskDefinition]:
    return [task for task in TASKS.values() if task.queue == queue]
