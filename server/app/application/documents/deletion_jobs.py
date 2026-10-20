"""Authorize and track queued deletion without keeping an HTTP request open."""

from application.identity import UserContextFactory
from application.ports.job_enqueuer import JobEnqueuerPort
from application.uow import UnitOfWork, UnitOfWorkFactory
from domain.knowledge.access_control import check_ownership
from domain.runtime.jobs import BackgroundJob, JobType
from domain.shared.exceptions import EntityNotFound


class DocumentDeletionJobs:
    def __init__(self, uow_factory: UnitOfWorkFactory, enqueuer: JobEnqueuerPort) -> None:
        self._uow_factory = uow_factory
        self._enqueuer = enqueuer
        self._contexts = UserContextFactory()

    async def authorize(
        self, uow: UnitOfWork, document_id: int, user_id: int, user_kind: str, user_role: str
    ) -> None:
        doc = await uow.documents.get_by_id(document_id)

        if doc is None:
            msg = "Document"
            raise EntityNotFound(msg, document_id)

        ctx = await self._contexts.build(uow, user_id, user_kind, user_role)
        check_ownership(doc, ctx, "delete")

    async def enqueue(self, document_id: int, user_id: int, user_kind: str, user_role: str) -> int:
        async with self._uow_factory.create(master=True) as uow:
            await self.authorize(uow, document_id, user_id, user_kind, user_role)
            job = await uow.background_jobs.create(
                BackgroundJob(job_type=JobType.DOCUMENT_DELETION, related_id=document_id)
            )

            if job.id is None:
                msg = "Deletion job has no ID"
                raise RuntimeError(msg)

            job_id = job.id
        try:
            await self._enqueuer.enqueue_document_deletion(
                document_id=document_id, user_id=user_id, job_id=job_id
            )
        except Exception:
            async with self._uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, "Failed to enqueue document deletion")
            raise
        return job_id

    async def get(
        self, document_id: int, job_id: int, user_id: int, user_kind: str, user_role: str
    ) -> BackgroundJob:
        async with self._uow_factory.create(master=True) as uow:
            # Once the document is gone this returns 404, like GET /documents/{id}.
            # Existing jobs are visible only to users who may delete this document.
            await self.authorize(uow, document_id, user_id, user_kind, user_role)
            job = await uow.background_jobs.get_by_id(job_id)

            if job is None or job.related_id != document_id or job.job_type != JobType.DOCUMENT_DELETION:
                msg = "DocumentDeletionJob"
                raise EntityNotFound(msg, job_id)

            return job
