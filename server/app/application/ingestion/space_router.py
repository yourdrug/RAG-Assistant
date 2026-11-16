"""Ingest storage files into a knowledge space with the rules of the space's pack."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.spaces.ports import DomainPackNotLoadedError
from application.spaces.resolution import resolve_space
from domain.knowledge.enums import DocumentVisibility, DomainDetectionMode


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from application.ingestion.dto import IngestRegistryResult, IngestStatusResult
    from application.ingestion.ports.upload import UploadFile
    from application.ingestion.service import IngestionService
    from application.uow import UnitOfWorkFactory


class SpaceIngestionService:
    """One ingestion service per loaded pack; storage operations do not depend on a pack."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        services: Mapping[str, IngestionService],
        storage: IngestionService,
    ) -> None:
        self._uow_factory = uow_factory
        self._services = dict(services)
        self._storage = storage

    async def for_space(self, space_id: int | None) -> tuple[IngestionService, int]:
        async with self._uow_factory.create() as uow:
            space = await resolve_space(uow, space_id)

        service = self._services.get(space.pack_key)

        if service is None:
            raise DomainPackNotLoadedError(space.pack_key)

        if space.id is None:
            msg = "Space is not persisted"
            raise RuntimeError(msg)

        return service, space.id

    async def run_full(  # noqa: PLR0913 - mirrors IngestionService.run_full and adds the space.
        self,
        docs_dir: str | None = None,
        *,
        reset: bool = False,
        domain: str = DomainDetectionMode.AUTO.value,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
        job_id: int | None = None,
        space_id: int | None = None,
    ) -> IngestStatusResult:
        service, resolved_space = await self.for_space(space_id)
        return await service.run_full(
            docs_dir,
            reset=reset,
            domain=domain,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
            job_id=job_id,
            space_id=resolved_space,
        )

    async def run_single(
        self,
        file_path: str,
        *,
        force: bool = False,
        domain: str = DomainDetectionMode.AUTO.value,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
        space_id: int | None = None,
    ) -> IngestStatusResult:
        service, resolved_space = await self.for_space(space_id)
        return await service.run_single(
            file_path,
            force=force,
            domain=domain,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
            space_id=resolved_space,
        )

    async def get_registry(self) -> IngestRegistryResult:
        return await self._storage.get_registry()

    def resolve_docs_dir(self, prefix: str) -> str:
        return self._storage.resolve_docs_dir(prefix)

    def resolve_ingest_target(self, file_path: str) -> str:
        return self._storage.resolve_ingest_target(file_path)

    async def force_reindex(self, filename: str) -> None:
        await self._storage.force_reindex(filename)

    async def upload_files(self, files: Sequence[UploadFile], prefix: str = "docs/") -> list[str]:
        return await self._storage.upload_files(files, prefix)
