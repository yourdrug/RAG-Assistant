"""Service factories — creates services that need infrastructure dependencies.

These factory methods were previously in InfrastructureContainer but violated
the dependency direction (infrastructure → application).  Now they live here,
in the composition layer, which is allowed to depend on both infrastructure
and application.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from composition._utils import _require

if TYPE_CHECKING:
    from composition.infrastructure import InfrastructureContainer
    from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


def create_ingestion_service(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
):
    """Create an IngestionService using infrastructure singletons.

    When *uow_factory* is ``None`` falls back to ``infra.uow_factory``.
    This is useful for CLI commands that don't need a database connection.
    """
    from application.services.act_versioning_service import ActVersioningService
    from infrastructure.services.ingestion_service import IngestionService

    uow = _require(
        uow_factory if uow_factory is not None else infra.uow_factory,
        "uow_factory",
    )

    act_versioning = None
    if infra.domain_registry is not None and infra.domain_settings is not None:
        # Unified versioning: CLI ingestion creates act versions the same way
        # the API upload path does (TZ section 8.4)
        act_versioning = ActVersioningService(uow_factory=uow, settings=infra.domain_settings)

    return IngestionService(
        vector_store_repo=_require(infra.vector_store_repo, "vector_store_repo"),
        file_storage=_require(infra.file_storage, "file_storage"),
        uow_factory=uow,
        domain_registry=infra.domain_registry,
        domain_settings=infra.domain_settings,
        act_versioning_service=act_versioning,
    )


def create_document_processor(
    infra: InfrastructureContainer,
    uow_factory: UnitOfWorkFactory | None = None,
):
    """Create a DocumentProcessor using infrastructure singletons."""
    from application.services.act_versioning_service import ActVersioningService
    from application.services.document_processor import DocumentProcessor
    from config import settings

    uow = _require(
        uow_factory if uow_factory is not None else infra.uow_factory,
        "uow_factory",
    )

    act_versioning = None
    if infra.domain_registry is not None:
        act_versioning = ActVersioningService(
            uow_factory=uow,
            settings=infra.domain_settings,
        )

    return DocumentProcessor(
        uow_factory=uow,
        vector_store_repo=_require(infra.vector_store_repo, "vector_store_repo"),
        file_storage=_require(infra.file_storage, "file_storage"),
        document_parser=_require(infra.document_parser, "document_parser"),
        document_splitter=_require(infra.document_splitter, "document_splitter"),
        content_extractor=_require(infra.content_extractor, "content_extractor"),
        pdf_quality_assessor=_require(infra.pdf_quality_assessor, "pdf_quality_assessor"),
        metrics=_require(infra.metrics_collector, "metrics_collector"),
        domain_marker_threshold=settings.document_domain_marker_threshold,
        ml_registry=infra.ml_clients,
        domain_registry=infra.domain_registry,
        domain_settings=infra.domain_settings,
        act_versioning_service=act_versioning,
    )
