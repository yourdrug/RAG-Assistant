"""Dataset names are available to administrators only."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from domain.exceptions import ClientException
from domain.value_objects.benchmark_dataset import BenchmarkDataset
from domain.value_objects.roles import UserKind, UserRole
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import (
    create_api_key_provider,
    create_auth_service,
    create_benchmark_question_service,
)
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.routes.benchmark_admin import router
from presentation.api.schemas import CurrentUser


@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.USER, None])
def test_dataset_catalog_requires_admin(role):
    datasets = [BenchmarkDataset.MAIN.value, "quick-30", "custom"]
    service = SimpleNamespace(list_datasets=AsyncMock(return_value=datasets))
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ClientException, handle_client_exception)
    app.dependency_overrides[create_auth_service] = lambda: None
    app.dependency_overrides[create_api_key_provider] = lambda: None
    app.dependency_overrides[create_benchmark_question_service] = lambda: service
    if role is not None:
        user = CurrentUser(
            id=7,
            email="catalog@example.com",
            role=role.value,
            kind=UserKind.INTERNAL.value,
            is_active=True,
        )
        app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        response = client.get("/admin/benchmark/datasets")
    if role == UserRole.ADMIN:
        assert response.status_code == 200
        assert response.json() == datasets
        service.list_datasets.assert_awaited_once_with()
    else:
        assert response.status_code == (401 if role is None else 403)
        service.list_datasets.assert_not_awaited()
