"""Scraping has a dedicated read-only credential; metrics remain private."""

from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from domain.shared.exceptions import AppClientError
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.middleware import prometheus_setup


@pytest.mark.parametrize("authorization,expected", [(None, 401), ("Bearer scrape-only", 200)])
def test_scrape_token_allows_only_authenticated_metrics(
    monkeypatch: pytest.MonkeyPatch,
    authorization: str | None,
    expected: int,
) -> None:
    monkeypatch.setattr(
        prometheus_setup,
        "request_section",
        lambda _: SimpleNamespace(
            metrics_scrape_token=SimpleNamespace(get_secret_value=lambda: "scrape-only"),
        ),
        raising=False,
    )
    app = FastAPI()
    app.add_exception_handler(AppClientError, handle_client_exception)

    @app.get("/metrics", dependencies=[Depends(prometheus_setup.require_metrics_access)])
    def metrics() -> dict[str, bool]:
        return {"scraped": True}

    response = TestClient(app).get(
        "/metrics", headers={"Authorization": authorization} if authorization else {}
    )
    assert response.status_code == expected
