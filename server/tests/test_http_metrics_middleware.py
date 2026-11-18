"""MetricsMiddleware records requests through the container's HTTP metrics adapter."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from presentation.api.middleware.metrics import MetricsMiddleware


class RecordingHttpMetrics:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def inc_requests(self, *, handler: str, method: str, status: str) -> None:
        self.calls.append({"handler": handler, "method": method, "status": status})


def test_middleware_counts_request_by_route_template() -> None:
    recorder = RecordingHttpMetrics()
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    app.state.container = SimpleNamespace(
        infrastructure=SimpleNamespace(services=SimpleNamespace(http_metrics=recorder)),
    )

    @app.get("/items/{item_id}")
    def read_item(item_id: int) -> dict[str, int]:
        return {"id": item_id}

    response = TestClient(app).get("/items/42")

    assert response.status_code == 200
    assert recorder.calls == [{"handler": "/items/{item_id}", "method": "GET", "status": "200"}]
