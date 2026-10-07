"""Route-coverage guard: every rate-limited route uses exactly one known policy.

This test is the uniformity contract — adding a limited endpoint means adding
one line to the route *and* one entry to the expected mapping below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from main import create_application  # noqa: E402
from application.ports.rate_limit import RateLimitPolicyName  # noqa: E402

_EXPECTED: dict[tuple[str, str], str] = {
    # login (IP-scoped)
    ("POST", "/auth/login"): "login",
    # chat (user-scoped)
    ("POST", "/chat"): "chat",
    ("POST", "/chat/sync"): "chat",
    # upload
    ("POST", "/documents"): "upload",
    ("POST", "/ingest"): "upload",
    ("POST", "/ingest/file"): "upload",
    ("POST", "/upload"): "upload",
    # search
    ("POST", "/search/exact"): "search",
    # write
    ("POST", "/auth/users"): "write",
    ("PATCH", "/auth/users/{user_id}"): "write",
    ("PATCH", "/auth/users/{user_id}/role"): "write",
    ("POST", "/documents/manual"): "write",
    ("DELETE", "/documents/{document_id}"): "write",
    ("PATCH", "/documents/{document_id}/rename"): "write",
    ("POST", "/documents/{document_id}/chunks"): "write",
    ("PUT", "/documents/{document_id}/chunks/{chunk_id}"): "write",
    ("DELETE", "/documents/{document_id}/chunks/{chunk_id}"): "write",
    ("POST", "/groups"): "write",
    ("POST", "/groups/{group_id}/members"): "write",
    ("DELETE", "/groups/{group_id}/members/{user_id}"): "write",
    ("POST", "/clients/{client_user_id}/api-keys"): "write",
    ("DELETE", "/clients/{client_user_id}/api-keys/{api_key_id}"): "write",
    ("POST", "/admin/curators/{curator_id}/users/{target_user_id}"): "write",
    ("DELETE", "/admin/curators/{curator_id}/users/{target_user_id}"): "write",
    ("POST", "/admin/curators/{curator_id}/groups/{group_id}"): "write",
    ("DELETE", "/admin/curators/{curator_id}/groups/{group_id}"): "write",
    ("PUT", "/admin/config/{key}"): "write",
    ("PATCH", "/admin/act-versions/{version_id}"): "write",
    ("POST", "/admin/documents/{document_id}/diagnose"): "write",
    ("POST", "/admin/documents/preview"): "write",
    ("POST", "/admin/documents/preview-ocr"): "write",
    ("POST", "/admin/documents/preview/page-image"): "write",
    ("POST", "/admin/documents/preview/{preview_id}/index"): "write",
    ("POST", "/admin/benchmark/questions"): "write",
    ("PUT", "/admin/benchmark/questions/{question_id}"): "write",
    ("DELETE", "/admin/benchmark/questions/{question_id}"): "write",
    ("POST", "/admin/benchmark/questions/import"): RateLimitPolicyName.WRITE.value,
    ("POST", "/admin/benchmark/sweep/{sweep_id}/cancel"): "write",
    ("POST", "/admin/benchmark/sweep/{sweep_id}/resume"): RateLimitPolicyName.BENCHMARK.value,
    # benchmark
    ("POST", "/benchmark"): "benchmark",
    ("POST", "/admin/benchmark/sweep"): "benchmark",
    ("POST", "/admin/benchmark/runs/{run_id}/apply"): "benchmark",
}


def _marked_routes() -> dict[tuple[str, str], str]:
    app = create_application()
    marked: dict[tuple[str, str], str] = {}
    for route in app.routes:
        for dep in getattr(route, "dependencies", []):
            policy = getattr(dep.dependency, "rate_limit_policy", None)
            if policy is None:
                continue
            for method in route.methods:
                if method in ("HEAD", "OPTIONS"):
                    continue
                marked[(method, route.path)] = policy.value
    return marked


class TestRouteCoverage:
    def test_marked_routes_match_expected_mapping(self):
        marked = _marked_routes()
        assert marked == _EXPECTED

    def test_health_endpoint_is_not_rate_limited(self):
        marked = _marked_routes()
        assert ("GET", "/health") not in marked

    def test_read_endpoints_are_not_rate_limited(self):
        marked = _marked_routes()
        assert ("GET", "/documents") not in marked
        assert ("GET", "/admin/config") not in marked
        assert ("GET", "/conversations") not in marked
