"""C-2 regression: hybrid-search BM25 hash resolution must enforce the ACL filter.

The BM25 index is global, so sparse hits may reference chunks outside the
caller's visibility scope. ``_resolve_hash_to_doc`` must combine the hash
condition with the caller's access filter so foreign private chunks can
never be resolved into candidates/sources.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain.schema import Document as LCDocument  # noqa: E402
from qdrant_client.models import FieldCondition, Filter  # noqa: E402

from infrastructure.acl import build_qdrant_filter  # noqa: E402
from infrastructure.ml.rag_service import _resolve_hash_to_doc  # noqa: E402

FOREIGN_HASH = "b" * 16
OWN_HASH = "a" * 16


# ---------------------------------------------------------------------------
# Fake Qdrant client that HONORS the filter it is given (like the real one)
# ---------------------------------------------------------------------------


def _match_field(cond: FieldCondition, payload: dict) -> bool:
    cur: object = payload
    for part in cond.key.split("."):
        if not isinstance(cur, dict):
            return False
        cur = cur.get(part)
    match = cond.match
    if hasattr(match, "value"):
        return cur == match.value
    if hasattr(match, "any"):
        return cur in match.any
    raise AssertionError(f"Unsupported match type: {type(match)}")


def _matches(node, payload: dict) -> bool:
    if isinstance(node, Filter):
        if node.must and not all(_matches(c, payload) for c in node.must):
            return False
        if node.must_not and any(_matches(c, payload) for c in node.must_not):
            return False
        if node.should and not any(_matches(c, payload) for c in node.should):
            return False
        return True
    if isinstance(node, FieldCondition):
        return _match_field(node, payload)
    raise AssertionError(f"Unsupported condition type: {type(node)}")


class _Point:
    def __init__(self, payload: dict):
        self.payload = payload


class _FakeQdrant:
    """scroll() that applies the received filter, like the real client."""

    def __init__(self, points: list[_Point]):
        self._points = points
        self.captured_filter = None

    def scroll(self, collection_name, scroll_filter, limit, with_payload):
        self.captured_filter = scroll_filter
        matched = [p for p in self._points if _matches(scroll_filter, p.payload)]
        return (matched, None)


class _FakeRegistry:
    def __init__(self, client: _FakeQdrant):
        self._client = client

    def qdrant_client(self):
        return self._client


def _point(content_hash: str, owner_id: int) -> _Point:
    return _Point(
        {
            "page_content": f"chunk of user {owner_id}",
            "metadata": {
                "content_hash": content_hash,
                "visibility": "client_private",
                "owner_id": owner_id,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHashResolutionAcl:
    def test_scroll_filter_combines_acl_with_hash(self):
        client = _FakeQdrant([])
        registry = _FakeRegistry(client)
        access_filter = build_qdrant_filter({"id": 1, "kind": "client"}, [])

        asyncio.run(_resolve_hash_to_doc(FOREIGN_HASH, access_filter, registry))

        captured = client.captured_filter
        assert captured is not None and captured.must, "scroll must receive a must-filter"
        # must = [hash condition, access filter] — both required
        hash_conds = [c for c in captured.must if isinstance(c, FieldCondition)]
        assert any(c.key == "metadata.content_hash" and c.match.value == FOREIGN_HASH for c in hash_conds)
        assert access_filter in captured.must, "the ACL filter must be part of the scroll filter"

    def test_foreign_private_hash_is_not_resolved(self):
        client = _FakeQdrant([_point(FOREIGN_HASH, owner_id=2), _point(OWN_HASH, owner_id=1)])
        registry = _FakeRegistry(client)
        access_filter = build_qdrant_filter({"id": 1, "kind": "client"}, [])

        doc = asyncio.run(_resolve_hash_to_doc(FOREIGN_HASH, access_filter, registry))

        assert doc is None, "a private chunk of another user must never be resolved by hash"

    def test_own_hash_is_resolved(self):
        client = _FakeQdrant([_point(FOREIGN_HASH, owner_id=2), _point(OWN_HASH, owner_id=1)])
        registry = _FakeRegistry(client)
        access_filter = build_qdrant_filter({"id": 1, "kind": "client"}, [])

        doc = asyncio.run(_resolve_hash_to_doc(OWN_HASH, access_filter, registry))

        assert isinstance(doc, LCDocument)
        assert doc.metadata["owner_id"] == 1
