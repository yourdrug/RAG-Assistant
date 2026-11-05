"""Check reference quotes independently of retrieval, within the request access scope."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import TYPE_CHECKING, Any

from domain.benchmark.evaluation import matches_fragment
from domain.benchmark.question import evaluation_annotations
from domain.rag.context import RagContext
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata


if TYPE_CHECKING:
    from domain.rag.ports import ChunkSearchRepository


REFERENCE_SEARCH_LIMIT = 100


class AnnotationValidator:
    def __init__(self, search: ChunkSearchRepository) -> None:
        self.search = search

    async def validate(self, annotations: dict[str, Any] | None, ctx: RagContext) -> dict[str, Any]:
        annotations = evaluation_annotations(annotations)
        fragments = (annotations or {}).get("expected_fragments") or []

        if not fragments:
            return {"status": "unvalidated", "reason": "no_reference_quotes"}

        missing = []
        verified_options = []
        truncated = False

        for index, fragment in enumerate(fragments):
            found = False
            verified = []

            for option_index, option in enumerate(fragment.get("alternatives", [fragment])):
                chunks = await self.search.search_substring(
                    option["text"],
                    ctx.user,
                    limit=REFERENCE_SEARCH_LIMIT,
                    document_id=option.get("document_id"),
                    as_of_date=ctx.as_of_date,
                )
                truncated = truncated or len(chunks) == REFERENCE_SEARCH_LIMIT

                if any(
                    matches_fragment(
                        option, {"content": chunk.content, "metadata": chunk_result_metadata(chunk)}
                    )
                    for chunk in chunks
                ):
                    found = True
                    verified.append(option_index)
                    break

            verified_options.append(verified)

            if not found:
                missing.append(index)

        return {
            "status": "unvalidated"
            if missing and truncated
            else "invalid_annotation"
            if missing
            else "valid",
            "missing_fragments": missing,
            "verified_options": verified_options,
            "scope": "authorized SQL chunks at request date",
            "limitations": [
                "quote validation does not establish completeness of required conditions",
                "SQL/vector corpus consistency requires the run corpus manifest",
            ],
        }


def question_context(question: dict[str, Any], ctx: RagContext) -> RagContext:
    if question.get("as_of_date"):
        return RagContext(ctx.user, ctx.depth, ctx.summary, date.fromisoformat(question["as_of_date"]))

    return ctx


def report_annotations(
    annotations: dict[str, Any] | None, validation: dict[str, Any]
) -> dict[str, Any] | None:
    if validation["status"] == "invalid_annotation" or validation.get("missing_fragments"):
        return None

    if annotations is None or "verified_options" not in validation:
        return annotations

    safe = deepcopy(annotations)

    for fragment, verified in zip(
        safe.get("expected_fragments") or [], validation["verified_options"], strict=True
    ):
        if "alternatives" in fragment:
            fragment["alternatives"] = [
                option for index, option in enumerate(fragment["alternatives"]) if index in verified
            ]

    return safe
