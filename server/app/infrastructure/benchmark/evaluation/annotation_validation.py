"""Check reference quotes independently of retrieval, within the request access scope."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, Any

from domain.benchmark.evaluation import matches_fragment
from domain.benchmark.question import evaluation_annotations
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from domain.knowledge.pack_metadata import MetadataCondition
    from domain.rag.context import RagContext
    from domain.rag.ports import ChunkSearchRepository

    type ChunkSelection = Callable[[date | None], MetadataCondition | None]


REFERENCE_SEARCH_LIMIT = 100


class AnnotationValidator:
    def __init__(
        self, search: ChunkSearchRepository, *, chunk_selections: Mapping[str, ChunkSelection] | None = None
    ) -> None:
        self.search = search
        # Each pack's selection at the question date by pack key; validation runs outside a RAG request.
        self.chunk_selections = dict(chunk_selections or {})

    async def validate(self, annotations: dict[str, Any] | None, ctx: RagContext) -> dict[str, Any]:
        annotations = evaluation_annotations(annotations)
        fragments = (annotations or {}).get("expected_fragments") or []

        if not fragments:
            return {"status": "unvalidated", "reason": "no_reference_quotes"}

        missing = []
        verified_options = []
        truncated = False
        chunk_selection = self.chunk_selections.get(ctx.space.pack_key) if ctx.space is not None else None
        selection = chunk_selection(ctx.as_of_date) if chunk_selection is not None else None

        for index, fragment in enumerate(fragments):
            found = False
            verified = []

            for option_index, option in enumerate(fragment.get("alternatives", [fragment])):
                chunks = await self.search.search_substring(
                    option["text"],
                    ctx.user,
                    limit=REFERENCE_SEARCH_LIMIT,
                    document_id=option.get("document_id"),
                    selection=selection,
                    space_id=ctx.space.id if ctx.space is not None else None,
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
        return replace(ctx, as_of_date=date.fromisoformat(question["as_of_date"]))

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
