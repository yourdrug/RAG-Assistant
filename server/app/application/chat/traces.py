"""Read trace fragments through journal scope and the viewer's current listing ACL."""

from copy import deepcopy
from typing import Any

from application.identity import UserContextFactory
from application.uow import UnitOfWorkFactory
from domain.identity.user_context import UserContext
from domain.knowledge.access_control import can_view_document, validate_chunk_acl
from domain.shared.exceptions import EntityNotFoundError
from domain.shared.hashing import content_hash


def trace_chunks(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    chunks = {}

    for attempt in data.get("attempts", []):
        for item in [
            *attempt["exclusions"],
            *[doc for stage in attempt["stages"].values() for doc in stage["documents"]],
        ]:
            chunk_id = item.get("metadata", {}).get("chunk_id")

            if isinstance(chunk_id, int):
                if chunk_id not in chunks:
                    chunks[chunk_id] = {**item, "content_hashes": []}

                chunks[chunk_id]["content_hashes"].append(item.get("content_hash"))

    return chunks


def public_trace(data: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(data)

    for attempt in result.get("attempts", []):
        for check in attempt.get("checks", []):
            check["result"].pop("reasoning", None)

    return result


class ChatTraceService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self.uow_factory = uow_factory

    async def get_trace(self, log_id: int, user_ids: tuple[int, ...] | None) -> dict[str, Any]:
        async with self.uow_factory.create() as uow:
            entry = await uow.chat_logs.get_by_id(log_id, user_ids)
            data = await uow.chat_logs.get_trace(log_id) if entry is not None else None

            if data is None:
                msg = "Trace"
                raise EntityNotFoundError(msg, log_id)

            return public_trace(data)

    async def read_chunks(  # noqa: PLR0915 - ordered journal, membership and content ACL checks
        self, log_id: int, chunk_ids: list[int], viewer: UserContext, user_ids: tuple[int, ...] | None
    ) -> dict[str, Any]:
        async with self.uow_factory.create() as uow:
            entry = await uow.chat_logs.get_by_id(log_id, user_ids)
            data = await uow.chat_logs.get_trace(log_id) if entry is not None else None

            if data is None:
                msg = "Trace"
                raise EntityNotFoundError(msg, log_id)

            references = trace_chunks(data)

            if any(chunk_id not in references for chunk_id in chunk_ids):
                msg = "Trace chunk"
                raise EntityNotFoundError(msg, log_id)

            # Never use the original requester's identity or groups to read content.
            ctx = await UserContextFactory().build(uow, viewer.user_id, viewer.user_kind, viewer.user_role)
            results = []

            for chunk_id in dict.fromkeys(chunk_ids):
                reference = references[chunk_id]
                document_id = reference["metadata"].get("document_id")
                document = (
                    await uow.documents.get_by_id(document_id) if isinstance(document_id, int) else None
                )
                result: dict[str, Any] = {"chunk_id": chunk_id}

                if document is None:
                    result["missing"] = True
                elif not can_view_document(document, ctx):
                    result["hidden"] = True
                else:
                    chunk = await uow.chunks.get_by_id(chunk_id)

                    if chunk is None or chunk.document_id != document_id:
                        result["missing"] = True
                    else:
                        validate_chunk_acl(document, [chunk])
                        result.update(
                            content=chunk.content,
                            changed=any(
                                content_hash(chunk.content) != saved for saved in reference["content_hashes"]
                            ),
                        )

                results.append(result)

            # Judge reasoning may quote any input fragment. Release it only when
            # the viewer can read every unchanged fragment, with no omitted tail.
            identities_complete = all(
                isinstance(item["metadata"].get("chunk_id"), int)
                and isinstance(item["metadata"].get("document_id"), int)
                for attempt in data.get("attempts", [])
                for stage in attempt["stages"].values()
                for item in stage["documents"]
            )
            all_visible = (
                identities_complete
                and bool(references)
                and set(chunk_ids) == set(references)
                and not data.get("truncated")
                and all("content" in item and not item["changed"] for item in results)
            )
            checks = (
                [check for attempt in data.get("attempts", []) for check in attempt.get("checks", [])]
                if all_visible
                else []
            )
            return {"chunks": results, "checks": checks}
