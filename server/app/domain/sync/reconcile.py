"""What a run changes: the difference between a full storage snapshot and the channel's documents.

Pure: the caller supplies the files the snapshot publishes (already filtered by folder,
exclusions and ``.ragignore``) and the channel's documents with the file version each
published. A file that disappeared while exactly one new path holds the same content
is a rename and keeps its document; any other disappearance is a deletion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class PublishedFile:
    document_id: int
    # ``None`` until a version was committed (e.g. a first publication that failed).
    version: str | None
    # A version already being indexed; it becomes ``version`` once published.
    pending: str | None = None

    def holds(self, version: str) -> bool:
        """The document publishes ``version`` or is already indexing it."""
        return version in {self.version, self.pending}


@dataclass(frozen=True)
class Rename:
    document_id: int
    old_path: str
    new_path: str


@dataclass
class ReconcilePlan:
    added: list[str] = field(default_factory=list)
    # Path -> document whose content changed.
    updated: dict[str, int] = field(default_factory=dict)
    renamed: list[Rename] = field(default_factory=list)
    # Path -> document of a file that left the snapshot.
    deleted: dict[str, int] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)


def reconcile(desired: Mapping[str, str], published: Mapping[str, PublishedFile]) -> ReconcilePlan:
    """``desired`` maps each published path of the snapshot to its file version."""
    plan = ReconcilePlan()
    appeared = compare_present(desired, published, plan)
    gone: dict[str, list[str]] = defaultdict(list)

    for path, current in sorted(published.items()):
        if path not in desired:
            gone[current.version or ""].append(path)

    for version, new_paths in appeared.items():
        old_paths = gone.get(version, [])

        if version and len(new_paths) == 1 and len(old_paths) == 1:
            old_path = old_paths.pop()
            plan.renamed.append(Rename(published[old_path].document_id, old_path, new_paths[0]))
        else:
            plan.added.extend(new_paths)

    plan.deleted = {path: published[path].document_id for paths in gone.values() for path in paths}
    plan.added.sort()
    return plan


def compare_present(
    desired: Mapping[str, str], published: Mapping[str, PublishedFile], plan: ReconcilePlan
) -> dict[str, list[str]]:
    """Record unchanged and updated files; return new paths grouped by their version."""
    appeared: dict[str, list[str]] = defaultdict(list)

    for path, version in sorted(desired.items()):
        current = published.get(path)

        if current is None:
            appeared[version].append(path)
        elif current.holds(version):
            plan.unchanged.append(path)
        else:
            plan.updated[path] = current.document_id

    return appeared
