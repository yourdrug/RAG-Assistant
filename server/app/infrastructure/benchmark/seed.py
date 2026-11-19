"""Benchmark seed of a domain pack: corpus files, their document profiles and a question snapshot.

Layout under the pack's seed directory::

    seed.json        {"space": "<space name>", "documents": {"<file>": "<document profile>"}}
    corpus/          the documents listed in seed.json, and nothing else
    questions.json   snapshot of the database datasets `<pack>-*`
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from typing import TYPE_CHECKING, Any

from domain.benchmark.question import BenchmarkQuestion
from domain.benchmark.question_snapshot import SNAPSHOT_FIELDS, snapshot_values
from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from infrastructure.domain_profile.packs import DomainPack


SEED_MANIFEST = "seed.json"
QUESTIONS_FILE = "questions.json"
CORPUS_DIR = "corpus"
# Storage prefix of seeded documents; the same key keeps re-seeding idempotent.
STORAGE_PREFIX = "benchmark-corpus"
SNAPSHOT_FORMAT = 1


@dataclass(frozen=True)
class SeedDocument:
    file: str
    domain: str
    path: Path


@dataclass(frozen=True)
class PackSeed:
    pack_key: str
    space_name: str
    documents: tuple[SeedDocument, ...]
    questions_path: Path

    @property
    def dataset_prefix(self) -> str:
        return f"{self.pack_key}-"

    def storage_key(self, document: SeedDocument) -> str:
        return f"{STORAGE_PREFIX}/{self.pack_key}/{document.file}"


def load_pack_seed(pack: DomainPack) -> PackSeed | None:
    """The pack's seed, or None when the pack ships none; an inconsistent seed is an error."""
    if pack.seed_dir is None or not (pack.seed_dir / SEED_MANIFEST).is_file():
        return None

    manifest_path = pack.seed_dir / SEED_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    space = manifest.get("space")
    listed = manifest.get("documents")

    if not isinstance(space, str) or not space.strip() or not isinstance(listed, dict):
        message = f"{manifest_path}: expected a space name and a documents object"
        raise ValidationError(message)

    corpus = pack.seed_dir / CORPUS_DIR
    present = {path.name for path in corpus.iterdir() if path.is_file()} if corpus.is_dir() else set()
    missing = sorted(set(listed) - present)
    unlisted = sorted(name for name in present - set(listed) if not name.startswith("."))

    if missing or unlisted:
        message = f"{manifest_path}: corpus mismatch; missing files {missing}, unlisted files {unlisted}"
        raise ValidationError(message)

    documents = tuple(
        SeedDocument(file=name, domain=str(domain), path=corpus / name)
        for name, domain in sorted(listed.items())
    )
    return PackSeed(pack.key, space.strip(), documents, pack.seed_dir / QUESTIONS_FILE)


def read_questions(path: Path) -> list[BenchmarkQuestion]:
    if not path.is_file():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict) or payload.get("format") != SNAPSHOT_FORMAT:
        message = f"{path}: unsupported question snapshot format"
        raise ValidationError(message)

    return [question_from_record(record, path) for record in payload.get("questions", [])]


def write_questions(path: Path, questions: Iterable[BenchmarkQuestion]) -> None:
    payload = {"format": SNAPSHOT_FORMAT, "questions": [question_record(question) for question in questions]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def question_record(question: BenchmarkQuestion) -> dict[str, Any]:
    record: dict[str, Any] = {}

    for name, value in snapshot_values(question).items():
        if value is None:
            continue

        record[name] = value.isoformat() if isinstance(value, date) else value

    return record


def question_from_record(record: object, path: Path) -> BenchmarkQuestion:
    if not isinstance(record, dict) or set(record) - set(SNAPSHOT_FIELDS):
        message = f"{path}: unexpected question record {record!r:.200}"
        raise ValidationError(message)

    if not isinstance(record.get("dataset"), str) or not isinstance(record.get("question"), str):
        message = f"{path}: a question record requires dataset and question"
        raise ValidationError(message)

    values = dict(record)
    as_of_date = values.get("as_of_date")

    if as_of_date is not None:
        values["as_of_date"] = date.fromisoformat(str(as_of_date))

    values.setdefault("is_active", True)
    return BenchmarkQuestion(**values)
