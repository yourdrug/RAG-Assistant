"""Record reproducibility identities without credentials or document contents."""

from __future__ import annotations

from hashlib import sha256
import logging
from pathlib import Path
from shutil import which
import subprocess
from typing import Any

from application.benchmark.identity import fingerprint
from infrastructure.benchmark.judging import judge_rubric
from infrastructure.benchmark.sweep.settings import settings_snapshot
from infrastructure.ml.rag import prompt_blocks, rag_prompts


log = logging.getLogger(__name__)


# The `app` package; containers ship it without `.git`, so its source is hashed directly.
APP_ROOT = Path(__file__).resolve().parents[3]


def source_digest(root: Path = APP_ROOT) -> str:
    """SHA-256 of every Python module path and content; the only files that shape the pipeline."""
    digest = sha256()

    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).digest())

    return digest.hexdigest()


def code_identity() -> dict[str, Any]:
    return {**git_identity(), "source_sha256": source_digest()}


def git_identity() -> dict[str, Any]:
    git = which("git")

    if git is None:
        return {"revision": None, "tracked_diff_sha256": None}

    try:
        revision = subprocess.run(  # noqa: S603 — executable resolved by which; fixed read-only arguments.
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        changes = subprocess.run(  # noqa: S603 — only the digest of the tracked patch is persisted.
            [git, "diff", "--binary", "HEAD", "--", "server/app", "scripts/benchmark"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return {
            "revision": revision.stdout.strip(),
            "tracked_diff_sha256": sha256(changes.stdout).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("Benchmark code revision unavailable: %s", type(error).__name__)
        return {"revision": None, "tracked_diff_sha256": None}


def build_run_manifest(
    questions: list[dict[str, Any]], corpus: dict[str, Any], judge_model: str
) -> dict[str, Any]:
    config = settings_snapshot()
    return {
        "evidence_schema_version": 2,
        "code": code_identity(),
        "corpus": corpus,
        "annotation_version": fingerprint(questions),
        "configuration_version": fingerprint(config),
        "models": {key: value for key, value in config.items() if "model" in key},
        "judge_model": judge_model,
        "judge_rubric_version": judge_rubric.JUDGE_RUBRIC_VERSION,
        "template_versions": {
            module.__name__: sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (rag_prompts, prompt_blocks, judge_rubric)
            if module.__file__ is not None
        },
        "limitations": [
            "external model names may refer to mutable provider artifacts",
            "git identity is empty without a checkout; source_sha256 covers the Python modules of app",
            "tracked diff excludes untracked files; exact generation messages are in prompt snapshots",
            "SQL chunks and vector corpus are not a transactional cross-store snapshot",
        ],
    }
