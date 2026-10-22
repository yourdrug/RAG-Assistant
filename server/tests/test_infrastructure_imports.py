"""Infrastructure dependencies must work without delayed project imports."""

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


APP = Path(__file__).resolve().parents[1] / "app"


def test_infrastructure_has_no_function_local_project_imports() -> None:
    roots = {path.stem for path in APP.iterdir()}
    local_imports = []

    for path in (APP / "infrastructure").rglob("*.py"):
        tree = ast.parse(path.read_text())

        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for node in ast.walk(function):
                if isinstance(node, ast.ImportFrom):
                    project_import = node.level or (node.module or "").split(".")[0] in roots
                elif isinstance(node, ast.Import):
                    project_import = any(alias.name.split(".")[0] in roots for alias in node.names)
                else:
                    continue

                if project_import:
                    local_imports.append(f"{path.relative_to(APP)}:{node.lineno}")

    assert not local_imports, local_imports


@pytest.mark.parametrize(
    "module",
    [
        "infrastructure.ml.rag.context_selection",
        "infrastructure.ml.rag.rag_formatting",
        "infrastructure.ml.clients.factories",
        "infrastructure.metrics.metrics",
        "infrastructure.benchmark.metrics",
    ],
)
def test_infrastructure_entrypoint_imports_in_fresh_process(module: str) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
        env={**os.environ, "PYTHONPATH": str(APP)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
