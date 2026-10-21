"""Regression cases for the symbol-level domain migration tool."""

import ast
from importlib import import_module
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "domain_imports.py"
REWRITE = runpy.run_path(str(SCRIPT))["rewrite"]
MOVES = {
    "domain.utils.content_hash": "domain.shared.hashing.content_hash",
    "domain.utils.parse_bool": "domain.runtime.config.parse_bool",
    "domain.value_objects.chat_context.ChatContext": "domain.rag.context.RagContext",
    "domain.value_objects.ChatContext": "domain.rag.context.RagContext",
}


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            'from domain.utils import content_hash as digest\nvalue = digest("x")\n',
            'from domain.shared.hashing import content_hash as digest\nvalue = digest("x")\n',
        ),
        (
            'from domain.utils import content_hash, parse_bool\nx = content_hash("x")\n',
            (
                "from domain.shared.hashing import content_hash\n"
                "from domain.runtime.config import parse_bool\n"
                'x = content_hash("x")\n'
            ),
        ),
        (
            "from domain.utils import content_hash  # public export\n",
            "from domain.shared.hashing import content_hash  # public export\n",
        ),
        (
            "from domain.value_objects import ChatContext\nx = ChatContext()\n",
            "from domain.rag.context import RagContext\nx = RagContext()\n",
        ),
        (
            "from domain.value_objects.chat_context import ChatContext as Context\nx = Context()\n",
            "from domain.rag.context import RagContext as Context\nx = Context()\n",
        ),
    ],
)
def test_symbol_imports(source: str, expected: str) -> None:
    result = REWRITE(source, MOVES)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))
    assert REWRITE(result, MOVES) == result


@pytest.mark.parametrize(
    "statement,reference",
    [
        ("import domain.utils", "domain.utils.content_hash"),
        ("import domain.utils as utils", "utils.content_hash"),
        ("from domain import utils", "utils.content_hash"),
    ],
)
def test_module_imports(statement: str, reference: str) -> None:
    result = REWRITE(f'{statement}\nvalue = {reference}("x")\n', MOVES)
    assert 'value = domain.shared.hashing.content_hash("x")' in result
    assert "domain.utils" not in result
    assert REWRITE(result, MOVES) == result


def test_partial_module_move_retains_other_members() -> None:
    source = 'import domain.utils as utils\nx = utils.content_hash("x")\ny = utils.parse_bool("yes")\n'
    moves = {"domain.utils.content_hash": MOVES["domain.utils.content_hash"]}
    result = REWRITE(source, moves)
    assert "import domain.utils as utils" in result
    assert 'x = domain.shared.hashing.content_hash("x")' in result
    assert 'y = utils.parse_bool("yes")' in result
    assert REWRITE(result, moves) == result


def test_unrelated_names_strings_comments_and_shadowed_binding() -> None:
    source = (
        "from domain.value_objects import ChatContext\n"
        "x = ChatContext()\n"
        "# domain.value_objects.ChatContext\n"
        'target = "domain.value_objects.ChatContext"\n'
        "def f(ChatContext):\n"
        "    return ChatContext()\n"
    )
    result = REWRITE(source, MOVES)
    assert "x = RagContext()" in result
    assert "def f(ChatContext):\n    return ChatContext()" in result
    assert "# domain.value_objects.ChatContext" in result
    assert 'target = "domain.value_objects.ChatContext"' in result


def test_explicit_alias_equal_to_old_name_is_scoped() -> None:
    source = (
        "from domain.value_objects import ChatContext\n"
        "x = ChatContext()\n"
        "def f():\n"
        "    from domain.value_objects import ChatContext as ChatContext\n"
        "    return ChatContext()\n"
    )
    result = REWRITE(source, MOVES)
    assert "x = RagContext()" in result
    assert "import RagContext as ChatContext" in result
    assert "return ChatContext()" in result
    assert REWRITE(result, MOVES) == result


def test_dry_run_write_and_idempotence(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    source = "from domain.utils import content_hash\n"
    path.write_text(source)
    mapping = tmp_path / "moves.json"
    mapping.write_text(json.dumps(MOVES))
    command = [sys.executable, str(SCRIPT), str(path), "--mapping", str(mapping)]
    dry_run = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    assert dry_run.returncode == 1
    assert b"domain.shared.hashing" in dry_run.stdout
    assert path.read_text() == source
    written = subprocess.run([*command, "--write"], capture_output=True, check=False)  # noqa: S603
    assert written.returncode == 0, written.stderr
    repeated = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    assert repeated.returncode == 0
    assert repeated.stdout == b""


def test_unsupported_star_import_prevents_all_writes(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    first.write_text("from domain.utils import content_hash\n")
    second = tmp_path / "b.py"
    second.write_text("from domain.utils import *\n")
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), str(tmp_path), "--write"], capture_output=True, check=False
    )
    assert result.returncode != 0
    assert b"Expand star import" in result.stderr
    assert first.read_text() == "from domain.utils import content_hash\n"


def test_table_splits_modules_and_resolves_barrels() -> None:
    moves = json.loads(SCRIPT.with_name("domain_moves.json").read_text())
    assert moves["domain.utils.content_hash"] == "domain.shared.hashing.content_hash"
    assert moves["domain.utils.parse_bool"] == "domain.runtime.config.parse_bool"
    assert moves["domain.value_objects.llm_provider.Breadth"] == "domain.rag.breadth.Breadth"
    assert moves["domain.value_objects.llm_provider.LLMProvider"] == "domain.runtime.enums.LLMProvider"
    assert moves["domain.exceptions.ValidationError"] == "domain.shared.exceptions.ValidationError"
    assert moves["domain.repositories.chunk_repository.ChunkRepository"] == (
        "application.uow.chunks.ChunkRepository"
    )


def test_table_resolves_rag_barrel_and_vector_read_port() -> None:
    moves = json.loads(SCRIPT.with_name("domain_moves.json").read_text())
    source = (
        "from domain.services.rag_policy import build_system_prompt, compute_context_budget\n"
        "from domain.repositories.vector_store_repository import VectorStoreRepository\n"
        "port: VectorStoreRepository\n"
    )
    result = REWRITE(source, moves)
    assert "from domain.rag.prompts import build_system_prompt" in result
    assert "from domain.rag.retrieval import compute_context_budget" in result
    assert "from domain.rag.ports import VectorSearch" in result
    assert "port: VectorSearch" in result
    assert REWRITE(result, moves) == result


def test_table_resolves_runtime_repository_model_imports() -> None:
    moves = json.loads(SCRIPT.with_name("domain_moves.json").read_text())
    source = (
        "from domain.repositories.background_job_repository import BackgroundJob\n"
        "from domain.repositories.config_parameter_repository import ConfigParameter\n"
    )
    result = REWRITE(source, moves)
    assert "from domain.runtime.jobs import BackgroundJob" in result
    assert "from domain.runtime.config import ConfigParameter" in result
    assert REWRITE(result, moves) == result


def test_table_domain_targets_exist() -> None:
    moves = json.loads(SCRIPT.with_name("domain_moves.json").read_text())
    targets = {target for target in moves.values() if target.startswith("domain.")}

    for target in sorted(targets):
        module, symbol = target.rsplit(".", 1)
        assert hasattr(import_module(module), symbol), target
