"""Behavior of the blank-line hook, including Python 3.12 compatibility."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "blank_lines.py"


@pytest.mark.parametrize(
    "source,expected,options",
    [
        (
            "x = 1\n# attached\nif x:\n    print(x)\nprint('end')\n",
            "x = 1\n\n# attached\nif x:\n    print(x)\n\nprint('end')\n",
            [],
        ),
        (
            (
                "if x:\n    x = 1\n    while x:\n        x -= 1\n    print(x)\n"
                "elif y:\n    pass\nelse:\n    pass\n"
            ),
            (
                "if x:\n    x = 1\n\n    while x:\n        x -= 1\n\n    print(x)\n"
                "elif y:\n    pass\nelse:\n    pass\n"
            ),
            [],
        ),
        (
            "x = 1\nfor x in xs:\n    pass\nelse:\n    pass\nprint(x)\n",
            "x = 1\n\nfor x in xs:\n    pass\nelse:\n    pass\n\nprint(x)\n",
            [],
        ),
        (
            "x = 1\ntry:\n    pass\nexcept ValueError:\n    pass\nx = 2\n",
            "x = 1\n\ntry:\n    pass\nexcept ValueError:\n    pass\n\nx = 2\n",
            ["--also", "try"],
        ),
        (
            "x = 1\r\nwith context():\r\n    pass\r\nx = 2\r\n",
            "x = 1\r\n\r\nwith context():\r\n    pass\r\n\r\nx = 2\r\n",
            ["--also", "with"],
        ),
    ],
)
def test_check_fix_and_idempotence(
    tmp_path: Path,
    source: str,
    expected: str,
    options: list[str],
) -> None:
    path = tmp_path / "sample.py"
    path.write_bytes(source.encode())
    command = [sys.executable, str(SCRIPT), *options, str(path)]
    checked = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        [*command, "--check"], capture_output=True, check=False
    )
    assert checked.returncode == 1
    assert path.read_bytes() == source.encode()

    fixed = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        command, capture_output=True, check=False
    )
    assert fixed.returncode == 1
    assert path.read_bytes() == expected.encode()
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(source))
    repeated = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        command,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode == 0


def test_invalid_syntax_fails_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    source = b"if :\n"
    path.write_bytes(source)
    result = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        [sys.executable, str(SCRIPT), "--check", str(path)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert b"syntax error" in result.stderr
    assert path.read_bytes() == source


def test_baseline_skips_only_listed_files_from_another_working_directory(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.py"
    fresh = tmp_path / "fresh.py"
    source = b"x = 1\nif x:\n    pass\nx = 2\n"
    legacy.write_bytes(source)
    fresh.write_bytes(source)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# legacy files\n\nlegacy.py\n", encoding="utf-8")
    command = [sys.executable, str(SCRIPT), "--exclude-from", str(baseline), str(legacy), str(fresh)]
    checked = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        [*command, "--check"], cwd=tmp_path.parent, capture_output=True, check=False
    )
    assert checked.returncode == 1
    assert str(fresh).encode() in checked.stdout
    assert str(legacy).encode() not in checked.stdout
    assert fresh.read_bytes() == source
    fixed = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        command, cwd=tmp_path.parent, capture_output=True, check=False
    )
    assert fixed.returncode == 1
    assert legacy.read_bytes() == source
    assert fresh.read_bytes() != source
    repeated = subprocess.run(  # noqa: S603 -- fixed Python and test-owned paths
        [*command, "--check"], cwd=tmp_path.parent, capture_output=True, check=False
    )
    assert repeated.returncode == 0
