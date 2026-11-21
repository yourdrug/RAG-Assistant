"""Which storage files a channel excludes: its own patterns and ``.ragignore`` files.

Both use ``.gitignore`` syntax with git semantics: a file inside an excluded directory
stays excluded even if a later ``!`` pattern names it, and a nested ``.ragignore``
applies to its own directory, overriding shallower files. The channel's patterns are a
separate layer that ``.ragignore`` cannot undo: ``.ragignore`` only narrows the scope an
administrator approved. Paths are POSIX, relative to the channel folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec

from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


RAGIGNORE = ".ragignore"
MAX_RAGIGNORE_BYTES = 64 * 1024


class InvalidIgnoreRulesError(ValidationError):
    """Patterns that cannot be parsed; ``source`` is the channel or the ``.ragignore`` path."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"Invalid ignore rules in {source}: {detail}")
        self.source = source


def normalize_folder(folder: str) -> str:
    """A relative POSIX folder without leading or trailing slashes; ``""`` is the root."""
    parts = [part for part in folder.replace("\\", "/").split("/") if part not in {"", "."}]

    if ".." in parts:
        msg = "Folder must not leave the storage root"
        raise ValidationError(msg, field="folder")

    return "/".join(parts)


def relative_to(path: str, folder: str) -> str | None:
    """``path`` relative to ``folder``, or ``None`` when it lies outside it."""
    if not folder:
        return path

    if path == folder:
        return ""

    return path[len(folder) + 1 :] if path.startswith(folder + "/") else None


def compile_patterns(lines: Iterable[str], source: str) -> GitIgnoreSpec:
    try:
        return GitIgnoreSpec.from_lines(list(lines))
    except ValueError as exc:  # pathspec reports malformed patterns as ValueError subclasses
        raise InvalidIgnoreRulesError(source, str(exc)) from exc


def ancestor_targets(path: str) -> list[str]:
    """Directories containing ``path`` (with a trailing slash), shallowest first, then the path."""
    parts = path.split("/")
    return [*("/".join(parts[: index + 1]) + "/" for index in range(len(parts) - 1)), path]


@dataclass(frozen=True)
class IgnoreLayer:
    """One rule set: patterns from ``.ragignore`` files keyed by their folder, deepest last."""

    specs: tuple[tuple[str, GitIgnoreSpec], ...]

    def decision(self, target: str) -> bool | None:
        """Last matching rule across applicable files: excluded, re-included, or no opinion."""
        result: bool | None = None

        for folder, spec in self.specs:
            relative = relative_to(target.rstrip("/"), folder)

            if relative is None or relative == "":
                continue

            check = spec.check_file(relative + ("/" if target.endswith("/") else ""))

            if check.include is not None:
                result = check.include

        return result

    def excludes(self, path: str) -> bool:
        # Git never looks inside an excluded directory, so a ``!`` cannot rescue its files.
        return any(self.decision(target) is True for target in ancestor_targets(path))


class IgnoreRules:
    """Exclusions of one channel: its own patterns plus the ``.ragignore`` files of a snapshot."""

    def __init__(
        self, channel_patterns: Iterable[str] = (), ragignore: Mapping[str, str] | None = None
    ) -> None:
        self._channel = IgnoreLayer((("", compile_patterns(channel_patterns, "channel settings")),))
        files = sorted(
            (ragignore or {}).items(), key=lambda item: (item[0].count("/") if item[0] else -1, item[0])
        )
        self._ragignore = IgnoreLayer(
            tuple(
                (
                    normalize_folder(folder),
                    compile_patterns(text.splitlines(), f"{folder}/{RAGIGNORE}" if folder else RAGIGNORE),
                )
                for folder, text in files
            )
        )

    def excludes_by_channel(self, path: str) -> bool:
        return self._channel.excludes(path)

    def is_excluded(self, path: str) -> bool:
        """``path`` is a file relative to the channel folder; ``.ragignore`` files are never documents."""
        if path.rsplit("/", 1)[-1] == RAGIGNORE:
            return True

        return self._channel.excludes(path) or self._ragignore.excludes(path)

    def excludes_directory(self, folder: str) -> bool:
        """Every file under ``folder`` is excluded by the channel's own patterns."""
        folder = normalize_folder(folder)
        return bool(folder) and self._channel.excludes(folder + "/")
