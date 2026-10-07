"""Durable database checkpoints with local export and legacy recovery adapters."""

import hashlib
import json
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_checkpoint(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_checkpoint(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class FileBenchmarkCheckpoints:
    """Local CLI storage; web sweeps use DatabaseBenchmarkCheckpoints."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def load(self, key: str) -> object | None:
        import asyncio

        return await asyncio.to_thread(read_checkpoint, self._root / key)

    async def save(self, key: str, value: object) -> None:
        import asyncio

        await asyncio.to_thread(write_checkpoint, self._root / key, value)


class DatabaseBenchmarkCheckpoints:
    def __init__(self, uow_factory, sweep_id: int, legacy_root: Path | None = None) -> None:
        self._uow_factory = uow_factory
        self._sweep_id = sweep_id
        self._legacy_root = legacy_root
        self._legacy = FileBenchmarkCheckpoints(legacy_root) if legacy_root else None

    async def load(self, key: str) -> object | None:
        async with self._uow_factory.create(master=True) as uow:
            value = await uow.benchmark_sweeps.load_checkpoint(self._sweep_id, key)
        if value is None and self._legacy is not None:
            # One-way import lets existing failed sweeps retain their paid work.
            value = await self._legacy.load(key)
            if value is None and key.startswith("phase-a-"):
                value = await self.load_legacy_phase_a()
            if value is not None:
                await self.save(key, value)
        return value

    async def load_legacy_phase_a(self) -> object | None:
        import asyncio

        def read_latest():
            # API sweeps have immutable search_space/dataset. Older namespaces
            # changed with operational settings; reuse the latest actual search.
            paths = list(self._legacy_root.glob("phase-a-*.json"))
            return read_checkpoint(max(paths, key=lambda path: path.stat().st_mtime)) if paths else None

        return await asyncio.to_thread(read_latest)

    async def save(self, key: str, value: object) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.save_checkpoint(self._sweep_id, key, value)


async def import_legacy_configuration(
    store, root: Path, prefix: str, *, single_config_root: Path | None = None
) -> None:
    """Move an unambiguous legacy config namespace into durable stage storage."""
    import asyncio

    def read_legacy() -> list[tuple[str, object]]:
        source = root
        if single_config_root is not None and not (root / "checkpoints").exists():
            source = latest_single_config(single_config_root) or root
        namespaces = list((source / "checkpoints").glob("*/"))
        if len(namespaces) != 1:
            return []
        return [(path.name, read_checkpoint(path)) for path in namespaces[0].glob("*.json")]

    for name, value in await asyncio.to_thread(read_legacy):
        key = f"checkpoints/{prefix}/{name}"
        if await store.load(key) is None:
            await store.save(key, value)


def latest_single_config(root: Path) -> Path | None:
    phases = list(root.glob("phase-a-*.json"))
    if not phases:
        return None
    phase_time = max(path.stat().st_mtime for path in phases)
    reports = [path for path in root.glob("*/benchmark_*.json") if path.stat().st_mtime >= phase_time]
    # This fallback is used only for an immutable sweep with one finalist.
    # Configuration folders from an earlier search must not be mixed in.
    return max(reports, key=lambda path: path.stat().st_mtime).parent if reports else None
