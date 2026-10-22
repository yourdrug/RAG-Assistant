"""Durable database checkpoints and deterministic benchmark identities."""

import hashlib
import json


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class DatabaseBenchmarkCheckpoints:
    def __init__(self, uow_factory, sweep_id: int) -> None:
        self._uow_factory = uow_factory
        self._sweep_id = sweep_id

    async def load(self, key: str) -> object | None:
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_sweeps.load_checkpoint(self._sweep_id, key)

    async def save(self, key: str, value: object) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.save_checkpoint(self._sweep_id, key, value)
