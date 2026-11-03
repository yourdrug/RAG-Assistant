# Async notification API used by PostgreSQL integration tests.

from collections.abc import Callable

class Connection:
    async def add_listener(
        self, channel: str, callback: Callable[[Connection, int, str, str], object]
    ) -> None: ...
    async def close(self) -> None: ...

async def connect(dsn: str) -> Connection: ...
