"""DatabaseURL dataclass -- builds an async-capable PostgreSQL connection string.

Accepts individual components (user, password, host, port, database) and
exposes ``.connection_string`` for synchronous drivers and
``.async_connection_string`` for ``asyncpg``/`` SQLAlchemy async`` drivers.
"""

from dataclasses import dataclass


@dataclass
class DatabaseURL:
    user: str
    password: str
    host: str
    port: str
    db_name: str

    def __str__(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"

    @property
    def url(self) -> str:
        return str(self)
