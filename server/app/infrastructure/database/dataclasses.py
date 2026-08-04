"""DatabaseURL dataclass — builds async-capable connection string."""

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
