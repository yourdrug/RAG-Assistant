"""PostgreSQL URL components with driver-safe escaping."""

from dataclasses import dataclass

from sqlalchemy.engine import URL


@dataclass
class DatabaseURL:
    user: str
    password: str
    host: str
    port: str
    db_name: str

    def __str__(self) -> str:
        return self.url.render_as_string(hide_password=False)

    @property
    def url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=int(self.port),
            database=self.db_name,
        )
