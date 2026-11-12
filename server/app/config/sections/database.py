"""PostgreSQL connection and pool settings."""

from __future__ import annotations

from pydantic import BaseModel, NonNegativeInt, PositiveInt

from config.ownership import Secret, Static


MAX_TCP_PORT = 65535


class DatabaseSection(BaseModel):
    db_host: Static[str]
    db_port: Static[str]
    db_user: Static[str]
    db_password: Secret
    db_name: Static[str]
    db_slave_hosts: Static[str] = ""
    db_slave_ports: Static[str] = ""
    db_pool_size: Static[PositiveInt] = 40
    db_max_overflow: Static[NonNegativeInt] = 20
    # pool hardening (defaults generous enough for OCR/ingest queries)
    db_pool_recycle_sec: Static[PositiveInt] = 1800
    # Positive only: Postgres treats 0 as "no timeout".
    db_statement_timeout_ms: Static[PositiveInt] = 600000
    db_idle_in_transaction_timeout_ms: Static[PositiveInt] = 120000

    @property
    def db_slave_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.db_slave_hosts.split(",") if h.strip()]

    @property
    def db_slave_ports_list(self) -> list[str]:
        return [p.strip() for p in self.db_slave_ports.split(",") if p.strip()]

    def check_database(self) -> list[str]:
        # Ports stay strings for the URL builders; check them here, not on the first connect.
        errors = [
            f"{name} must be a TCP port number (1-{MAX_TCP_PORT}), got {value!r}"
            for name, value in (
                ("DB_PORT", self.db_port),
                *(("DB_SLAVE_PORTS", port) for port in self.db_slave_ports_list),
            )
            if not is_tcp_port(value)
        ]

        if len(self.db_slave_hosts_list) != len(self.db_slave_ports_list):
            errors.append(
                "DB_SLAVE_HOSTS and DB_SLAVE_PORTS must list the same number of entries "
                f"({len(self.db_slave_hosts_list)} hosts, {len(self.db_slave_ports_list)} ports)"
            )

        return errors


def is_tcp_port(value: str) -> bool:
    return value.isascii() and value.isdigit() and 1 <= int(value) <= MAX_TCP_PORT
