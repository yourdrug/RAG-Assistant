"""PostgreSQL connection and pool settings."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Secret, Static


class DatabaseSection(BaseModel):
    db_host: Static[str]
    db_port: Static[str]
    db_user: Static[str]
    db_password: Secret[str]
    db_name: Static[str]
    db_slave_hosts: Static[str] = ""
    db_slave_ports: Static[str] = ""
    db_pool_size: Static[int] = 40
    db_max_overflow: Static[int] = 20
    # pool hardening (defaults generous enough for OCR/ingest queries)
    db_pool_recycle_sec: Static[int] = 1800
    db_statement_timeout_ms: Static[int] = 600000
    db_idle_in_transaction_timeout_ms: Static[int] = 120000

    @property
    def db_slave_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.db_slave_hosts.split(",") if h.strip()]

    @property
    def db_slave_ports_list(self) -> list[str]:
        return [p.strip() for p in self.db_slave_ports.split(",") if p.strip()]
