"""Arq connection settings shared by lightweight probes and queue clients."""

from dataclasses import replace
from urllib.parse import unquote

from arq.connections import RedisSettings


def arq_redis_settings(dsn: str) -> RedisSettings:
    """Build arq settings from a DSN, decoding percent-encoded credentials like redis-py does."""
    parsed = RedisSettings.from_dsn(dsn)
    return replace(
        parsed,
        username=unquote(parsed.username) if parsed.username else parsed.username,
        password=unquote(parsed.password) if parsed.password else parsed.password,
    )
