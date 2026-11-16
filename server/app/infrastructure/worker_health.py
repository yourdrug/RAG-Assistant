"""Compose probe for a single worker replica's Arq heartbeat."""

import argparse
import asyncio
import socket

from arq.worker import async_check_health

from config import RedisSection, process_section
from infrastructure.redis.arq_settings import arq_redis_settings


def health_check_key(queue_name: str) -> str:
    return f"{queue_name}:health-check:{socket.gethostname()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue")
    args = parser.parse_args()
    result = asyncio.run(
        async_check_health(
            arq_redis_settings(process_section(RedisSection).redis_url),
            health_check_key(args.queue),
            args.queue,
        )
    )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
