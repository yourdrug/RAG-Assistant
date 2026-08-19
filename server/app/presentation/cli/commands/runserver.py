"""CLI-команда: запуск uvicorn-сервера."""

from __future__ import annotations

import logging
import sys
from typing import Literal, cast

import typer
import uvicorn

logger = logging.getLogger("cli")


def runserver(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8001, "--port", help="Bind port"),
    loop: str = "auto",
    reload: bool = False,
    proxy_headers: bool = True,
    forwarded_allow_ips: str | None = None,
) -> None:
    """Запустить uvicorn-сервер."""
    try:
        loop = cast(Literal["none", "auto", "asyncio", "uvloop"], loop)

        uvicorn.run(
            app="main:create_application",
            host=host,
            port=port,
            loop=loop,
            reload=reload,
            proxy_headers=proxy_headers,
            forwarded_allow_ips=forwarded_allow_ips,
            factory=True,
        )
    except Exception as exception:
        logger.error("Ошибка при запуске uvicorn server", exc_info=exception)
        sys.exit(-1)
