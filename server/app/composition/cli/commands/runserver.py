"""CLI-команда: запуск uvicorn-сервера."""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Literal, cast

import typer
import uvicorn

from config import settings


logger = logging.getLogger("cli")


def runserver(
    *,
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "0.0.0.0",  # noqa: S104
    port: Annotated[int, typer.Option("--port", help="Bind port")] = 8001,
    loop: str = "auto",
    reload: bool = False,
    proxy_headers: bool = True,
    forwarded_allow_ips: str | None = None,
) -> None:
    """Запустить uvicorn-сервер."""
    try:
        loop = cast("Literal['none', 'auto', 'asyncio', 'uvloop']", loop)

        uvicorn.run(
            app="main:create_application",
            host=host,
            port=port,
            loop=loop,
            reload=reload,
            proxy_headers=proxy_headers,
            forwarded_allow_ips=forwarded_allow_ips or settings.forwarded_allow_ips,
            factory=True,
        )
    except Exception as exception:
        logger.exception("Ошибка при запуске uvicorn server", exc_info=exception)
        sys.exit(-1)
