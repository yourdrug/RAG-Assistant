"""
CLI-команда: запуск uvicorn-сервера.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal, cast

import uvicorn

logger = logging.getLogger("cli")


def runserver(
    host: str = "0.0.0.0",
    port: int = 8001,
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
        )
    except Exception as exception:
        logger.error("Ошибка при запуске uvicorn server", exc_info=exception)
        sys.exit(-1)
