"""Shared retry decorator for transient I/O failures (Qdrant, HTTP, etc.)."""

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

retry_on_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(
        (
            TimeoutError,
            ConnectionError,
            ResponseHandlingException,
            httpx.TransportError,
        )
    ),
    reraise=True,
)
