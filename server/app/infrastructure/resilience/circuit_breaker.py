"""Circuit breaker wrapper for LLM operations with Prometheus metrics.

Manual implementation for predictable async behavior.
States:
- CLOSED: normal operation, failures counted
- OPEN: all calls rejected immediately with CircuitBreakerError
- HALF-OPEN: parallel probes allowed; first success -> CLOSED, any failure -> OPEN

Lock scope: lock protects only state reads/writes, NOT func execution.
This allows concurrent calls through the breaker.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any, Callable

from prometheus_client import Counter, Gauge

from domain.exceptions import LLMUnavailableError

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
CB_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["operation"],
)
CB_FAILURES = Counter(
    "circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["operation"],
)


class CBState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(LLMUnavailableError):
    """Raised when the circuit breaker is OPEN (fail-fast).

    Inherits from LLMUnavailableError (domain exception) so the SSE layer
    can catch it and present a user-friendly message.
    """

    def __init__(self, message: str = "LLM временно недоступен, попробуйте позже") -> None:
        super().__init__(detail=message)


class LLMCircuitBreaker:
    """Thin async circuit breaker with Prometheus instrumentation.

    Lock scope: ``_lock`` protects only state transitions, not func execution.
    In HALF_OPEN state, all concurrent calls are allowed (parallel probes).
    First success closes the breaker; any failure reopens it.
    """

    def __init__(
        self,
        fail_max: int = 5,
        timeout_duration: int = 300,
        operation_name: str = "llm",
    ) -> None:
        self._fail_max = fail_max
        self._timeout_duration = timeout_duration
        self._operation_name = operation_name
        self._state = CBState.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Current breaker state as string: 'closed', 'open', or 'half_open'."""
        return self._state.value

    # -- State check (short lock) --------------------------------------------

    def check_open(self) -> None:
        """Check if breaker is OPEN. Raises CircuitBreakerError if OPEN.

        If timeout has expired, allows the probe (returns without error).
        Transition to HALF_OPEN will happen in report_success/failure under lock.
        """
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self._timeout_duration:
                return
            CB_FAILURES.labels(operation=self._operation_name).inc()
            self._update_metrics()
            raise CircuitBreakerError(
                f"Circuit breaker '{self._operation_name}' is OPEN. Retry after {self._timeout_duration}s."
            )

    def _transition_to_half_open_if_needed(self) -> None:
        """Transition from OPEN to HALF_OPEN if the timeout has expired (called under lock)."""
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self._timeout_duration:
                self._state = CBState.HALF_OPEN
                log.info(
                    "Circuit breaker '%s' transitioned to HALF_OPEN (probe allowed)",
                    self._operation_name,
                )

    # -- Report success/failure (short lock) ----------------------------------

    async def report_success(self) -> None:
        """Report a successful call. Under lock, check and transition if needed."""
        async with self._lock:
            self._transition_to_half_open_if_needed()
            if self._state == CBState.HALF_OPEN:
                self._state = CBState.CLOSED
                self._failure_count = 0
                log.info(
                    "Circuit breaker '%s' closed (half-open probe succeeded)",
                    self._operation_name,
                )
            elif self._state == CBState.CLOSED:
                self._failure_count = 0
            self._update_metrics()

    async def report_failure(self) -> None:
        """Report a failed call. Under lock, increment counter and potentially open."""
        async with self._lock:
            self._transition_to_half_open_if_needed()
            if self._state == CBState.HALF_OPEN:
                # Any failure during half-open probe reopens immediately
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                log.warning(
                    "Circuit breaker '%s' reopened (half-open probe failed)",
                    self._operation_name,
                )
            else:
                self._failure_count += 1
                if self._failure_count >= self._fail_max:
                    self._state = CBState.OPEN
                    self._opened_at = time.monotonic()
                    log.warning(
                        "Circuit breaker '%s' opened after %d consecutive failures",
                        self._operation_name,
                        self._failure_count,
                    )
            self._update_metrics()

    # -- Unified call API (for simple use cases) ------------------------------

    def _update_metrics(self) -> None:
        state_val = {"closed": 0, "open": 1, "half_open": 2}.get(self._state.value, -1)
        CB_STATE.labels(operation=self._operation_name).set(state_val)

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through circuit breaker (convenience wrapper).

        check_open() + report_success/report_failure around func.
        Lock is NOT held during func execution — allows concurrency.
        """
        self.check_open()

        try:
            result = await func(*args, **kwargs)
        except CircuitBreakerError:
            raise
        except Exception:
            await self.report_failure()
            raise
        else:
            await self.report_success()
            return result

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Use as decorator: ``@llm_generate_breaker``."""
        _func = func

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.call(_func, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper


# ---------------------------------------------------------------------------
# Breaker registry — no globals, just a dict
# ---------------------------------------------------------------------------
_breakers: dict[str, LLMCircuitBreaker] = {}


def get_breaker(name: str) -> LLMCircuitBreaker:
    """Get a named breaker. Raises KeyError if not initialised."""
    try:
        return _breakers[name]
    except KeyError:
        raise KeyError(
            f"Circuit breaker '{name}' not initialised. Call init_breakers() at startup."
        ) from None


def init_breakers(
    fail_max: int = 5,
    timeout_duration: int = 300,
) -> None:
    """Create the standard LLM breakers.  Call once during app startup."""
    _breakers.clear()
    for name in ("llm_generate", "llm_auxiliary"):
        _breakers[name] = LLMCircuitBreaker(
            fail_max=fail_max,
            timeout_duration=timeout_duration,
            operation_name=name,
        )
    log.info(
        "LLM circuit breakers initialised (fail_max=%d, timeout=%ds)",
        fail_max,
        timeout_duration,
    )
