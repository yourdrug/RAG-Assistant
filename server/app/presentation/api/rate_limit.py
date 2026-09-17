"""Rate limiting dependency for API routes — presentation-layer glue.

``rate_limit(policy)`` returns a FastAPI dependency that resolves the request
principal and enforces the named policy through the rate limiter port. Routes
reference policies exactly once, in a single line::

    @router.post("/chat", dependencies=[Depends(rate_limit(RateLimitPolicyName.CHAT))])

User-scoped policies key on the authenticated user (the dependency declares
``get_current_user`` itself, so it may be attached at route level); the login
policy is IP-scoped and must stay reachable without authentication.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from application.ports.rate_limit import RateLimiterPort, RateLimitPolicyName
from domain.exceptions import RateLimitExceededError
from fastapi import Depends, Request

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_rate_limiter
from presentation.api.schemas import CurrentUser

_IP_SCOPED = frozenset({RateLimitPolicyName.LOGIN})


def rate_limit(
    policy: RateLimitPolicyName,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Build a route dependency enforcing ``policy`` for the request principal."""
    if policy in _IP_SCOPED:

        async def _dep_ip(
            request: Request,
            limiter: RateLimiterPort | None = Depends(create_rate_limiter),
        ) -> None:
            if limiter is None:
                return
            await _enforce(limiter, policy, f"ip:{_client_ip(request)}")

        # Expose the policy for the route-coverage test (rendering/markup only).
        _dep_ip.rate_limit_policy = policy  # type: ignore[attr-defined]
        return _dep_ip

    async def _dep_user(
        current_user: CurrentUser = Depends(get_current_user),
        limiter: RateLimiterPort | None = Depends(create_rate_limiter),
    ) -> None:
        if limiter is None:
            return
        await _enforce(limiter, policy, f"user:{current_user.kind}:{current_user.id}")

    _dep_user.rate_limit_policy = policy  # type: ignore[attr-defined]
    return _dep_user


async def _enforce(limiter: RateLimiterPort, policy: RateLimitPolicyName, principal: str) -> None:
    decision = await limiter.check(policy, principal)
    if not decision.allowed:
        raise RateLimitExceededError(
            retry_after_sec=decision.retry_after_sec,
            limit=decision.limit,
        )


def _client_ip(request: Request) -> str:
    client = request.client
    return client.host if client is not None else "unknown"
