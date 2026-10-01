"""Distributed per-principal document worker admission without holding waiting slots."""

from __future__ import annotations

import logging
import uuid
from functools import wraps

from arq import Retry

log = logging.getLogger("default")
_LEASE_SECONDS = 31 * 60  # Exceeds the document task's 30 minute hard timeout.
_RELEASE = "if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('DEL',KEYS[1]) end return 0"


def one_document_per_principal(function):
    @wraps(function)
    async def admitted(ctx, **kwargs):
        redis = ctx["redis"]
        owner_id = kwargs.get("principal_id")
        if owner_id is None:
            owner_id = kwargs.get("owner_id")  # Previously queued jobs lack principal_id.
        principal = str(owner_id) if owner_id is not None else "internal"
        key = f"rag:worker:principal:{principal}"
        waits_key = f"rag:worker:waits:{kwargs['job_id']}"
        token = uuid.uuid4().hex
        acquired = await redis.set(key, token, nx=True, ex=_LEASE_SECONDS)
        if not acquired:
            container = ctx.get("container")
            if container is not None and await redis.set(
                f"rag:worker:wait-heartbeat:{kwargs['job_id']}", "1", nx=True, ex=60
            ):
                async with container.infrastructure.db.uow_factory.create(master=True) as uow:
                    await uow.background_jobs.touch_heartbeat(kwargs["job_id"])
            await redis.incr(waits_key)
            await redis.expire(waits_key, 24 * 3600)
            raise Retry(defer=5)
        try:
            # Queue admission attempts do not consume processing failure retries.
            waits = int(await redis.get(waits_key) or 0)
            run_ctx = dict(ctx, job_try=max(1, ctx.get("job_try", 1) - waits))
            return await function(run_ctx, **kwargs)
        finally:
            try:
                await redis.eval(_RELEASE, 1, key, token)
            except Exception:
                log.exception("Failed to release document worker lease for principal %s", principal)

    return admitted
