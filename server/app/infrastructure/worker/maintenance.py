"""Short global maintenance tasks, isolated from document and batch computation."""

from typing import Any


async def cron_outbox_dispatch(ctx: dict[str, Any]) -> None:
    await ctx["outbox_dispatcher"].run_once(batch_size=50)


async def cron_outbox_reconcile(ctx: dict[str, Any]) -> None:
    await ctx["outbox_dispatcher"].reconcile_stuck_documents()
