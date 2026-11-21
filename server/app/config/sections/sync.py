"""Sync channels: the GitHub instance and the webhook secret of its App."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr

from config.ownership import Secret, Static


class SyncSection(BaseModel):
    # Host of the GitHub instance channels read (github.com, or a GitHub Enterprise Server host).
    github_instance: Static[str] = "github.com"
    # Signs webhook deliveries of the GitHub App; empty disables the webhook endpoint.
    github_webhook_secret: Secret = SecretStr("")
