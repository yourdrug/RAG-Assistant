"""CORS and trusted proxy settings."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Static


class HttpSection(BaseModel):
    allowed_origins: Static[str]
    # Networks/IPs from which uvicorn may rewrite request.client from
    # X-Forwarded-For (the reverse proxy). Never trust client-supplied XFF.
    forwarded_allow_ips: Static[str] = "127.0.0.1"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
