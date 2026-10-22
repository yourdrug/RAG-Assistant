"""Deployment-owned exact origins prevent SSRF and provider key forwarding."""

from urllib.parse import urlsplit

from config import Settings
from domain.shared.exceptions import ValidationError

ENDPOINT_KEYS = frozenset(
    {
        "ollama_base_url",
        "tei_embed_url",
        "tei_rerank_url",
        "openrouter_base_url",
        "deepinfra_base_url",
    }
)
CLOUD_ORIGINS = {
    "openrouter_base_url": "https://openrouter.ai:443",
    "deepinfra_base_url": "https://api.deepinfra.com:443",
}


def endpoint_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "\\" in value
            or any(character.isspace() or ord(character) < 32 for character in value)
        ):
            raise ValueError

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"{parsed.scheme}://{parsed.hostname.lower()}:{port}"
    except ValueError:
        raise ValidationError(
            "ML endpoint requires an HTTP(S) URL without credentials, query or fragment"
        ) from None


def validate_endpoints(candidate: Settings) -> None:
    internal = {endpoint_origin(value) for value in candidate.ml_internal_allowed_origins}

    for key in ENDPOINT_KEYS:
        value = getattr(candidate, key)

        # Empty inactive internal services remain supported for cloud-only deployments.
        if not value and key not in CLOUD_ORIGINS:
            continue

        origin = endpoint_origin(value)
        allowed = {CLOUD_ORIGINS[key]} if key in CLOUD_ORIGINS else internal

        if origin not in allowed:
            raise ValidationError(f"ML endpoint origin is not deployment-approved: {key}")


def validate_pools(candidate: Settings) -> None:
    errors = candidate.check_http_pools()

    if errors:
        raise ValidationError("; ".join(errors))
