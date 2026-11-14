"""Deployment-owned exact origins prevent SSRF and provider key forwarding."""

from urllib.parse import SplitResult, urlsplit

from config import EndpointSettings, MLProviderSection
from domain.shared.exceptions import ValidationError


ENDPOINT_KEYS = frozenset(
    {
        "tei_embed_url",
        "tei_rerank_url",
        "openrouter_base_url",
        "openai_compatible_base_url",
        "deepinfra_base_url",
    }
)
CLOUD_ORIGINS = {
    "openrouter_base_url": "https://openrouter.ai:443",
    "deepinfra_base_url": "https://api.deepinfra.com:443",
}


FIRST_PRINTABLE_CODEPOINT = 32


def validate_endpoint_components(parsed: SplitResult, value: str) -> None:
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(character.isspace() or ord(character) < FIRST_PRINTABLE_CODEPOINT for character in value)
    ):
        raise ValueError


def endpoint_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)

        validate_endpoint_components(parsed, value)

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"{parsed.scheme}://{(parsed.hostname or '').lower()}:{port}"
    except ValueError:
        message = "ML endpoint requires an HTTP(S) URL without credentials, query or fragment"
        raise ValidationError(message) from None


def validate_endpoints(candidate: EndpointSettings) -> None:
    internal = {endpoint_origin(value) for value in candidate.ml_internal_allowed_origins}

    for key in ENDPOINT_KEYS:
        value = getattr(candidate, key)

        # Empty inactive internal services remain supported for cloud-only deployments.
        if not value and key not in CLOUD_ORIGINS:
            continue

        origin = endpoint_origin(value)
        allowed = {CLOUD_ORIGINS[key]} if key in CLOUD_ORIGINS else internal

        if origin not in allowed:
            message = f"ML endpoint origin is not deployment-approved: {key}"
            raise ValidationError(message)


def validate_pools(candidate: MLProviderSection) -> None:
    errors = candidate.check_http_pools()

    if errors:
        raise ValidationError("; ".join(errors))
