"""Deployment-owned exact origins prevent SSRF and provider key forwarding."""

from urllib.parse import SplitResult, urlsplit

from config import EndpointSettings, MLProviderSection
from domain.shared.exceptions import ValidationError


# Each endpoint receives its env-owned key, so its origin is limited to deployment-approved
# lists: the endpoint's external allowlist plus internal ML services.
EXTERNAL_ALLOWLISTS = {
    "llm_base_url": "llm_allowed_origins",
    "embedding_base_url": "ml_allowed_origins",
    "rerank_base_url": "ml_allowed_origins",
}
ENDPOINT_KEYS = frozenset(EXTERNAL_ALLOWLISTS)


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

    for key, allowlist in EXTERNAL_ALLOWLISTS.items():
        value = getattr(candidate, key)

        # Required endpoints are reported by the provider check with a clearer message.
        if not value:
            continue

        allowed = internal | {endpoint_origin(item) for item in getattr(candidate, allowlist)}

        if endpoint_origin(value) not in allowed:
            message = f"ML endpoint origin is not deployment-approved: {key}"
            raise ValidationError(message)


def validate_pools(candidate: MLProviderSection) -> None:
    errors = candidate.check_http_pools()

    if errors:
        raise ValidationError("; ".join(errors))
