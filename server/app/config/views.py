"""Read-only structural views spanning several settings sections.

Some consumers need a fixed handful of fields from more than one section (for
example, every ML endpoint URL or a whole ML-client snapshot). These protocols
name exactly those fields, so such modules depend on a narrow view instead of
the full ``Settings``. ``Settings`` satisfies every view structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import SecretStr


class EndpointSettings(Protocol):
    """Deployment endpoints checked by ``endpoint_policy.validate_endpoints``."""

    @property
    def tei_embed_url(self) -> str: ...

    @property
    def tei_rerank_url(self) -> str: ...

    @property
    def openrouter_base_url(self) -> str: ...

    @property
    def openai_compatible_base_url(self) -> str: ...

    @property
    def deepinfra_base_url(self) -> str: ...

    @property
    def ml_internal_allowed_origins(self) -> Sequence[str]: ...


class EmbeddingIdentitySettings(Protocol):
    """Fields that determine the configured embedding model binding."""

    @property
    def ml_provider(self) -> str: ...

    @property
    def deepinfra_embed_model(self) -> str: ...

    @property
    def tei_embed_model(self) -> str: ...

    @property
    def embedding_revision(self) -> str: ...

    def model_copy(self) -> Self: ...


class MLClientSettings(EmbeddingIdentitySettings, Protocol):
    """One settings snapshot from which ML clients are built."""

    @property
    def llm_provider(self) -> str: ...

    @property
    def llm_temperature(self) -> float: ...

    @property
    def llm_num_predict_narrow(self) -> int: ...

    @property
    def llm_num_predict_broad(self) -> int: ...

    @property
    def openrouter_api_key(self) -> SecretStr: ...

    @property
    def openrouter_base_url(self) -> str: ...

    @property
    def openrouter_model(self) -> str: ...

    @property
    def openai_compatible_api_key(self) -> SecretStr: ...

    @property
    def openai_compatible_base_url(self) -> str: ...

    @property
    def openai_compatible_model(self) -> str: ...

    @property
    def openai_compatible_auxiliary_model(self) -> str: ...

    @property
    def active_llm_base_url(self) -> str: ...

    @property
    def active_llm_api_key(self) -> SecretStr: ...

    @property
    def active_llm_model(self) -> str: ...

    @property
    def active_auxiliary_model(self) -> str: ...

    @property
    def deepinfra_api_key(self) -> SecretStr: ...

    @property
    def deepinfra_base_url(self) -> str: ...

    @property
    def deepinfra_rerank_model(self) -> str: ...

    @property
    def tei_embed_url(self) -> str: ...

    @property
    def tei_rerank_url(self) -> str: ...

    @property
    def embedding_http_max_connections(self) -> int: ...

    @property
    def embedding_http_max_keepalive_connections(self) -> int: ...

    @property
    def rerank_http_max_connections(self) -> int: ...

    @property
    def rerank_http_max_keepalive_connections(self) -> int: ...


__all__ = ["EmbeddingIdentitySettings", "EndpointSettings", "MLClientSettings"]
