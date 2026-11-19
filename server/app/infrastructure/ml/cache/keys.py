"""Build answer cache identities from generation context and current permissions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from typing import TYPE_CHECKING, Any

from config import (
    HybridSearchSection,
    LLMSection,
    MLProviderSection,
    RankingSection,
    RetrievalSection,
    request_section,
)
from domain.rag.breadth import Breadth
from infrastructure.domain_profile.answer_policy import policy_for
from infrastructure.ml.cache.answers import compute_question_hash, compute_visibility_scope_hash
from infrastructure.ml.evidence_profiles.profiles import current_evidence_profile
from infrastructure.ml.rag.prompts import build_system_prompt


if TYPE_CHECKING:
    from datetime import date

    from application.rag.dto import PreparedQuery
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.rag.settings import RagSettings


class CacheKeyBuilder:
    @staticmethod
    def build(
        state: PreparedQuery,
        domain_keys: tuple[str, ...] | None = None,
        domain_registry: DomainProfileRegistry | None = None,
    ) -> tuple[str, str]:
        vis_hash = compute_visibility_scope_hash(
            state.request.ctx.user_kind,
            state.request.ctx.user_id,
            list(state.request.ctx.user_group_ids),
            user_role=state.request.ctx.user_role,
            curator_scope=state.request.ctx.curator_scope,
        )
        q_hash = compute_question_hash(
            state.query,
            context={
                "as_of_date": (state.request.ctx.as_of_date or cache_today()).isoformat(),
                # The space selects the searched documents; an answer never leaves its space.
                "space_id": state.request.ctx.space.id if state.request.ctx.space is not None else None,
                "depth": state.request.ctx.depth,
                "summary": state.request.ctx.summary,
                "question": state.request.question,
                "history": [
                    {"role": message.role, "content": str(message.content)}
                    for message in state.request.history
                ],
                "rag_settings": asdict(state.request.settings),
                "corpus_revision": state.cache_revision,
                "generation": dict(state.request.generation_config) or dict(generation_identity()),
                "evidence_policy": (current_evidence_profile().key, current_evidence_profile().version),
                "domain_prompts": {
                    domain_key: [
                        policy_for(domain_key, domain_registry).prompt_addendum(state.request.ctx.as_of_date),
                        policy_for(domain_key, domain_registry).prompt_addendum(state.request.ctx.as_of_date),
                    ]
                    for domain_key in domain_keys
                }
                if domain_keys is not None
                else {},
                "system_prompts": [
                    build_system_prompt(Breadth.NARROW, question=state.request.question),
                    build_system_prompt(Breadth.BROAD, enumerate_cases=True, question=state.request.question),
                ],
            },
        )
        return q_hash, vis_hash


def generation_identity() -> tuple[tuple[str, str | float], ...]:
    llm = request_section(LLMSection)
    ml = request_section(MLProviderSection)
    retrieval = request_section(RetrievalSection)
    return (
        ("llm_model", llm.llm_model),
        # Relevance/sufficiency gates run on the auxiliary model and can change the answer.
        ("auxiliary_model", llm.active_auxiliary_model),
        ("llm_base_url", llm.llm_base_url),
        ("embedding_model", ml.embedding_model),
        ("llm_temperature", llm.llm_temperature),
        ("llm_top_p", llm.llm_top_p),
        # Reasoning mode changes the generated text, so answers cached under the other mode are stale.
        ("llm_disable_reasoning", llm.llm_disable_reasoning),
        ("embedding_revision", ml.embedding_revision),
        ("rerank_model", ml.rerank_model),
        ("rerank_api_format", ml.rerank_api_format),
        ("rag_retrieval_score_threshold", retrieval.rag_retrieval_score_threshold),
    )


def cache_today() -> date:
    return datetime.now(UTC).astimezone().date()


def config_snapshot(
    settings: RagSettings, space: dict[str, Any] | None, domain_registry: DomainProfileRegistry | None = None
) -> tuple[str, dict[str, Any]]:
    """Hash an explicit non-secret projection; never dump the runtime Settings object."""
    profile = current_evidence_profile()
    data: dict[str, Any] = {
        "rag": asdict(settings),
        "space": space,
        "evidence_profile": f"{profile.key}@{profile.version}",
        "evidence_focus_chars": profile.max_focus_chars,
        "domain_prompt_versions": {
            domain.key: hashlib.sha256(
                (policy_for(domain.key, domain_registry).prompt_addendum(None) or "").encode()
            ).hexdigest()
            for domain in domain_registry.all()
        }
        if domain_registry is not None
        else {},
        "prompt_version": hashlib.sha256(
            "\n".join(build_system_prompt(breadth) for breadth in Breadth).encode()
        ).hexdigest(),
    }

    for section in (LLMSection, MLProviderSection, RetrievalSection, RankingSection, HybridSearchSection):
        current = request_section(section)
        data[section.__name__] = {
            name: getattr(current, name)
            for name in section.model_fields
            if not any(part in name.lower() for part in ("key", "url", "origin", "secret", "password"))
        }

    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest(), json.loads(normalized)
