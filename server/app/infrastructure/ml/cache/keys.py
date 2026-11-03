"""Build answer cache identities from generation context and current permissions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from config import get_request_settings
from domain.rag.breadth import Breadth
from infrastructure.domain_profile.answer_policy import policy_for
from infrastructure.ml.cache.answers import compute_question_hash, compute_visibility_scope_hash
from infrastructure.ml.evidence_profiles.profiles import active_evidence_profile
from infrastructure.ml.rag.prompts import build_system_prompt


if TYPE_CHECKING:
    from datetime import date

    from application.rag.dto import PreparedQuery
    from domain.ingestion.profiles.registry import DomainProfileRegistry


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
                "evidence_policy": (active_evidence_profile.get().key, active_evidence_profile.get().version),
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
    model = get_request_settings()
    return (
        ("llm_provider", model.llm_provider),
        ("llm_model", model.llm_model),
        ("openrouter_model", model.openrouter_model),
        ("ml_provider", model.ml_provider),
        ("llm_temperature", model.llm_temperature),
        ("llm_top_p", model.llm_top_p),
        ("embedding_revision", model.embedding_revision),
        ("tei_embed_model", model.tei_embed_model),
        ("deepinfra_embed_model", model.deepinfra_embed_model),
        ("deepinfra_rerank_model", model.deepinfra_rerank_model),
        ("rag_retrieval_score_threshold", model.rag_retrieval_score_threshold),
    )


def cache_today() -> date:
    return datetime.now(UTC).astimezone().date()
