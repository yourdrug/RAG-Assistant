"""Build answer cache identities from generation context and current permissions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import TYPE_CHECKING

from config import get_request_settings
from domain.rag.answer_policy import policy_for
from domain.rag.breadth import Breadth
from domain.rag.evidence.focus import EVIDENCE_POLICY_VERSION
from domain.rag.prompts import build_system_prompt
from infrastructure.ml.cache.answers import compute_question_hash, compute_visibility_scope_hash


if TYPE_CHECKING:
    from application.rag.dto import PreparedQuery


class CacheKeyBuilder:
    @staticmethod
    def build(state: PreparedQuery, domain_keys: tuple[str, ...] | None = None) -> tuple[str, str]:
        vis_hash = compute_visibility_scope_hash(
            state.request.ctx.user_kind,
            state.request.ctx.user_id,
            state.request.ctx.user_group_ids,
            user_role=state.request.ctx.user_role,
            curator_scope=state.request.ctx.curator_scope,
        )
        q_hash = compute_question_hash(
            state.query,
            context={
                "as_of_date": (state.request.ctx.as_of_date or date.today()).isoformat(),  # noqa: DTZ011
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
                "evidence_policy": EVIDENCE_POLICY_VERSION,
                "domain_prompts": {
                    domain_key: [
                        policy_for(domain_key).prompt_addendum(state.request.ctx.as_of_date),
                        policy_for(domain_key).prompt_addendum(state.request.ctx.as_of_date),
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
