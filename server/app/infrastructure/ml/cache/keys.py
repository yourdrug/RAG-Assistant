"""Build answer cache identities from generation context and current permissions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import TYPE_CHECKING

from config import settings
from domain.rag.answer_policy import policy_for
from domain.rag.breadth import Breadth
from domain.rag.evidence.focus import EVIDENCE_POLICY_VERSION
from domain.rag.prompts import build_system_prompt
from infrastructure.ml.cache.answers import compute_question_hash, compute_visibility_scope_hash


if TYPE_CHECKING:
    from infrastructure.ml.rag_pipeline import RagPipelineState


class CacheKeyBuilder:
    @staticmethod
    def build(state: RagPipelineState, domain_keys: tuple[str, ...] | None = None) -> tuple[str, str]:
        vis_hash = compute_visibility_scope_hash(
            state.ctx.user_kind,
            state.ctx.user_id,
            state.ctx.user_group_ids,
            user_role=state.ctx.user_role,
            curator_scope=state.ctx.curator_scope,
        )
        q_hash = compute_question_hash(
            state.query_for_search,
            context={
                "as_of_date": (state.ctx.as_of_date or date.today()).isoformat(),  # noqa: DTZ011
                "depth": state.ctx.depth,
                "summary": state.ctx.summary,
                "question": state.question,
                "history": [
                    {"role": message.type, "content": str(message.content)}
                    for message in state.history_messages
                ],
                "rag_settings": asdict(state.rag),
                "corpus_revision": state.cache_revision,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "openrouter_model": settings.openrouter_model,
                "ml_provider": settings.ml_provider,
                "llm_temperature": settings.llm_temperature,
                "llm_top_p": settings.llm_top_p,
                "embedding_revision": settings.embedding_revision,
                "tei_embed_model": settings.tei_embed_model,
                "deepinfra_embed_model": settings.deepinfra_embed_model,
                "deepinfra_rerank_model": settings.deepinfra_rerank_model,
                "retrieval_score_threshold": settings.rag_retrieval_score_threshold,
                "evidence_policy": EVIDENCE_POLICY_VERSION,
                "domain_prompts": {
                    domain_key: [
                        policy_for(domain_key).prompt_addendum(state.ctx.as_of_date),
                        policy_for(domain_key).prompt_addendum(state.ctx.as_of_date),
                    ]
                    for domain_key in domain_keys
                }
                if domain_keys is not None
                else {},
                "system_prompts": [
                    build_system_prompt(Breadth.NARROW, question=state.question),
                    build_system_prompt(Breadth.BROAD, enumerate_cases=True, question=state.question),
                ],
            },
        )
        return q_hash, vis_hash
