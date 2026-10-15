"""Verify the provider's model context bound; tariffs themselves remain DB data."""

import httpx

from config import settings
from domain.value_objects.cost_budget import BudgetError, ModelBillingContract
from infrastructure.ml.config.endpoint_policy import validate_endpoints


class OpenRouterBillingCatalog:
    async def contract(self, provider: str, model: str) -> ModelBillingContract:
        if provider == 'deepinfra':
            contracts = {
                'BAAI/bge-m3': ModelBillingContract(
                    'tokens',
                    'embedding',
                    8192,
                    frozenset({'model', 'input', 'encoding_format'}),
                    'https://deepinfra.com/BAAI/bge-m3/api',
                ),
                'Qwen/Qwen3-Reranker-4B': ModelBillingContract(
                    'tokens',
                    'reranking',
                    32768,
                    frozenset({'queries', 'documents'}),
                    'https://deepinfra.com/Qwen/Qwen3-Reranker-4B/api',
                ),
            }
            if model not in contracts:
                raise BudgetError('Unverified DeepInfra billing contract')
            return contracts[model]
        context = await self.openrouter_context(provider, model)
        return ModelBillingContract(
            'tokens',
            'llm',
            context,
            frozenset(
                {
                    'model',
                    'messages',
                    'max_tokens',
                    'max_completion_tokens',
                    'temperature',
                    'top_p',
                    'seed',
                    'stop',
                    'stream',
                    'stream_options',
                    'response_format',
                    'frequency_penalty',
                    'presence_penalty',
                    'n',
                    'reasoning',
                    'user',
                    'provider',
                }
            ),
            'https://openrouter.ai/api/v1/models',
        )

    async def max_context(self, provider: str, model: str) -> int:
        return (await self.contract(provider, model)).input_limit

    async def openrouter_context(self, provider: str, model: str) -> int:
        if provider != 'openrouter' or model == 'openrouter/auto' or model.endswith(':online'):
            raise BudgetError('Unsupported billing contract')
        validate_endpoints(settings)
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            response = await client.get(
                f'{settings.openrouter_base_url.rstrip("/")}/models',
                headers={'Authorization': f'Bearer {settings.openrouter_api_key}'}
                if settings.openrouter_api_key
                else {},
            )
            response.raise_for_status()
            data = response.json()
        entry = next((item for item in data.get('data', []) if item.get('id') == model), None)
        context = entry.get('context_length') if entry else None
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            raise BudgetError('Provider model/context bound is unknown; paid call blocked')
        return context
