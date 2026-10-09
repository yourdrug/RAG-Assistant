"""Request-local model usage; unknown cost remains unknown."""

from contextvars import ContextVar

from infrastructure.metrics.metrics import extract_usage_from_langchain

active_llm_usage: ContextVar[list[dict] | None] = ContextVar('rag_model_usage', default=None)
active_stream_response: ContextVar[dict | None] = ContextVar('rag_stream_response', default=None)


class UsageCapturingStream:
    """Preserve raw OpenAI-compatible usage discarded by LangChain conversion."""

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    async def __aenter__(self):
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.stream.__aexit__(*args)

    async def __aiter__(self):
        async for chunk in self.stream:
            captured = active_stream_response.get()
            if captured is not None:
                for name in ('id', 'model', 'usage'):
                    value = getattr(chunk, name, None)
                    if value is not None:
                        captured[name] = value
            yield chunk


class UsageCapturingCompletions:
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def create(self, **kwargs):
        response = await self.client.create(**kwargs)
        if kwargs.get('stream') and active_stream_response.get() is not None:
            return UsageCapturingStream(response)
        return response


def record_model_response(response, *, model: str = '', operation: str = 'llm') -> None:
    records = active_llm_usage.get()
    if records is None:
        return
    metadata = getattr(response, 'response_metadata', {}) or {}
    raw_usage = getattr(response, 'usage', None)
    input_tokens, output_tokens = extract_usage_from_langchain(response)
    usage = (
        raw_usage.model_dump() if raw_usage is not None and hasattr(raw_usage, 'model_dump') else raw_usage
    )
    usage = usage or metadata.get('token_usage') or metadata.get('usage') or {}
    if not isinstance(usage, dict):
        usage = {key: getattr(usage, key, None) for key in ('prompt_tokens', 'completion_tokens', 'cost')}
    input_tokens = input_tokens if input_tokens is not None else usage.get('prompt_tokens')
    output_tokens = output_tokens if output_tokens is not None else usage.get('completion_tokens')
    actual_model = (
        getattr(response, 'model', None) or metadata.get('model_name') or metadata.get('model') or model
    )
    response_id = metadata.get('id') or getattr(response, 'id', None)
    records.append(
        {
            'model': actual_model,
            'operation': operation,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': usage.get('cost'),
            'response_id': response_id
            if isinstance(response_id, str) and response_id.startswith('gen-')
            else None,
        }
    )


def summarize_usage(records: list[dict]) -> dict:
    complete = all(record.get('cost_usd') is not None for record in records)
    known = sum(float(record['cost_usd']) for record in records if record.get('cost_usd') is not None)
    return {
        'calls': len(records),
        'input_tokens': sum(record.get('input_tokens') or 0 for record in records),
        'output_tokens': sum(record.get('output_tokens') or 0 for record in records),
        'tokens_complete': all(
            record.get('input_tokens') is not None and record.get('output_tokens') is not None
            for record in records
        ),
        'known_cost_usd': known,
        'cost_usd': known if complete else None,
        'cost_complete': complete,
    }


def record_inference_response(data: dict, *, model: str, operation: str) -> None:
    records = active_llm_usage.get()
    if records is None:
        return
    usage = data.get('usage') or {}
    status = data.get('inference_status') or {}
    records.append(
        {
            'model': data.get('model') or model,
            'operation': operation,
            'input_tokens': usage.get('prompt_tokens', status.get('tokens_input')),
            'output_tokens': 0,
            'cost_usd': usage.get('cost', status.get('cost')),
            'response_id': None,
        }
    )
