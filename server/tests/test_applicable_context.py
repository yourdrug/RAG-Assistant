"""Source links are recovered only inside an authorized document and version."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.page_content_type import PageContentType
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.ml.rag.applicable_context import enrich_applicable_context
from infrastructure.ml.rag.grounded_response import extractive_timing_response, missing_field_response
from infrastructure.ml.rag.rag_formatting import content_with_parent_scope

USER = ChatContext(1, UserKind.INTERNAL, UserRole.ADMIN).to_user_context()
QUESTION = 'Формат поля «Код» (позиция 18 таблицы 4.2, BLRDLN)?'


def source_chunk(index, text, table_id=None, version=1):
    return ChunkSearchResult(
        chunk_id=index,
        document_id=1,
        chunk_index=index,
        filename='Постановление N 71.docx',
        content=text,
        act_version_id=version,
        content_type=PageContentType.TABLE.value if table_id else None,
        context_metadata={'table_id': table_id} if table_id else {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['', 'gap', 'version', 'other_table'])
async def test_table_scope_requires_contiguous_matching_table_and_version(mutation):
    doc = Document(
        page_content='Теги ЭТН:\n| 18 | Код an..15 |',
        metadata={
            'source': 'Постановление N 71.docx',
            'document_id': 1,
            'act_version_id': 1,
            'chunk_index': 12,
            'table_id': 'table-2',
            'content_type': PageContentType.TABLE.value,
        },
    )
    neighbors = [
        source_chunk(10, 'Таблица 4.2 - структура сообщения BLRDLN'),
        source_chunk(11, '| 1 | Начало |', 'table-2'),
        source_chunk(12, doc.page_content, 'table-2'),
    ]
    if mutation == 'gap':
        neighbors.pop(1)
    elif mutation == 'version':
        neighbors[0].act_version_id = 2
    elif mutation == 'other_table':
        neighbors[1].context_metadata['table_id'] = 'table-3'
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=neighbors))
    result = await enrich_applicable_context([(doc, 0.9)], QUESTION, search, USER)
    assert bool(missing_field_response(content_with_parent_scope(result[0][0]), QUESTION)) == bool(mutation)
    assert 'parent_units' not in doc.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [1, 2])
async def test_timing_reference_requires_definition_and_appendix_in_same_version(version):
    question = 'В каком режиме EDI-провайдер передаёт сведения о товарах, включённых в перечни, по N 71?'
    doc = Document(
        page_content='2.2. сведения, указанные в пункте 1 настоящего постановления:\n'
        'передаются EDI-провайдером в режиме реального времени;',
        metadata={
            'source': 'Постановление N 71.docx',
            'document_id': 1,
            'act_version_id': 1,
            'chunk_index': 4,
            'citation_id': 1,
        },
    )
    definition = source_chunk(1, '1. Определить состав сведений согласно приложению.')
    appendix = source_chunk(27, 'Приложение\nк постановлению N 71\nСОСТАВ СВЕДЕНИЙ')
    category = source_chunk(30, '3. При передаче сведений о товарах, включенных в перечни:', version=version)
    search = SimpleNamespace(search_substring=AsyncMock(side_effect=[[definition], [appendix], [category]]))
    result = await enrich_applicable_context([(doc, 0.9)], question, search, USER)
    answer = extractive_timing_response(result, question)
    assert bool(answer) == (version == 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [1, 2])
async def test_amendment_target_link_is_explicit_and_same_version(version):
    question = 'Изменился ли формат поля «Код» (позиция 18 таблицы 4.2 BLRDLN) постановлением N 71?'
    doc = Document(
        page_content='Таблица 4.2 BLRDLN\n| 18 | Код an..8 | М |',
        metadata={
            'source': 'Постановление N 71.docx',
            'document_id': 1,
            'act_version_id': 1,
            'chunk_index': 20,
            'citation_id': 1,
        },
    )
    link = source_chunk(
        10,
        '1. Внести в приложение к структуре, утвержденным постановлением N 58, следующие изменения:',
        version=version,
    )
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[link]))
    result = await enrich_applicable_context([(doc, 0.9)], question, search, USER)
    assert result[0][0].metadata.get('verified_amended_acts') == (['58'] if version == 1 else None)
    assert 'verified_amended_acts' not in doc.metadata
