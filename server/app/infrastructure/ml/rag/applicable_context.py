"""Recover source-bound scope without inferring it from retrieval relevance."""

import logging
import re
from copy import copy

from domain.services.evidence_applicability import act_numbers, normalize_act_number, timing_question
from domain.services.evidence_queries import listed_goods_polarities, table_scope_confirmed
from domain.value_objects.page_content_type import PageContentType
from domain.value_objects.search_mode import SearchMode
from infrastructure.ml.rag.evidence_focus import matching_field_rows, requested_fields
from infrastructure.ml.rag.rag_formatting import content_with_parent_scope

log = logging.getLogger('default')
SCOPE_WINDOW = 32
MAX_SCOPE_LOOKUPS = 8
MAX_AMENDMENT_LOOKUPS = 3


def same_version(chunk, metadata) -> bool:
    return chunk.document_id == metadata.get('document_id') and chunk.act_version_id == metadata.get(
        'act_version_id'
    )


def table_heading_scope(neighbors, metadata) -> str | None:
    local = sorted((c for c in neighbors if same_version(c, metadata)), key=lambda c: c.chunk_index)
    index = metadata['chunk_index']
    headings = [c for c in local if c.chunk_index < index and re.match(r'Таблица\s+\d', c.content)]
    if not headings:
        return None
    heading = headings[-1]
    chain = [c for c in local if heading.chunk_index <= c.chunk_index <= index]
    if [c.chunk_index for c in chain] != list(range(heading.chunk_index, index + 1)):
        return None
    table_id = metadata.get('table_id')
    if not table_id or any(c.context_metadata.get('table_id') != table_id for c in chain[1:]):
        return None
    return heading.content


async def timing_reference_scope(doc, question, chunk_search, user) -> str | None:
    text = content_with_parent_scope(doc)
    reference = re.search(r'сведения, указанные в пункте (\d+) настоящего постановления', text, re.IGNORECASE)
    category = listed_goods_polarities(question)
    if not reference or len(category) != 1:
        return None
    metadata = doc.metadata
    options = {
        'user': user,
        'document_id': metadata['document_id'],
        'limit': SCOPE_WINDOW,
        'mode': SearchMode.ICONTAINS.value,
    }
    definitions = await chunk_search.search_substring(query='согласно приложению', **options)
    definitions = [
        c
        for c in definitions
        if same_version(c, metadata) and re.search(rf'^{reference[1]}\.\s', c.content, re.MULTILINE)
    ]
    if len(definitions) != 1:
        return None
    appendices = await chunk_search.search_substring(query='Приложение', **options)
    appendices = [
        c
        for c in appendices
        if same_version(c, metadata)
        and re.search(r'Приложение\s+к постановлению', c.content, re.IGNORECASE)
        and 'СОСТАВ' in c.content.upper()
        and 'СВЕДЕНИЙ' in c.content.upper()
    ]
    if len(appendices) != 1:
        return None
    clauses = await chunk_search.search_substring(query='включен', **options)
    clauses = [
        c
        for c in clauses
        if same_version(c, metadata)
        and c.chunk_index > appendices[0].chunk_index
        and listed_goods_polarities(c.content) == category
        and re.search(r'При передаче', c.content, re.IGNORECASE)
    ]
    if len(clauses) != 1:
        return None
    return '\n'.join(c.content for c in (definitions[0], appendices[0], clauses[0]))


async def enrich_applicable_context(docs, question, chunk_search, user):
    """Attach exact table headings or explicit point -> appendix -> category links.

    Every lookup is ACL-scoped. No missing neighbor, version mismatch, or lookup
    failure is permission to fill a scope from the question itself.
    """
    if chunk_search is None:
        return docs
    docs = await enrich_amendment_scope(docs, question, chunk_search, user)
    result = []
    lookups = 0
    fields = requested_fields(question)
    for doc, score in docs:
        metadata = doc.metadata
        scope = None
        timing_scope = None
        if metadata.get('document_id') and lookups < MAX_SCOPE_LOOKUPS:
            try:
                if (
                    fields
                    and metadata.get('content_type') == PageContentType.TABLE.value
                    and isinstance(metadata.get('chunk_index'), int)
                    and any(matching_field_rows(doc.page_content, n, name) for n, name in fields)
                    and not table_scope_confirmed(question, content_with_parent_scope(doc))
                ):
                    lookups += 1
                    neighbors = await chunk_search.get_neighbors(
                        metadata['document_id'],
                        metadata['chunk_index'],
                        window=SCOPE_WINDOW,
                        user=user,
                    )
                    scope = table_heading_scope(neighbors, metadata)
                elif (
                    timing_question(question)
                    and not listed_goods_polarities(content_with_parent_scope(doc))
                    and re.search(
                        r'сведения, указанные в пункте \d+ настоящего постановления', doc.page_content
                    )
                    and lookups + 3 <= MAX_SCOPE_LOOKUPS
                ):
                    lookups += 3
                    timing_scope = await timing_reference_scope(doc, question, chunk_search, user)
                    scope = timing_scope
            except Exception:
                # Enrichment is optional: a failed lookup leaves the scope unconfirmed.
                log.warning(
                    'Applicability scope lookup failed for document_id=%s',
                    metadata['document_id'],
                    exc_info=True,
                )
        if scope:
            doc = copy(doc)
            doc.metadata = {
                **metadata,
                'parent_units': [*metadata.get('parent_units', []), {'content': scope}],
            }
            if timing_scope:
                doc.metadata['verified_timing_scope'] = timing_scope
        result.append((doc, score))
    return result


async def enrich_amendment_scope(docs, question, chunk_search, user):
    requested = act_numbers(question)
    if not requested or not re.search(r'измен|постановлением|замен', question, re.IGNORECASE):
        return docs
    links: dict[tuple, list[str]] = {}
    for doc, _ in docs:
        metadata = doc.metadata
        identity = (metadata.get('document_id'), metadata.get('act_version_id'))
        if not all(identity) or identity in links or len(links) >= MAX_AMENDMENT_LOOKUPS:
            continue
        number = metadata.get('act_number')
        actual = (
            {normalize_act_number(str(number))}
            if number
            else act_numbers(metadata.get('doc_title') or metadata.get('source', ''))
        )
        if actual != requested:
            continue
        links[identity] = []
        try:
            roots = await chunk_search.search_substring(
                query='следующие изменения',
                user=user,
                document_id=identity[0],
                limit=SCOPE_WINDOW,
                mode=SearchMode.ICONTAINS.value,
            )
        except Exception:
            log.warning('Amendment root lookup failed for document_id=%s', identity[0], exc_info=True)
            continue
        targets = amendment_targets(roots, metadata)
        if len(targets) == 1:
            links[identity] = sorted(targets)
    result = []
    for doc, score in docs:
        target = links.get((doc.metadata.get('document_id'), doc.metadata.get('act_version_id')))
        if target:
            doc = copy(doc)
            doc.metadata = {**doc.metadata, 'verified_amended_acts': target}
        result.append((doc, score))
    return result


def amendment_targets(roots, metadata) -> set[str]:
    targets = set()
    for chunk in roots:
        if not same_version(chunk, metadata):
            continue
        match = re.search(
            r'Внести[\s\S]*?постановлен\w+[\s\S]*?(?:\bN|№)\s*(\d+(?:[/\-]\d+)*),?\s*следующие изменения',
            chunk.content,
            re.IGNORECASE,
        )
        if match:
            targets.add(normalize_act_number(match[1]))
    return targets
