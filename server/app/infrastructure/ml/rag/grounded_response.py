"""Extractive responses for evidence that must retain exact source relationships."""

import re

from domain.services.evidence_queries import table_scope_confirmed
from domain.services.evidence_applicability import (
    act_numbers,
    evidence_conflicts,
    procedure_assessment,
    procedure_question,
    timing_assessment,
    timing_question,
)
from domain.value_objects.evidence_assessment import EvidenceAssessment, EvidenceFragment, EvidenceStatus
from infrastructure.ml.rag.evidence_focus import evidence_reading_map, requested_fields, matching_field_rows
from infrastructure.ml.rag.rag_formatting import content_with_parent_scope


def missing_field_response(context: str, question: str) -> str | None:
    missing = [item for item in evidence_reading_map(context, question) if "в контексте не найдена" in item]
    if not missing:
        return None
    return "\n".join(missing) + " Нельзя подтвердить значение или сделать вывод о его изменении."


def extractive_timing_response(docs, question: str) -> str | None:
    """Return a single explicit timing norm from the act named in the question."""
    if not timing_question(question):
        return None
    assessment = timing_assessment(question, evidence_fragments(docs))
    if assessment.status != EvidenceStatus.COMPLETE:
        return None
    return render_evidence_assessment(assessment)


def evidence_fragments(docs) -> list[EvidenceFragment]:
    fragments = []
    for doc, _ in docs:
        metadata = doc.metadata
        source = metadata.get('citation_id')
        if not source:
            continue
        numbers = act_numbers(metadata.get('doc_title', ''))
        if metadata.get('act_number'):
            numbers = {str(metadata['act_number']).replace('/', '').replace('-', '')}
        filename_numbers = act_numbers(metadata.get('source', ''))
        if numbers and filename_numbers and numbers != filename_numbers:
            continue
        numbers = numbers or filename_numbers
        fragments.append(
            EvidenceFragment(
                text=content_with_parent_scope(doc),
                source=source,
                document_id=metadata.get('document_id') or metadata.get('source'),
                version_id=metadata.get('act_version_id'),
                act_number=next(iter(numbers)) if len(numbers) == 1 else None,
                chunk_index=metadata.get('chunk_index', 0),
                linked_scope=metadata.get('verified_timing_scope', ''),
            )
        )
    return fragments


def render_evidence_assessment(assessment: EvidenceAssessment) -> str:
    parts = [f'{claim.quote} [{claim.source}]' for claim in assessment.claims]
    if assessment.status == EvidenceStatus.CONFLICT:
        parts.append(
            'Конфликт источников: нельзя подтвердить применимый акт, редакцию или единственный режим.'
        )
    elif assessment.missing:
        parts.append(
            'Не подтверждены: ' + ', '.join(assessment.missing) + '. Нельзя подтвердить полный вывод.'
        )
    return '\n\n'.join(parts)


def scoped_evidence_response(
    docs, context: str, question: str
) -> tuple[str | None, EvidenceAssessment | None]:
    """Keep unsupported scoped requests out of unrestricted model generation."""
    if procedure_question(question):
        assessment = procedure_assessment(question, evidence_fragments(docs))
        return render_evidence_assessment(assessment), assessment
    if timing_question(question):
        assessment = timing_assessment(question, evidence_fragments(docs))
        return render_evidence_assessment(assessment), assessment
    if requested_fields(question):
        context = applicable_table_context(docs, context, question)
        missing = missing_field_response(context, question)
        if missing:
            return missing, EvidenceAssessment(
                EvidenceStatus.INSUFFICIENT, missing=('целевая строка таблицы',)
            )
        answer = extractive_table_comparison_response(context, question)
        if answer:
            return answer, EvidenceAssessment(EvidenceStatus.COMPLETE)
        if re.search(r'измен|замен|статус|обязательност', question, re.IGNORECASE):
            return amendment_field_response(context, question)
        if 'формат' in question.casefold():
            parts = []
            for number, name in requested_fields(question):
                values, changes = field_format_evidence(context, question, number, name)
                if len(values) == 1 and not changes:
                    value, source = next(iter(values.items()))
                    parts.append(
                        f'Поле {number} «{name}»: в целевой строке указан формат {value}. [{source}]'
                    )
            parts.append('История изменения и полный вывод не подтверждены доступным контекстом.')
            status = EvidenceStatus.PARTIAL if len(parts) > 1 else EvidenceStatus.INSUFFICIENT
            return '\n\n'.join(parts), EvidenceAssessment(status, missing=('история изменения',))
        return field_rows_response(context, question)
    return None, None


def amendment_field_response(context: str, question: str) -> tuple[str, EvidenceAssessment]:
    parts = []
    for source, block in sourced_context_blocks(context):
        for projection in evidence_reading_map(block, question):
            if projection.startswith('Поле ') and 'ПОСЛЕ изменения' in projection:
                if f'{projection} [{source}]' not in parts:
                    parts.append(f'{projection} [{source}]')
    if not parts:
        return (
            'Операция замены целевого поля не подтверждена. Нельзя сделать вывод об изменении.',
            EvidenceAssessment(EvidenceStatus.INSUFFICIENT, missing=('операция замены поля',)),
        )
    # Quote the verified replacement only; do not infer history of other fields.
    parts.append('Приведены подтверждённые замены; история остальных полей не подтверждена.')
    return '\n\n'.join(parts), EvidenceAssessment(EvidenceStatus.PARTIAL)


def field_rows_response(context: str, question: str) -> tuple[str, EvidenceAssessment]:
    parts = []
    for number, name in requested_fields(question):
        rows = {
            row: source
            for source, block in sourced_context_blocks(context)
            if table_scope_confirmed(question, block)
            for row in matching_field_rows(block, number, name)
        }
        if len(rows) != 1:
            return (
                'Единственная применимая строка поля не подтверждена. Нельзя сделать полный вывод.',
                EvidenceAssessment(EvidenceStatus.CONFLICT, missing=('единственная строка поля',)),
            )
        row, source = next(iter(rows.items()))
        parts.append(f'{row} [{source}]')
    return '\n\n'.join(parts), EvidenceAssessment(EvidenceStatus.COMPLETE)


def applicable_table_context(docs, context: str, question: str) -> str:
    requested = act_numbers(question)
    if not requested:
        return context
    allowed = set(requested)
    if re.search(r'измен|постановлением|замен', question, re.IGNORECASE):
        for doc, score in docs:
            fragments = evidence_fragments([(doc, score)])
            if fragments and fragments[0].act_number in requested:
                allowed.update(doc.metadata.get('verified_amended_acts', ()))
    fragments = evidence_fragments(docs)
    for number in allowed:
        if evidence_conflicts([f for f in fragments if f.act_number == number]):
            return ''
    by_source = {f.source: f for f in fragments}
    blocks = []
    for source, block in sourced_context_blocks(context):
        fragment = by_source.get(source)
        identities = {fragment.act_number} if fragment else act_numbers(block.splitlines()[0].split('|')[0])
        if len(identities) == 1 and identities <= allowed:
            blocks.append(block if re.match(r'\s*\[\d+\] ', block) else f'[{source}] {block}')
    return '\n\n---\n\n'.join(blocks)


def sourced_context_blocks(context: str) -> list[tuple[int, str]]:
    result = []
    source = None
    for block in re.split(r"\n\s*---\s*\n", context):
        reference = re.match(r"\[(\d+)\] ", block.strip())
        if reference:
            source = int(reference[1])
        if source:
            result.append((source, block))
    return result


def field_format_evidence(context, question, number, name):
    values: dict[str, int] = {}
    changes = []
    for source, block in sourced_context_blocks(context):
        if not table_scope_confirmed(question, block):
            continue
        for row in matching_field_rows(block, number, name):
            formats = re.findall(r"\b(?:an|n|a)\.\.\d+", row.split("|")[2])
            if len(formats) == 1:
                values.setdefault(formats[0], source)
        for projection in evidence_reading_map(block, question):
            if not projection.startswith(f"Поле {number} «") or "ПОСЛЕ изменения" not in projection:
                continue
            before, after = projection.split("ПОСЛЕ изменения", 1)
            old = re.findall(r"\b(?:an|n|a)\.\.\d+", before)
            new = re.findall(r"\b(?:an|n|a)\.\.\d+", after)
            if len(old) == len(new) == 1:
                changes.append((old[0], new[0], source))
    return values, changes


def extractive_table_comparison_response(context: str, question: str) -> str | None:
    """Compare verified formats without inventing an old value for an untouched field."""
    fields = requested_fields(question)
    if len(fields) != 2 or "формат" not in question.casefold():
        return None
    answers = []
    replacements = 0
    for number, name in fields:
        values, changes = field_format_evidence(context, question, number, name)
        if changes:
            if len(set(changes)) != 1:
                return None
            old, new, source = changes[0]
            answers.append(f"Поле {number} «{name}»: поправкой формат изменён {old} → {new}. [{source}]")
            replacements += 1
        elif len(values) == 1:
            value, source = next(iter(values.items()))
            answers.append(f"Поле {number} «{name}»: в целевой строке указан формат {value}. [{source}]")
        else:
            return None
    if replacements != 1:
        return None
    return "\n\n".join(answers) + "\n\nПоправку к одному полю нельзя переносить на другое поле."


def extractive_procedure_response(docs, question: str) -> str | None:
    """Return a full procedure only when every requested stage is applicable."""
    if not procedure_question(question):
        return None
    assessment = procedure_assessment(question, evidence_fragments(docs))
    if assessment.status != EvidenceStatus.COMPLETE:
        return None
    return render_evidence_assessment(assessment)
