"""Conservative applicability checks for procedural stages and transmission timing."""

import re

from domain.services.evidence_queries import listed_goods_polarities
from domain.value_objects.evidence_assessment import (
    EvidenceAssessment,
    EvidenceClaim,
    EvidenceFragment,
    EvidenceStatus,
)

ACT_NUMBER_RE = re.compile(r'(?:\bN|№)\s*(\d+(?:[/\-]\d+)*)', re.IGNORECASE)
STAGE_PATTERNS = {
    'хранение': re.compile(r'хранени', re.IGNORECASE),
    'отгрузка': re.compile(r'отгруз', re.IGNORECASE),
    'реализация': re.compile(r'реализац', re.IGNORECASE),
}
STAGE_START_RE = re.compile(r'^При\s+(?:передаче|отгрузке|реализации)\b', re.IGNORECASE)
TIMING_RE = re.compile(r'в режиме|раз[а]? в сут|ежесуточ|ежеднев', re.IGNORECASE)


def normalize_act_number(number: str) -> str:
    return re.sub(r'[/\-]', '', number)


def act_numbers(text: str) -> set[str]:
    return {normalize_act_number(m[1]) for m in ACT_NUMBER_RE.finditer(text)}


def scoped_fragments(question: str, fragments: list[EvidenceFragment]) -> list[EvidenceFragment]:
    requested = act_numbers(question)
    return [f for f in fragments if not requested or (f.act_number and requested == {f.act_number})]


def evidence_conflicts(fragments: list[EvidenceFragment]) -> bool:
    identities = {(f.document_id, f.version_id) for f in fragments}
    # Unknown version cannot be combined with a known version or another document.
    return len(identities) > 1


def requested_procedure_stages(question: str) -> tuple[str, ...]:
    stages = tuple(name for name, pattern in STAGE_PATTERNS.items() if pattern.search(question))
    if not stages and re.search(r'логистическ\w* оператор', question, re.IGNORECASE):
        return ('хранение', 'отгрузка')
    return stages


def procedure_question(question: str) -> bool:
    return bool(
        re.search(r'поряд|этап|последующ', question, re.IGNORECASE) and requested_procedure_stages(question)
    )


def timing_question(question: str) -> bool:
    return bool(re.search(r'в каком режиме|как часто|периодичност', question, re.IGNORECASE))


def procedure_stage(line: str) -> str | None:
    if re.match(r'При реализации\b', line, re.IGNORECASE):
        return 'реализация'
    if re.match(r'При отгрузке\b', line, re.IGNORECASE):
        return 'отгрузка'
    condition = re.split(r'оформля|составля|содерж|получатель|ТТН|ТН-', line, maxsplit=1)[0]
    stages = [name for name, pattern in STAGE_PATTERNS.items() if pattern.search(condition)]
    return stages[0] if len(stages) == 1 else None


def procedure_assessment(question: str, fragments: list[EvidenceFragment]) -> EvidenceAssessment:
    required = requested_procedure_stages(question)
    participant = re.search(r'\bс\s+(.+?)\s+согласно\b', question, re.IGNORECASE)
    terms = (
        {word[:7] for word in re.findall(r'[а-яё]{5,}', participant[1].casefold())} if participant else set()
    )
    candidates = []
    claims: dict[tuple[str, str], EvidenceClaim] = {}
    for fragment in sorted(scoped_fragments(question, fragments), key=lambda f: f.chunk_index):
        lines = fragment.text.splitlines()
        for index, line in enumerate(lines):
            if not STAGE_START_RE.match(line):
                continue
            if terms and not terms <= {word[:7] for word in re.findall(r'[а-яё]{5,}', line.casefold())}:
                continue
            # Classify the conditional clause, not a later reference to a different stage.
            stage = procedure_stage(line)
            if stage is None:
                continue
            quote = line
            if index + 1 < len(lines) and lines[index + 1].startswith('В накладной '):
                quote += '\n' + lines[index + 1]
            candidates.append(fragment)
            key = (stage, line)
            if key not in claims or len(quote) > len(claims[key].quote):
                claims[key] = EvidenceClaim(stage, quote, fragment.source)
    if evidence_conflicts(candidates):
        return EvidenceAssessment(EvidenceStatus.CONFLICT, missing=('акт или редакция этапов',))
    present = {claim.label for claim in claims.values()}
    missing = tuple(stage for stage in required if stage not in present)
    if not required:
        missing = ('запрошенные этапы',)
    status = (
        EvidenceStatus.COMPLETE
        if not missing
        else EvidenceStatus.PARTIAL
        if claims
        else EvidenceStatus.INSUFFICIENT
    )
    return EvidenceAssessment(status, tuple(claims.values()), missing)


def timing_line_matches(question: str, line: str, previous: str, fragment: EvidenceFragment) -> bool:
    if not re.search(r'переда[её]т|передаются', line, re.IGNORECASE) or not TIMING_RE.search(line):
        return False
    scope = previous + '\n' + line if previous.rstrip().endswith(':') else line
    if re.search(r'сведени', question, re.IGNORECASE) and not re.search(r'сведени', scope, re.IGNORECASE):
        return False
    if re.search(r'товар', question, re.IGNORECASE) and not re.search(r'товар', scope, re.IGNORECASE):
        if not (
            fragment.linked_scope
            and re.search(r'сведения, указанные в пункте \d+ настоящего постановления', scope, re.IGNORECASE)
        ):
            return False
    if re.search(r'EDI[- ]провайдер', question, re.IGNORECASE) and not re.search(
        r'EDI[- ]провайдер', line, re.IGNORECASE
    ):
        return False
    if not transmission_participants_match(question, scope):
        return False
    requested = listed_goods_polarities(question)
    return not requested or listed_goods_polarities(scope) == requested or bool(fragment.linked_scope)


def transmission_participants_match(question: str, scope: str) -> bool:
    if re.search(r'EDI[- ]провайдер\w* грузоотправител', question, re.IGNORECASE) and not re.search(
        r'EDI[- ]провайдер\w* грузоотправител', scope, re.IGNORECASE
    ):
        return False
    recipient = r'\bв\s+(?:МНС\b|Министерство по налогам и сборам)'
    if re.search(recipient, question, re.IGNORECASE) and not re.search(recipient, scope, re.IGNORECASE):
        return False
    return True


def timing_assessment(question: str, fragments: list[EvidenceFragment]) -> EvidenceAssessment:
    requested = listed_goods_polarities(question)
    candidates = []
    claims: dict[str, EvidenceClaim] = {}
    for fragment in scoped_fragments(question, fragments):
        # A mixed category block has no unique local interpretation.
        actual = listed_goods_polarities(fragment.linked_scope or fragment.text)
        if requested and (len(requested) != 1 or actual != requested):
            continue
        lines = fragment.text.splitlines()
        for index, line in enumerate(lines):
            if not timing_line_matches(question, line, lines[index - 1] if index else '', fragment):
                continue
            candidates.append(fragment)
            claims.setdefault(line.strip(), EvidenceClaim('режим передачи', line.strip(), fragment.source))
    if evidence_conflicts(candidates) or len(claims) > 1:
        return EvidenceAssessment(EvidenceStatus.CONFLICT, missing=('единственный применимый режим',))
    if not claims:
        return EvidenceAssessment(
            EvidenceStatus.INSUFFICIENT, missing=('действие, категория товаров и режим передачи',)
        )
    return EvidenceAssessment(EvidenceStatus.COMPLETE, tuple(claims.values()))
