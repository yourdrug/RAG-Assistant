"""Conservative applicability checks for procedural stages and transmission timing."""

from collections.abc import Iterator
import re

from domain.rag.evidence.assessment import (
    EvidenceAssessment,
    EvidenceClaim,
    EvidenceFragment,
    EvidenceStatus,
)
from infrastructure.domain_packs.belarus_law.evidence.predicates import listed_goods_polarities


ACT_NUMBER_RE = re.compile(r"(?:\bN|№)\s*(\d+(?:[/\-]\d+)*)", re.IGNORECASE)
STAGE_PATTERNS = {
    "хранение": re.compile(r"хранени", re.IGNORECASE),
    "отгрузка": re.compile(r"отгруз", re.IGNORECASE),
    "реализация": re.compile(r"реализац", re.IGNORECASE),
}
STAGE_START_RE = re.compile(r"^При\s+(?:передаче|отгрузке|реализации)\b", re.IGNORECASE)
TIMING_RE = re.compile(r"в режиме|раз[а]? в сут|ежесуточ|ежеднев", re.IGNORECASE)


def normalize_act_number(number: str) -> str:
    return re.sub(r"[/\-]", "", number)


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

    if not stages and re.search(r"логистическ\w* оператор", question, re.IGNORECASE):
        return ("хранение", "отгрузка")

    return stages


def procedure_question(question: str) -> bool:
    return bool(
        re.search(r"поряд|этап|последующ", question, re.IGNORECASE) and requested_procedure_stages(question)
    )


def timing_question(question: str) -> bool:
    return bool(re.search(r"в каком режиме|как часто|периодичност", question, re.IGNORECASE))


CHANGE_QUESTION_RE = re.compile(r"\bизмен(?:ил|ит|яет|яют|ял)\w*\s+ли\b", re.IGNORECASE)
AMENDMENT_INSTRUCTION_RE = re.compile(
    r"в\s+таблиц|заменить|изложить|дополнить|исключить|после\s+позици|позици[юи]", re.IGNORECASE
)
MIN_SUBJECT_QUOTE_CHARS = 20
MAX_SUBJECT_QUOTE_CHARS = 300


def timing_change_question(question: str) -> bool:
    """Whether the question asks if a named act changed how often information is transmitted.

    Such a question compares the amending act with the norm that sets the timing;
    the timing applicability check of a single act does not answer it.
    """
    return (
        timing_question(question)
        and bool(act_numbers(question))
        and bool(CHANGE_QUESTION_RE.search(question))
    )


def timing_requested(question: str) -> bool:
    """Broader than ``timing_question``: retrieval may look for timing, answers stay generated."""
    return bool(re.search(r"режим|периодич|как часто|частот", question, re.IGNORECASE))


CLASSIFIER_RE = re.compile(
    r"\bклассификатор\w*\s+(.+?)(?=,|[?]|\s+согласно\b|\s+на\s+котор|$)", re.IGNORECASE
)
APPROVER_QUESTION_RE = re.compile(r"утвержда|утверд|какой\s+орган|кем", re.IGNORECASE)
ACT_REFERENCE_RE = re.compile(r"(?:согласно|в\s+соответствии\s+с)\s+приложени\w*\s+\S+", re.IGNORECASE)


def approved_classifier(question: str) -> str | None:
    """The classifier whose approving body is asked for («какой орган утверждает классификатор X»)."""
    match = CLASSIFIER_RE.search(question)
    return match[1].strip() if match and APPROVER_QUESTION_RE.search(question) else None


def classifier_reference(classifier: str, text: str) -> bool:
    """A line naming the classifier and the appendix of the act that establishes it.

    The referring structures and the defining act both state it this way; they
    rank below provisions that only share the referring structure's vocabulary.
    """
    stems = {word[:5] for word in re.findall(r"[а-яё]{4,}", classifier.casefold())}
    return bool(stems) and any(
        "классификат" in line.casefold()
        and ACT_REFERENCE_RE.search(line)
        and stems <= {word[:5] for word in re.findall(r"[а-яё]{4,}", line.casefold())}
        for line in text.splitlines()
    )


def timing_provision(text: str) -> bool:
    """A sentence that sets how often information is transmitted."""
    return any(
        re.search(r"переда[её]т|передаются", line, re.IGNORECASE) and TIMING_RE.search(line)
        for line in text.splitlines()
    )


def procedure_stage(line: str) -> str | None:
    if re.match(r"При реализации\b", line, re.IGNORECASE):
        return "реализация"

    if re.match(r"При отгрузке\b", line, re.IGNORECASE):
        return "отгрузка"

    condition = re.split(r"оформля|составля|содерж|получатель|ТТН|ТН-", line, maxsplit=1)[0]
    stages = [name for name, pattern in STAGE_PATTERNS.items() if pattern.search(condition)]
    return stages[0] if len(stages) == 1 else None


def procedure_quotes(fragment: EvidenceFragment, terms: set[str]) -> Iterator[tuple[str, str, str]]:
    lines = fragment.text.splitlines()

    for index, line in enumerate(lines):
        if not STAGE_START_RE.match(line):
            continue

        if terms and not terms <= {word[:7] for word in re.findall(r"[а-яё]{5,}", line.casefold())}:
            continue

        # Classify the conditional clause, not a later reference to a different stage.
        stage = procedure_stage(line)

        if stage is None:
            continue

        quote = line

        if index + 1 < len(lines) and lines[index + 1].startswith("В накладной "):
            quote += "\n" + lines[index + 1]

        yield stage, line, quote


def procedure_assessment(question: str, fragments: list[EvidenceFragment]) -> EvidenceAssessment:
    required = requested_procedure_stages(question)
    participant = re.search(r"\bс\s+(.+?)\s+согласно\b", question, re.IGNORECASE)
    terms = (
        {word[:7] for word in re.findall(r"[а-яё]{5,}", participant[1].casefold())} if participant else set()
    )
    candidates = []
    claims: dict[tuple[str, str], EvidenceClaim] = {}

    for fragment in sorted(scoped_fragments(question, fragments), key=lambda f: f.chunk_index):
        for stage, line, quote in procedure_quotes(fragment, terms):
            candidates.append(fragment)
            key = (stage, line)

            if key not in claims or len(quote) > len(claims[key].quote):
                claims[key] = EvidenceClaim(stage, quote, fragment.source)

    if evidence_conflicts(candidates):
        return EvidenceAssessment(EvidenceStatus.CONFLICT, missing=("акт или редакция этапов",))

    present = {claim.label for claim in claims.values()}
    missing = tuple(stage for stage in required if stage not in present)

    if not required:
        missing = ("запрошенные этапы",)

    status = (
        EvidenceStatus.COMPLETE
        if not missing
        else EvidenceStatus.PARTIAL
        if claims
        else EvidenceStatus.INSUFFICIENT
    )
    return EvidenceAssessment(status, tuple(claims.values()), missing)


def timing_line_matches(question: str, line: str, previous: str, fragment: EvidenceFragment) -> bool:
    if not re.search(r"переда[её]т|передаются", line, re.IGNORECASE) or not TIMING_RE.search(line):
        return False

    scope = previous + "\n" + line if previous.rstrip().endswith(":") else line

    if not (
        (not re.search(r"сведени", question, re.IGNORECASE) or re.search(r"сведени", scope, re.IGNORECASE))
        and timing_goods_scope_matches(question, scope, fragment)
        and (
            not re.search(r"EDI[- ]провайдер", question, re.IGNORECASE)
            or re.search(r"EDI[- ]провайдер", line, re.IGNORECASE)
        )
        and transmission_participants_match(question, scope)
    ):
        return False

    requested = listed_goods_polarities(question)
    return not requested or listed_goods_polarities(scope) == requested or bool(fragment.linked_scope)


def timing_goods_scope_matches(question: str, scope: str, fragment: EvidenceFragment) -> bool:
    if not re.search(r"товар", question, re.IGNORECASE) or re.search(r"товар", scope, re.IGNORECASE):
        return True

    return bool(
        fragment.linked_scope
        and re.search(r"сведения, указанные в пункте \d+ настоящего постановления", scope, re.IGNORECASE)
    )


def transmission_participants_match(question: str, scope: str) -> bool:
    if re.search(r"EDI[- ]провайдер\w* грузоотправител", question, re.IGNORECASE) and not re.search(
        r"EDI[- ]провайдер\w* грузоотправител", scope, re.IGNORECASE
    ):
        return False

    recipient = r"\bв\s+(?:МНС\b|Министерство по налогам и сборам)"

    return not (
        re.search(recipient, question, re.IGNORECASE) and not re.search(recipient, scope, re.IGNORECASE)
    )


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
            if not timing_line_matches(question, line, lines[index - 1] if index else "", fragment):
                continue

            candidates.append(fragment)
            claims.setdefault(line.strip(), EvidenceClaim("режим передачи", line.strip(), fragment.source))

    if evidence_conflicts(candidates) or len(claims) > 1:
        return EvidenceAssessment(EvidenceStatus.CONFLICT, missing=("единственный применимый режим",))

    if not claims:
        return EvidenceAssessment(
            EvidenceStatus.INSUFFICIENT, missing=("действие, категория товаров и режим передачи",)
        )

    return EvidenceAssessment(EvidenceStatus.COMPLETE, tuple(claims.values()))


def amendment_subject_quote(fragment: EvidenceFragment) -> str:
    """The first informative line of an amending act that says what it changes, else its whole text."""
    lines = [line.strip() for line in fragment.text.splitlines() if line.strip()]
    quote = next(
        (
            line
            for line in lines
            if AMENDMENT_INSTRUCTION_RE.search(line) and len(line) >= MIN_SUBJECT_QUOTE_CHARS
        ),
        " ".join(lines),
    )
    return quote if len(quote) <= MAX_SUBJECT_QUOTE_CHARS else quote[:MAX_SUBJECT_QUOTE_CHARS].rstrip() + "…"


def timing_change_assessment(question: str, fragments: list[EvidenceFragment]) -> EvidenceAssessment:
    """Pair the subject of the named amending act with the norm that sets transmission timing.

    The named act's own fragments show what it changes; a timing sentence of any
    act is the norm to compare with. Both are needed for a conclusion; with one
    of them the confirmed part is quoted and the other is named as missing.
    """
    amending = sorted(scoped_fragments(question, fragments), key=lambda f: f.chunk_index)
    claims = [EvidenceClaim("предмет изменения", amendment_subject_quote(f), f.source) for f in amending[:1]]
    norms: dict[str, EvidenceClaim] = {}

    for fragment in fragments:
        lines = fragment.text.splitlines()

        for index, line in enumerate(lines):
            if timing_line_matches(question, line, lines[index - 1] if index else "", fragment):
                norms.setdefault(
                    line.strip(), EvidenceClaim("норма о периодичности", line.strip(), fragment.source)
                )

    claims.extend(norms.values())
    missing = (
        *(() if amending else ("текст изменений, внесённых названным актом",)),
        *(() if norms else ("норма, устанавливающая периодичность передачи сведений",)),
    )
    status = (
        EvidenceStatus.COMPLETE
        if not missing
        else EvidenceStatus.PARTIAL
        if claims
        else EvidenceStatus.INSUFFICIENT
    )
    return EvidenceAssessment(status, tuple(claims), missing)
