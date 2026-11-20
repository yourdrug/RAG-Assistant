"""Shared semantic criteria for grouped and individual benchmark judges."""

from infrastructure.ml.rag.prompt_blocks import is_not_found_answer


JUDGE_RUBRIC_VERSION = "2026-11-20-refusal-correctness-absence-claims-v5"

REFUSAL_FAITHFULNESS_SCORE = 10.0
REFUSAL_FAITHFULNESS_REASON = "Отказ без фактических утверждений (правило рубрики, судья не вызывался)"


def refusal_faithfulness(answer: str) -> tuple[float, str] | None:
    """Faithfulness of a whole-answer refusal, fixed by the rubric instead of the judge.

    A refusal makes no factual claims, so nothing can be unsupported. Asked anyway, the
    judge sees the expected answer type and required facts and penalises the refusal here,
    which belongs to correctness.
    """
    if not is_not_found_answer(answer):
        return None

    return REFUSAL_FAITHFULNESS_SCORE, REFUSAL_FAITHFULNESS_REASON


# A whole-answer refusal either is the expected behaviour or cannot match the expected answer;
# the judge added only noise here (the same refusal of case 550 scored 10, 10, 10 and 0).
REFUSAL_BEHAVIOR_SCORES = {
    "not_in_corpus": 10.0,
    "out_of_scope": 10.0,
    "factual": 0.0,
    "negative": 0.0,
    "false_premise": 0.0,
}
REFUSAL_BEHAVIOR_REASONS = {
    10.0: "Ожидаемый отказ (правило рубрики, судья не вызывался)",
    0.0: "Отказ вместо ожидаемого ответа (правило рубрики, судья не вызывался)",
}


def refusal_behavior(answer: str, kind: str) -> tuple[float, str] | None:
    """Correctness and answer behaviour of a whole-answer refusal, fixed by the answer type.

    An ambiguous question keeps the judge: explaining the ambiguity is not a refusal.
    """
    score = REFUSAL_BEHAVIOR_SCORES.get(kind)

    if score is None or not is_not_found_answer(answer):
        return None

    return score, REFUSAL_BEHAVIOR_REASONS[score]


CORRECTNESS_CRITERIA = (
    "CORRECTNESS: оцени совпадение по смыслу с эталоном, включая полноту, ограничения "
    "и исключения. Полное совпадение = 10, противоречие = 0. "
    "Это отдельная метрика: совпадение с эталоном не доказывает FAITHFULNESS. "
    "Ответ может иметь correctness=10 и faithfulness=0. "
    "При наличии answer_type применяй answer_rubric; оцени смысл, а не совпадение фраз. "
    "Если ответ, как и эталон, отрицает предпосылку вопроса (объекта нет, норма не применяется), "
    "это совпадение, а не противоречие, даже при другой формулировке."
)

FAITHFULNESS_CRITERIA = (
    "FAITHFULNESS: разбей ответ на проверяемые фактические утверждения и для каждого "
    "найди подтверждение именно в предоставленном контексте. Эталон, required_facts, "
    "required_conditions, вопрос и собственные знания судьи не являются источниками. "
    "Проверяй также вводные пояснения, расшифровки сокращений (например, ПК СПТ), "
    "названия систем, числа, сроки, отрицания, условия и исключения. "
    "Правдоподобная расшифровка без подтверждения — неподтверждённый факт. "
    "Наличие ссылки само по себе ничего не доказывает. Не требуй дословного совпадения: "
    "допустимы перефразирование и вывод, однозначно следующий из контекста. "
    "10 — все фактические утверждения подтверждены; иначе оцени долю подтверждённых "
    "утверждений по шкале 0\u201310, но не выше 7 при любом неподтверждённом факте. "
    "Если не подтверждён основной вывод, оценка не выше 3. "
    "Если подтверждений нет или контекст пуст при содержательном ответе, поставь 0, "
    "даже при полном совпадении с эталоном. "
    "Отказ без фактических утверждений = 10 по faithfulness; полезность отказа оценивается отдельно. "
    "Утверждение об отсутствии («в документе нет статьи 40», «формы ТН-1 в инструкции нет», «лимит "
    "не указан») подтверждено, если контекст содержит этот документ, его перечень или соответствующую "
    "норму и названного объекта в них нет; не снижай оценку за то, что отсутствующий объект "
    "не упомянут в контексте. "
    "В reason назови неподтверждённое утверждение, если оно есть."
)

CITATION_SUPPORT_CRITERIA = (
    "CITATION SUPPORT: каждая ссылка должна подтверждать связанное утверждение, включая "
    "расшифровки сокращений, условия и исключения. Наличие ссылки на тематически близкий "
    "документ не является подтверждением. Выдуманные ссылки и неподтверждённые утверждения "
    "снижают оценку. Содержательный ответ без ссылок = 0. "
    "Для корректного отказа без фактических утверждений ссылки не нужны."
)


ANSWER_TYPE_RUBRICS = {
    "factual": "Правильный полный содержательный ответ с условиями и исключениями.",
    "negative": "Явный отрицательный ответ с обоснованием; отсутствие сведений не считается ответом.",
    "false_premise": (
        "Явно выявить и исправить неверную предпосылку, не выдумывать детали; простой отказ недостаточен."
    ),
    "not_in_corpus": (
        "Честно указать отсутствие необходимых сведений в корпусе без выдуманного конкретного ответа."
    ),
    "out_of_scope": "Отказаться от посторонней задачи вне тематики документного RAG без выдуманного ответа.",
    "ambiguous": "Объяснить неоднозначность и ограничения источника без выдуманной определённости.",
}

ANSWER_BEHAVIOR_CRITERIA = (
    "Оцени по answer_type и answer_rubric: 10 — поведение полностью соответствует типу, "
    "0 — не соответствует. Отрицательный ответ и исправление предпосылки не являются отказом. "
    "expected_refusal используй только для старых данных без answer_type."
)
