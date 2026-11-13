# Обновление обработчика benchmark для RAG (v2)

## Что изменилось в датасете

Исходный `benchmark-questions (3).json` сохранён без изменений. `benchmark-questions-v2.json` — **по-прежнему JSON-массив объектов**, совместимый с прежним циклом чтения вопросов.

- 202 записи всего: **172 `main` + 30 `quick-30`**. Быстрый набор содержит 30 повторов основных кейсов. Из 172 основных кейсов **171 активен**; один пограничный кейс (`rag-0023`) временно отключён для ручной проверки.
- 40 новых вариантов вопросов на более естественном языке; у каждого есть `variant_of`, указывающий на исходную задачу.
- `case_id` — устойчивый идентификатор семантически одинакового исходного вопроса. Повтор из `quick-30` имеет **тот же** `case_id`, что и `main`; новые перефразировки имеют собственный `case_id` и `variant_of`.
- Доработаны эталоны для кейсов с ложными предпосылками и отрицательными ответами. Дополнены доказательные фрагменты для ряда таблиц и перечней.
- Сохранены старые поля: `question`, `expected_answer`, `source_hint`, `annotations.required_facts`, `annotations.expected_fragments`, `annotations.expected_refusal`, `tags`, `dataset`, `is_active` (где были), чтобы минимизировать миграцию.
- Добавлены: `annotations.answer_type`, `annotations.expected_behavior`, `annotations.source_files`, `annotations.evidence_text_check`, для отдельных записей `annotations.gold_evidence`, а также `annotations.review_required`/`review_reason`.

Важное ограничение: сверка эталонных фрагментов с DOCX — **текстовая**, а не полная юридическая экспертиза всех утверждений. `evidence_text_check="matched"` означает, что текст фрагмента найден в файле, **но не гарантирует**, что он достаточно полный и юридически верно интерпретирован. `gold_evidence` добавлена точечно, а не ко всем вопросам.

## Классы ответов — используйте их вместо одного `expected_refusal`

| `annotations.answer_type` | `expected_behavior` | Как считать успешным | `expected_refusal` (совместимость) |
| --- | --- | --- | --- |
| `factual` | `answer` | Правильный содержательный ответ, подтверждённый источником | `false` |
| `negative` | `answer` | Явное **нет** с правильным объяснением на основе источника, а не «информация не найдена» | `false` |
| `false_premise` | `correct_premise` | Обнаружить неверную предпосылку (несуществующий пункт, форму, поле), объяснить её и не выдумать детали | `false` |
| `not_in_corpus` | `abstain` | Не выдавать придуманный конкретный ответ, честно указать на отсутствие необходимых сведений в предоставленном корпусе | `true` |
| `out_of_scope` | `abstain` | В строгом документном RAG не отвечать на постороннюю задачу как будто она подтверждена базой | `true` |
| `ambiguous` | `clarify_ambiguity` | Объяснить неоднозначность и ограничения источника; **в текущей версии этот кейс неактивен** | `false` |

**Критически важно:** `false_premise` — **не** то же самое, что `not_in_corpus`. Например, на вопрос о несуществующем поле нельзя считать правильным ответом одно лишь «не нашёл сведений»; правильный ответ должен указать, что названного поля нет в приложении. Аналогично `negative` нельзя засчитывать как удачный отказ.

## Конкретные изменения по сравнению с v1

- `rag-0021` (исходный вопрос №51): «Изменилась ли периодичность передачи сведений ... постановлением 2024 года?» → `negative`, ожидание: **нет, этим актом не изменялась**; `expected_refusal=false`.
- `rag-0026` (№56): «Применяется ли ТН-1?» → `false_premise`, т.к. в источнике формы **ТТН-1 и ТН-2**, а не ТН-1.
- `rag-0028` (№58): «Код для бумажной основы мехового изделия?» → `false_premise`, такая основа в указанном перечне не предусмотрена.
- `rag-0029` (№59): «Сколько экземпляров ТТН-2?» → `false_premise`, такой формы нет.
- `rag-0031` (№61): «Есть ли отдельный код лизинга?» → `negative`, отдельного специального кода не выделено.
- `rag-0007`, `rag-0008`, `rag-0010`, `rag-0038`: вопросы про несуществующие структурные элементы/ссылки → `false_premise`.
- `rag-0023` (№53): срок рассрочки **ровно один год**, конфликт возможных толкований выражения «до одного года». Помечен `ambiguous`, `review_required=true`, **`is_active=false`** до утверждения человеком.

## Минимальные обязательные правки обработчика

1. **Загрузка:** поддержать `case_id`, `annotations.answer_type`, `annotations.expected_behavior`; если `answer_type` отсутствует (старый JSON), применять fallback `expected_refusal ? not_in_corpus : factual`. Фильтровать `is_active=false` **до запуска тестов**.
2. **Диспетчер оценивания:** выбирать рубрику по `answer_type`, а не по тегу `not_found` или одному булеву `expected_refusal`.
3. **Оценка ответа:** семантически проверять `expected_answer`, корректность условий/исключений и обоснованность по источникам. `required_facts` — вспомогательная подсказка, **нельзя** считать совпадения подстрок единственным показателем правильности (особенно для кодов `00`, `202`, `230`, артикулов, чисел и отрицаний).
4. **Оценка retrieval отдельно:** проверять попадание релевантного **документа и фрагмента** в top-k, а не только итоговый ответ. Дополнительно использовать `gold_evidence` (если есть); `expected_fragments` остаются для совместимости. Извлечение только названия документа не равно найденному ответу.
5. **Агрегация:** считать качество отдельно по `answer_type`, `question_style` (`natural`/`reference_guided`) и по `dataset`. Для общего отчёта **не суммировать** `main` и `quick-30` как независимые тесты. При объединении наборов дедуплицировать по `case_id`.
6. **Изоляция теста:** в запрос к RAG передавать **только `question`** (и обычные runtime-параметры), никогда не отправлять `expected_answer`, `source_hint`, `gold_evidence` или метки как подсказку модели. Эти поля нужны только оценщику.
7. **Наблюдаемость:** логировать `case_id`, `answer_type`, `dataset`, `variant_of`, реальный ответ, использованные source/chunk IDs, разбор judge, задержку и стоимость. Сохранять полный отчёт отдельно от эталонов.

### Минимальный пример диспетчеризации на Python

```python
import json
from collections import defaultdict
from pathlib import Path

DATASET = Path("benchmark-questions-v2.json")
cases = json.loads(DATASET.read_text(encoding="utf-8"))


def select_cases(dataset="main"):
    return [
        c for c in cases
        if c.get("dataset") == dataset and c.get("is_active", True)
    ]


def answer_type(case):
    ann = case.get("annotations", {})
    return ann.get("answer_type") or (
        "not_in_corpus" if ann.get("expected_refusal") else "factual"
    )


def grade(case, rag_result, semantic_judge):
    """semantic_judge принимает рубрику и возвращает structured result.

    Не реализуйте его через один поиск слов в строке: нужны смысловая
    проверка и сверка с источниками. Ниже только контракт интеграции.
    """
    t = answer_type(case)
    rubric = {
        "factual": "correct_and_grounded",
        "negative": "grounded_negative_answer",
        "false_premise": "explicit_premise_correction",
        "not_in_corpus": "abstain_without_hallucination",
        "out_of_scope": "domain_refusal_without_hallucination",
        "ambiguous": "acknowledge_ambiguity_without_invention",
    }[t]
    return semantic_judge(
        rubric=rubric,
        question=case["question"],
        answer=rag_result["answer"],
        reference=case["expected_answer"],
        evidence=case["annotations"].get("gold_evidence")
                 or case["annotations"].get("expected_fragments", []),
        retrieved=rag_result.get("retrieved_chunks", []),
        sources=case["annotations"].get("source_files", []),
    )


def report_by_type(results):
    # results: iterable {case_id, answer_type, passed}
    buckets = defaultdict(list)
    for result in results:
        buckets[result["answer_type"]].append(bool(result["passed"]))
    return {
        kind: {"passed": sum(vals), "total": len(vals), "accuracy": sum(vals) / len(vals)}
        for kind, vals in buckets.items() if vals
    }
```

### Как судить ответы

Рекомендуемый формат результата judge: `{ "passed": bool, "answer_correct": bool, "grounded": bool, "abstained": bool, "premise_corrected": bool, "explanation": str }`. Не путать «ответ содержит *нет*» и «отказывает отвечать»: решает смысл, подтверждённый документами.

Для `not_in_corpus` и `out_of_scope` метрика retrieval hit по `gold_evidence` неприменима. Для `false_premise` и `negative` поиск релевантного источника оценивается, если есть опорный текст; отсутствие отдельного поля/кода в классификаторе иногда требует проверки всей релевантной таблицы, а не одной строки.

## Проверка перед заменой старого датасета

- Прогоните только `quick-30` и сравните результаты с v1: некоторые `expected_refusal` специально изменились, поэтому сравнение raw accuracy между версиями **некорректно** без повторного скоринга старых ответов.
- Затем прогоните `main`, исключив `is_active=false`; основные метрики считайте отдельно для `natural` и `reference_guided`.
- Включите дифф ошибок: *retrieval miss*, *incorrect answer*, *unsupported answer*, *wrong refusal*, *failed to detect false premise*.
- Для юридически критичного использования проверьте отключённый `rag-0023` и выполните выборочную экспертную проверку эталонов.

## Ограничения версии

Здесь **не запускался твой RAG** и **не проводилась независимая экспертиза всех 172 эталонных вопросов**. Датасет доработан по предоставленным нормативным документам и проверен на формат, согласованность меток и текстовые доказательные ссылки. Для строгого автоматического `Recall@K` в будущем лучше добавить стабильные chunk IDs / paragraph IDs к эталонной разметке; сейчас доступны текстовые опорные фрагменты и имена файлов.
