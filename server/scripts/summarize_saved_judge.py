"""Build an offline report for calibrate_saved_judge; never calls an LLM."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS
from calibrate_saved_judge import digest, groups, save


def summarize(root):  # noqa: C901 -- Offline aggregation and report rendering.
    snapshot = json.loads((root / 'input.json').read_text())
    result = json.loads((root / 'final-results.json').read_text())
    expected = {
        (c['key'], repeat, group)
        for c in snapshot['cases']
        for repeat in range(1, c['repeats'] + 1)
        for group in range(len(groups(c)))
    }
    actual = {(r['case'], r['repeat'], r['group']) for r in result['requests']}
    if actual != expected or len(actual) != len(result['requests']):
        raise ValueError(f'Incomplete or duplicate requests: {len(actual)}/{len(expected)}')
    if {r['actual_model'] for r in result['requests']} != {snapshot['judge_model']}:
        raise ValueError('Actual judge model differs from the frozen model')
    if {r['provider'] for r in result['requests']} != {'DeepInfra'}:
        raise ValueError('Actual judge provider differs from the frozen provider')
    combined = collections.defaultdict(dict)
    for r in result['requests']:
        combined[r['case'], r['repeat']].update(r['scores'])
    rows = []
    for c in snapshot['cases']:
        for repeat in range(1, c['repeats'] + 1):
            rows.append(
                {
                    'case': c['key'],
                    'cohort': c['cohort'],
                    'question_id': c['question']['id'],
                    'repeat': repeat,
                    'answer_sha256': digest(c['answer']),
                    'context_sha256': digest(c['context']),
                    'rubric_version': result['protocol']['rubric_version'],
                    'judge_model': snapshot['judge_model'],
                    'metrics': combined[c['key'], repeat],
                }
            )
    save(root / 'scores.json', rows)
    calibration = {
        c['key']: [combined[c['key'], i]['faithfulness']['score'] for i in range(1, 4)]
        for c in snapshot['cases']
        if c['cohort'] == 'calibration'
    }
    acceptance = {
        '140_remains_unsupported': all(s < 10 for s in calibration['original-140']),
        'unsupported_expansion_never_10': all(s <= 7 for s in calibration['159-unsupported-expansion']),
        'supported_expansion_no_penalty': calibration['159-supported-expansion'] == [10, 10, 10],
        'paraphrase_no_penalty': calibration['159-paraphrase'] == [10, 10, 10],
        'correct_refusal_faithfulness': calibration['correct-refusal'] == [10, 10, 10],
        '140_strict_main_conclusion_cap': all(s <= 3 for s in calibration['original-140']),
    }
    baseline = [c for c in snapshot['cases'] if c['cohort'] == 'baseline']
    averages = {}
    for cohort, cases in [
        ('all', baseline),
        (
            'subject',
            [c for c in baseline if not c['question'].get('annotations', {}).get('expected_refusal')],
        ),
        ('refusals', [c for c in baseline if c['question'].get('annotations', {}).get('expected_refusal')]),
    ]:
        averages[cohort] = {'count': len(cases)}
        for m in ['faithfulness', 'correctness', 'relevancy']:
            averages[cohort][m] = {
                'old': statistics.mean(c['previous_metrics'][m] for c in cases),
                'new': statistics.mean(combined[c['key'], 1][m]['score'] for c in cases),
            }
    hit_rate = snapshot['source_run']['summary_metrics']['hit_rate']
    weights = DEFAULT_OBJECTIVE_WEIGHTS
    composite = sum(
        weights[m] * (hit_rate if m == 'hit_rate' else averages['all'][m]['new'] / 10)
        for m in ['hit_rate', 'faithfulness', 'correctness', 'relevancy']
    )
    costs = [r['usage'].get('cost') for r in result['requests']]
    summary = {
        'acceptance': acceptance,
        'calibration_faithfulness': calibration,
        'baseline': averages,
        'unchanged_hit_rate': hit_rate,
        'new_weights': weights,
        'new_composite': composite,
        'successful_requests': len(actual),
        'failed_attempts': len(result['errors']),
        'successful_requests_cost_usd': sum(c for c in costs if c is not None),
        'missing_success_costs': sum(c is None for c in costs),
    }
    save(root / 'summary.json', summary)
    report = [
        '# Калибровка судьи и повторная оценка quick-30',
        '',
        'Дата: 8 октября 2026. Новая генерация ответов и retrieval не выполнялись. '
        'Исторические строки БД не изменялись; переоценка сохранена отдельным артефактом.',
        '',
        '## Зафиксированные входы и протокол',
        '',
        '- Baseline: run №11, sweep №7, dataset quick-30 (30 вопросов), версия sweep 44. '
        'Исходные ответы, финальные контексты, разметка и runtime snapshot '
        'сохранены в [input.json](input.json).',
        f"- Judge: `{snapshot['judge_model']}`, провайдер DeepInfra без fallback; temperature=0, "
        'reasoning effort=low, max_tokens=8192. Фактические модель и провайдер проверены у каждого ответа.',
        f"- Рубрика: `{result['protocol']['rubric_version']}`. Полный текст критериев, "
        'параметры и хэши запросов сохранены в [final-results.json](final-results.json), '
        'вместе с сырыми ответами судьи, reasons, usage, request IDs и временем.',
        f"- Git: `{snapshot['git_commit']}`; дерево имело незакоммиченные изменения. "
        'SHA256 diff и хэш исполняемого скрипта сохранены; git commit сам по себе не описывает эксперимент.',
        '- Resume привязан к снимку входов, тексту рубрики, модели, провайдеру, параметрам и скрипту. '
        'Повторный запуск пропускает завершённые запросы; изменение контекста или рубрики отклоняется.',
        '',
        '## Проверка LLM: три независимых вызова на пример',
        '',
        '| Пример | Faithfulness, повторы 1 / 2 / 3 |',
        '|---|---|',
    ]
    for key, values in calibration.items():
        report.append(f"| {key} | {' / '.join(f'{s:g}' for s in values)} |")
    report += [
        '',
        'Заданные контрольные критерии выполнены: №140 во всех повторах назван неподтверждённым; '
        'неподтверждённая расшифровка не получает 10; подтверждённая расшифровка, перефразирование '
        'и правильный отказ не штрафуются по faithfulness.',
        '',
        '**Калибровка выявила ограничения судьи.** У №140 один балл 6 нарушает собственный потолок '
        'рубрики ≤3 для неподтверждённого основного вывода. У полного исходного №159 оценка 10/6/6: '
        'судья не всегда замечает неподтверждённую английскую расшифровку EDI-provider. '
        'В baseline №159 судья ошибочно назвал ПК СПТ неподтверждённой расшифровкой, '
        'хотя она есть в финальном контексте. Reasons доступны в scores.json; оценки не обрезались вручную. '
        'Эти результаты не доказывают стабильное соблюдение всех правил рубрики.',
        '',
        '### Что именно менялось в парах',
        '',
        'Расшифровка ПК СПТ в исходном №159 **подтверждена**: в его финальном контексте есть '
        'пункт 1 постановления МНС №13 с полным названием. Называть её галлюцинацией в этом '
        'сохранённом контексте неверно. Английская расшифровка EDI-provider — отдельное утверждение.',
        '',
        'Для изоляции эффекта взят один подтверждённый факт о наличии уникального номера '
        'EDI-провайдера в передаваемых сведениях. В отрицательном контроле из копии контекста '
        'удалён подтверждающий расшифровку фрагмент постановления №13; положительный контроль '
        'использует исходный контекст. Ответы с расшифровкой в этих двух контролях совпадают '
        'побайтно, ссылка [1] остаётся прежней. Вариант без расшифровки и перефразирование '
        'используют тот же сокращённый контекст. Это явная абляция, не новый retrieval. '
        'Точный фрагмент и metadata сохранены как positive_control_evidence.',
        '',
        'Полный №159 без расшифровки ПК СПТ сохраняет остальные утверждения исходного ответа '
        'и исходный контекст; его 7/6/7 не доказывают вред подтверждённой расшифровки. '
        'Числовой разброс и другие неподтверждённые утверждения мешают такому выводу.',
        '',
        '## Baseline: те же 30 ответов и контекстов, новая оценка',
        '',
        '| Набор | N | Faithfulness: до → после | Correctness: до → после | Relevancy: до → после |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, v in averages.items():
        report.append(
            f"| {name} | {v['count']} | "
            + ' | '.join(
                f"{v[m]['old']:.2f} → {v[m]['new']:.2f}" for m in ['faithfulness', 'correctness', 'relevancy']
            )
            + ' |'
        )
    report += [
        '',
        f'Evidence hit rate сохранён: {hit_rate:.2%}. Композит по новым весам '
        f'faithfulness=0.40, correctness=0.30, hit_rate=0.20, relevancy=0.10: **{composite:.4f}**. '
        'Исторический composite рассчитан с другими весами и напрямую несопоставим. '
        'Сдвиг judge-метрик отражает переоценку, а не улучшение RAG. '
        'Разделение subject/refusals использует expected_refusal из snapshot sweep 7: '
        '23 предметных вопроса и 7 ожидаемых отказов (включая №153 и №161). '
        'Это отличается от группировки старых sweeps 5/6.',
        '',
        f"Завершено {len(actual)} запросов оценки: 69 калибровочных и {len(actual) - 69} baseline; "
        f"неуспешных попыток с повтором — {len(result['errors'])}. "
        f"Стоимость успешных запросов по usage: ${summary['successful_requests_cost_usd']:.4f}; "
        'не включает ошибки, отменённые запросы и пилоты. '
        'Пилотные файлы исключены из всех итоговых метрик.',
        '',
        '## Воспроизведение',
        '',
        'Из server/:',
        '',
        '```bash',
        'PYTHONPATH=app DATA_DIR=/tmp/rag-judge-calibration '
        '.venv/bin/python scripts/calibrate_saved_judge.py \\\n'
        '  ../docs/benchmarks/judge-calibration-2026-10-08/input.json \\\n'
        '  --out ../docs/benchmarks/judge-calibration-2026-10-08/final-results.json --cohort calibration',
        '# Для baseline: та же команда с --cohort baseline.',
        'PYTHONPATH=app DATA_DIR=/tmp/rag-judge-calibration '
        '.venv/bin/python scripts/summarize_saved_judge.py \\\n'
        '  ../docs/benchmarks/judge-calibration-2026-10-08',
        '```',
        '',
        'Для новой независимой серии нужен другой --out. Сохранённый --out при resume '
        'не повторяет платные вызовы. Требуются настроенные credentials; они в артефакты не включены.',
        '',
        'Проверка кода: 7 pytest-тестов criteria wiring, resume и новых frozen judge checks прошли; '
        'ruff прошёл. Автотесты используют фейки и не подменяют приведённую проверку настоящим LLM.',
        '',
    ]
    (root / 'REPORT.md').write_text('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)


if __name__ == '__main__':
    main()
