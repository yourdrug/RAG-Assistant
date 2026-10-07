import assert from "node:assert/strict";
import { test } from "node:test";
import type { BenchmarkResultDetail, BenchmarkRun } from "../src/shared/api/types.ts";
import {
  formatMetricDelta,
  formatTrendMetric,
  metricsWithQuestionCoverage,
  metricValue,
  preferredTrendRun,
  TREND_METRICS,
} from "../src/shared/lib/benchmark-trends.ts";

function run(id: number, evaluated = false): BenchmarkRun {
  return {
    id,
    llm_evaluated: evaluated,
    config_json: {},
    summary_metrics: {},
    dataset: "test",
    duration_sec: 0,
  };
}

test("later cheap trials do not replace the latest evaluated answer run", () => {
  assert.equal(preferredTrendRun([run(9), run(8), run(2, true), run(1, true)])?.id, 2);
  assert.equal(preferredTrendRun([run(9), run(8)])?.id, 9);
  assert.equal(preferredTrendRun([]), undefined);
});

test("unmeasured metrics stay missing, zero is a measurement, and legacy averages work", () => {
  assert.equal(metricValue({ hit_rate: 0, avg_hit_rate: 1 }, "hit_rate"), 0);
  assert.equal(metricValue({ faithfulness: null, avg_faithfulness: 8.2 }, "faithfulness"), 8.2);
  for (const value of [null, undefined, "0.5", Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(metricValue({ hit_rate: value }, "hit_rate"), null);
  }
});

test("coverage uses measured questions only, includes zeros and cannot leak across selected runs", () => {
  const selected = run(1, true);
  const detail: BenchmarkResultDetail = {
    id: 1,
    summary: selected,
    per_question_results: [
      { id: 1, question: "a", answer: "", evidence_metrics: { answer_fact_coverage: 0 } },
      { id: 2, question: "b", answer: "", evidence_metrics: { answer_fact_coverage: 1 } },
      { id: 3, question: "c", answer: "", evidence_metrics: { answer_fact_coverage: null } },
    ],
  };
  const metrics = metricsWithQuestionCoverage(selected, detail);
  assert.equal(metricValue(metrics, "answer_fact_coverage"), 0.5);
  assert.equal(metrics.answer_fact_coverage_evaluated_count, 2);
  assert.equal(metricValue(metrics, "answer_condition_coverage"), null);
  assert.equal(
    metricValue(metricsWithQuestionCoverage(run(2), detail), "answer_fact_coverage"),
    null,
  );
  selected.summary_metrics.avg_answer_fact_coverage = 0.75;
  assert.equal(
    metricValue(metricsWithQuestionCoverage(selected, detail), "answer_fact_coverage"),
    0.75,
  );
});

test("percentages and judge scores show their own units and comparable deltas", () => {
  const hit = TREND_METRICS.find((metric) => metric.key === "hit_rate")!;
  const correctness = TREND_METRICS.find((metric) => metric.key === "correctness")!;
  assert.equal(formatTrendMetric(0.68, hit), "68.0%");
  assert.equal(formatTrendMetric(7, correctness), "7.0 / 10");
  assert.equal(formatTrendMetric(null, hit), "Not measured");
  assert.equal(formatMetricDelta(0.68, 0.56, hit), "+12.0 pp");
  assert.equal(formatMetricDelta(7, 8.2, correctness), "-1.2 pts");
  assert.equal(formatMetricDelta(null, 0, hit), null);
});
