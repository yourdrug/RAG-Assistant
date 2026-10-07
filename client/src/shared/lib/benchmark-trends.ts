import type { BenchmarkResultDetail, BenchmarkRun } from "../api/types.ts";

export interface TrendMetric {
  key: string;
  label: string;
  description: string;
  scale: number;
  stage: "Retrieval" | "Context" | "Answer";
}

export const TREND_METRICS: TrendMetric[] = [
  {
    key: "source_hit_rate",
    label: "Source found",
    description: "Expected source found; does not guarantee the right passage.",
    scale: 1,
    stage: "Retrieval",
  },
  {
    key: "hit_rate",
    label: "Evidence hit rate",
    description:
      "Questions with a matching fragment and all required facts. Source-only cases use source matching.",
    scale: 1,
    stage: "Retrieval",
  },
  {
    key: "fragment_recall_at_k",
    label: "Fragments found",
    description: "Share of annotated fragments present in the retrieved results.",
    scale: 1,
    stage: "Retrieval",
  },
  {
    key: "retrieval_fact_coverage",
    label: "Facts retrieved",
    description: "Required fact phrases found in the retrieved text.",
    scale: 1,
    stage: "Retrieval",
  },
  {
    key: "context_fragment_recall",
    label: "Fragments in prompt",
    description: "Annotated fragments retained in the final LLM context.",
    scale: 1,
    stage: "Context",
  },
  {
    key: "context_fact_coverage",
    label: "Facts in prompt",
    description: "Required fact phrases present in the final context.",
    scale: 1,
    stage: "Context",
  },
  {
    key: "context_condition_coverage",
    label: "Conditions in prompt",
    description: "Required condition phrases present in the final context.",
    scale: 1,
    stage: "Context",
  },
  {
    key: "context_precision",
    label: "Context precision",
    description: "Judge score for the relevance of the supplied context.",
    scale: 10,
    stage: "Context",
  },
  {
    key: "correctness",
    label: "Answer correctness",
    description: "Judge score against the reference answer.",
    scale: 10,
    stage: "Answer",
  },
  {
    key: "faithfulness",
    label: "Grounded in context",
    description: "Judge score for support by context; a wrong answer can still be grounded.",
    scale: 10,
    stage: "Answer",
  },
  {
    key: "answer_fact_coverage",
    label: "Facts in answer",
    description:
      "Required fact phrases retained in the answer; exact phrase matching can miss paraphrases.",
    scale: 1,
    stage: "Answer",
  },
  {
    key: "answer_condition_coverage",
    label: "Conditions in answer",
    description: "Required condition phrases retained in the answer.",
    scale: 1,
    stage: "Answer",
  },
];

export function metricValue(metrics: Record<string, unknown>, key: string): number | null {
  for (const name of [key, `avg_${key}`]) {
    const value = metrics[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

export function formatTrendMetric(value: number | null, metric: TrendMetric): string {
  if (value === null) return "Not measured";
  return metric.scale === 1 ? `${(value * 100).toFixed(1)}%` : `${value.toFixed(1)} / 10`;
}

export function formatMetricDelta(
  current: number | null,
  previous: number | null,
  metric: TrendMetric,
): string | null {
  if (current === null || previous === null) return null;
  const delta = (current - previous) * (metric.scale === 1 ? 100 : 1);
  return `${delta > 0 ? "+" : ""}${delta.toFixed(1)} ${metric.scale === 1 ? "pp" : "pts"}`;
}

export function preferredTrendRun(runs: BenchmarkRun[]): BenchmarkRun | undefined {
  // Callers supply newest first. Prefer actual answer evaluation to a later cheap trial.
  return runs.find((run) => run.llm_evaluated) ?? runs[0];
}

export function metricsWithQuestionCoverage(
  run: BenchmarkRun,
  detail?: BenchmarkResultDetail,
): Record<string, unknown> {
  const metrics: Record<string, unknown> = { ...run.summary_metrics };
  if (detail?.id !== run.id) return metrics;
  for (const key of ["answer_fact_coverage", "answer_condition_coverage"]) {
    if (metricValue(metrics, key) !== null) continue;
    const values = (detail.per_question_results ?? []).flatMap((question) => {
      const value = metricValue(question.evidence_metrics ?? {}, key);
      return value === null ? [] : [value];
    });
    if (values.length) {
      metrics[key] = values.reduce((sum, value) => sum + value, 0) / values.length;
      metrics[`${key}_evaluated_count`] = values.length;
    }
  }
  return metrics;
}
