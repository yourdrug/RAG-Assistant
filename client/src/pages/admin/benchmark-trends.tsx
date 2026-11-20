import { History, RefreshCw } from "lucide-react";
import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useBenchmarkDatasets, useBenchmarkRuns } from "@/shared/api/hooks";
import {
  useBenchmarkReviewQuestion,
  useBenchmarkReviewQuestions,
} from "@/shared/api/hooks/use-benchmark";
import type { BenchmarkReviewQuestion, BenchmarkRun } from "@/shared/api/types";
import {
  formatMetricDelta,
  formatTrendMetric,
  metricValue,
  preferredTrendRun,
  TREND_METRICS,
  trendDomain,
} from "@/shared/lib/benchmark-trends";
import { formatDateTime } from "@/shared/lib/date-time";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

const RUN_LIMIT = 100;
const DIAGNOSTIC_PAGE_SIZE = 25;

function configLabel(run: BenchmarkRun): string {
  return (
    Object.entries(run.config_json)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(" · ") || "Default config"
  );
}

function runLabel(run: BenchmarkRun): string {
  return `#${run.id} · ${run.llm_evaluated ? "Answers evaluated" : "Retrieval only"} · ${run.creation_date ? formatDateTime(run.creation_date) : "Unknown date"}`;
}

function TrendQuestionAnswer({
  runId,
  question,
}: {
  runId: number;
  question: BenchmarkReviewQuestion;
}) {
  const detail = useBenchmarkReviewQuestion(runId, question.position, question.key);
  if (detail.isError) {
    return <QueryError query={detail} fallback="Could not load this answer." />;
  }
  if (!detail.data)
    return (
      <p className="mt-3 text-sm" role="status">
        Loading answer…
      </p>
    );

  return (
    <div className="mt-3 space-y-2 whitespace-pre-wrap break-words text-sm">
      <p>
        <strong>Answer</strong> {detail.data.answer || "Not recorded"}
      </p>
      <p>
        <strong>Reference</strong> {detail.data.expected_answer || "Not recorded"}
      </p>
    </div>
  );
}

function TrendQuestion({ runId, question }: { runId: number; question: BenchmarkReviewQuestion }) {
  const [open, setOpen] = useState(false);
  return (
    <TableRow>
      <TableCell className="min-w-64 max-w-xl">
        <details onToggle={(event) => setOpen(event.currentTarget.open)}>
          <summary className="cursor-pointer whitespace-normal text-sm">
            #{question.position} · {question.question}
          </summary>
          {open && <TrendQuestionAnswer runId={runId} question={question} />}
        </details>
      </TableCell>
      <TableCell>{question.correctness} / 10</TableCell>
      <TableCell>
        {question.hit_rate == null ? "Not measured" : question.hit_rate ? "Yes" : "No"}
      </TableCell>
      <TableCell>
        {formatTrendMetric(
          metricValue(question.evidence_metrics, "context_fact_coverage"),
          TREND_METRICS[5],
        )}
      </TableCell>
    </TableRow>
  );
}

export function BenchmarkTrendsTab() {
  const [datasetChoice, setDatasetChoice] = useState("");
  const [evaluation, setEvaluation] = useState("answers");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [baselineId, setBaselineId] = useState<number | null>(null);
  const [metricKey, setMetricKey] = useState("hit_rate");
  const [diagnosticPage, setDiagnosticPage] = useState(0);
  const { data: datasets } = useBenchmarkDatasets();
  const { data, isLoading, isError, refetch } = useBenchmarkRuns({
    dataset: datasetChoice || undefined,
    sort_by: "creation_date",
    sort_order: "desc",
    limit: RUN_LIMIT,
  });
  const dataset = datasetChoice || data?.runs[0]?.dataset || "";
  const datasetRuns = (data?.runs ?? []).filter((run) => run.dataset === dataset);
  const runs = datasetRuns.filter((run) => evaluation === "all" || run.llm_evaluated);
  const selected = runs.find((run) => run.id === selectedId) ?? preferredTrendRun(runs);
  const baseline = runs.find((run) => run.id === baselineId && run.id !== selected?.id);
  const diagnostics = useBenchmarkReviewQuestions(selected?.llm_evaluated ? selected.id : null, {
    score: 7,
    comparison: "lt",
    sort: "score_asc",
    limit: DIAGNOSTIC_PAGE_SIZE,
    offset: diagnosticPage * DIAGNOSTIC_PAGE_SIZE,
  });
  const options = [
    ...new Set([
      ...(datasets ?? []),
      ...(data?.runs ?? []).map((run) => run.dataset),
      ...(dataset ? [dataset] : []),
    ]),
  ].sort();
  const metric = TREND_METRICS.find((item) => item.key === metricKey) ?? TREND_METRICS[1];
  const selectedMetrics = selected?.summary_metrics ?? {};
  const baselineMetrics = baseline?.summary_metrics ?? {};
  const chartData = [...runs].reverse().map((run) => ({
    run: `#${run.id}`,
    date: run.creation_date ? formatDateTime(run.creation_date) : "Unknown date",
    value: metricValue(run.summary_metrics, metric.key),
  }));
  const plottedCount = chartData.filter((point) => point.value !== null).length;
  const weakAnswers = diagnostics.data?.questions ?? [];
  const retrievedFragments = metricValue(selectedMetrics, "fragment_recall_at_k");
  const contextFragments = metricValue(selectedMetrics, "context_fragment_recall");

  function resetSelection() {
    setSelectedId(null);
    setBaselineId(null);
    setDiagnosticPage(0);
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1 min-w-40">
          <Label htmlFor="trend-dataset">Question dataset</Label>
          <Select
            value={dataset}
            onValueChange={(value) => {
              setDatasetChoice(value);
              resetSelection();
            }}
          >
            <SelectTrigger id="trend-dataset">
              <SelectValue placeholder="Select dataset" />
            </SelectTrigger>
            <SelectContent>
              {options.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1 min-w-48">
          <Label htmlFor="trend-evaluation">Show runs</Label>
          <Select
            value={evaluation}
            onValueChange={(value) => {
              setEvaluation(value);
              resetSelection();
            }}
          >
            <SelectTrigger id="trend-evaluation">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="answers">With answer evaluation</SelectItem>
              <SelectItem value="all">All configurations</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void refetch();
            if (selected?.llm_evaluated) void diagnostics.refetch();
          }}
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {isLoading ? (
        <Skeleton className="h-48 w-full" />
      ) : isError ? (
        <p role="alert" className="text-destructive">
          Could not load benchmark runs. Try refreshing.
        </p>
      ) : !selected ? (
        <div className="py-10 text-center text-muted-foreground">
          <History className="h-10 w-10 mx-auto mb-3 opacity-40" />
          <p>
            {datasetRuns.length
              ? "No answers have been evaluated for this dataset. Choose All configurations to inspect retrieval."
              : "No benchmark runs for this dataset yet."}
          </p>
        </div>
      ) : (
        <>
          <p className="text-sm text-muted-foreground">
            {runs.length} shown · {datasetRuns.filter((run) => run.llm_evaluated).length} with
            answer evaluation · {datasetRuns.filter((run) => !run.llm_evaluated).length} retrieval
            only.
            {data &&
              data.total > RUN_LIMIT &&
              ` Showing the latest ${RUN_LIMIT} saved runs; older runs are outside this view.`}{" "}
            Trials from one sweep compare configurations, not improvements over time.
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-1 min-w-0">
              <Label htmlFor="trend-run">Inspect run</Label>
              <Select
                value={String(selected.id)}
                onValueChange={(value) => {
                  setSelectedId(Number(value));
                  setDiagnosticPage(0);
                }}
              >
                <SelectTrigger id="trend-run">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {runs.map((run) => (
                    <SelectItem key={run.id} value={String(run.id)}>
                      {runLabel(run)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1 min-w-0">
              <Label htmlFor="trend-baseline">Compare with</Label>
              <Select
                value={baseline ? String(baseline.id) : "none"}
                onValueChange={(value) => setBaselineId(value === "none" ? null : Number(value))}
              >
                <SelectTrigger id="trend-baseline">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No comparison selected</SelectItem>
                  {runs
                    .filter((run) => run.id !== selected.id)
                    .map((run) => (
                      <SelectItem key={run.id} value={String(run.id)}>
                        {runLabel(run)}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="rounded-lg border p-4 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold">Run #{selected.id}</span>
              <Badge variant="secondary">{selected.dataset}</Badge>
              <Badge variant={selected.llm_evaluated ? "success" : "secondary"}>
                {selected.llm_evaluated ? "Answers evaluated" : "Retrieval only"}
              </Badge>
              {selected.sweep_id != null && (
                <span className="text-sm text-muted-foreground">Sweep #{selected.sweep_id}</span>
              )}
            </div>
            <p className="text-xs font-mono break-words">{configLabel(selected)}</p>
            {!selected.llm_evaluated && (
              <p className="text-sm text-muted-foreground">
                Context and answer quality were not measured. Missing scores are not zero.
              </p>
            )}
            {baseline && (
              <p className="text-sm text-muted-foreground">
                Deltas are against run #{baseline.id}: {configLabel(baseline)}. Compare the same
                question set and judge settings before calling a change a regression.
              </p>
            )}
            {baseline && baseline.llm_evaluated !== selected.llm_evaluated && (
              <p className="text-sm text-amber-600">
                These runs cover different evaluation stages. Only metrics measured in both can be
                compared.
              </p>
            )}
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            {(["Retrieval", "Context", "Answer"] as const).map((stage, index) => (
              <Card key={stage}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">
                    {index + 1}. {stage}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {TREND_METRICS.filter((item) => item.stage === stage).map((item) => {
                    const value = metricValue(selectedMetrics, item.key);
                    const delta = baseline
                      ? formatMetricDelta(value, metricValue(baselineMetrics, item.key), item)
                      : null;
                    const count = selectedMetrics[`${item.key}_evaluated_count`];
                    return (
                      <div key={item.key}>
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="text-sm font-medium">{item.label}</span>
                          <span className="font-mono text-sm shrink-0">
                            {formatTrendMetric(value, item)}
                          </span>
                        </div>
                        {value !== null && (
                          <div className="h-1.5 mt-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{
                                width: `${Math.max(0, Math.min(100, (value / item.scale) * 100))}%`,
                              }}
                            />
                          </div>
                        )}
                        <p className="text-xs text-muted-foreground mt-1">
                          {item.description}
                          {typeof count === "number" ? ` Measured: ${count} questions.` : ""}
                        </p>
                        {delta && (
                          <p className="text-xs font-mono mt-1">
                            {delta} vs #{baseline?.id}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Where to investigate</CardTitle>
              <CardDescription>
                These are diagnostic signals, not an automatic root-cause verdict. Higher metric
                values are better.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>
                <strong>Retrieval:</strong> compare Source found with Evidence hit rate. A document
                match can hide missing passages, facts or conditions. Check candidate recall,
                ranking and chunk boundaries.
              </p>
              <p>
                <strong>Context:</strong> compare Fragments found with Fragments in prompt. Check
                grouping, deduplication and the context token budget.
              </p>
              {retrievedFragments !== null &&
                contextFragments !== null &&
                retrievedFragments > contextFragments && (
                  <p className="rounded-md bg-muted p-3">
                    In this run, fragment recall falls from {(retrievedFragments * 100).toFixed(1)}%
                    after retrieval to {(contextFragments * 100).toFixed(1)}% in the prompt (
                    {((retrievedFragments - contextFragments) * 100).toFixed(1)} pp). Inspect the
                    lost passages before changing the model.
                  </p>
                )}
              <p>
                <strong>Answer:</strong> if the prompt contains the required facts but correctness
                is low, inspect generation, table interpretation and version handling. Phrase
                coverage and judge scores can disagree; verify the examples below.
              </p>
              {selected.llm_evaluated && diagnostics.isLoading && (
                <p className="text-muted-foreground" role="status">
                  Loading question diagnostics…
                </p>
              )}
              {diagnostics.isError && (
                <QueryError
                  query={diagnostics}
                  fallback="Question diagnostics could not be loaded."
                />
              )}
              {diagnostics.data && (
                <p className="text-muted-foreground">
                  {diagnostics.data.matched} of {diagnostics.data.scored} scored answers have
                  correctness below 7 / 10.
                </p>
              )}
              {weakAnswers.length > 0 && (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Question</TableHead>
                        <TableHead>Correctness</TableHead>
                        <TableHead>Evidence hit</TableHead>
                        <TableHead>Facts in prompt</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {weakAnswers.map((question) => (
                        <TrendQuestion
                          key={`${selected.id}:${question.position}`}
                          runId={selected.id}
                          question={question}
                        />
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
              {(diagnosticPage > 0 || (diagnostics.data?.matched ?? 0) > DIAGNOSTIC_PAGE_SIZE) && (
                <nav className="flex gap-2" aria-label="Question diagnostics pagination">
                  <Button
                    variant="outline"
                    disabled={diagnosticPage === 0}
                    onClick={() => setDiagnosticPage(diagnosticPage - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    disabled={
                      !diagnostics.data ||
                      (diagnosticPage + 1) * DIAGNOSTIC_PAGE_SIZE >= diagnostics.data.matched
                    }
                    onClick={() => setDiagnosticPage(diagnosticPage + 1)}
                  >
                    Next
                  </Button>
                </nav>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Metric across saved runs</CardTitle>
              <CardDescription>
                Oldest to newest. Configuration trials from the same sweep are separate points.
                Missing measurements are gaps.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={metricKey} onValueChange={setMetricKey}>
                <SelectTrigger className="max-w-xs" aria-label="Chart metric">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TREND_METRICS.map((item) => (
                    <SelectItem key={item.key} value={item.key}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {plottedCount < 2 && (
                <p className="text-sm text-muted-foreground mt-3">
                  {plottedCount === 0
                    ? "This metric was not measured in the shown runs."
                    : "Only one measured run. Run another comparable benchmark to see a trend."}
                </p>
              )}
              {plottedCount > 0 && (
                <div className="mt-4">
                  <p className="text-xs text-muted-foreground">Y-axis zooms to measured values.</p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={chartData}
                        margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                        <XAxis dataKey="run" tick={{ fontSize: 12 }} />
                        <YAxis
                          domain={trendDomain(
                            chartData.map((point) => point.value),
                            metric.scale,
                          )}
                          allowDataOverflow
                          tickCount={5}
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value: number) =>
                            metric.scale === 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)
                          }
                        />
                        <Tooltip
                          content={({ active, payload }) =>
                            active && payload?.length ? (
                              <div className="rounded-md border bg-popover p-3 text-xs text-popover-foreground shadow-md">
                                <p>
                                  {payload[0].payload.run} · {payload[0].payload.date}
                                </p>
                                <p>
                                  {metric.label}:{" "}
                                  {formatTrendMetric(payload[0].payload.value, metric)}
                                </p>
                              </div>
                            ) : null
                          }
                        />
                        <Line
                          type="linear"
                          dataKey="value"
                          stroke="var(--color-primary)"
                          strokeWidth={2}
                          dot={{ r: 4 }}
                          connectNulls={false}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
              <div className="overflow-x-auto mt-4 max-h-80">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Run / date</TableHead>
                      <TableHead>Configuration</TableHead>
                      <TableHead>Evaluation</TableHead>
                      <TableHead>{metric.label}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {runs.map((run) => (
                      <TableRow key={run.id} className={run.id === selected.id ? "bg-muted" : ""}>
                        <TableCell>
                          <Button variant="link" size="sm" onClick={() => setSelectedId(run.id)}>
                            Inspect #{run.id}
                          </Button>
                          <p className="text-xs text-muted-foreground">
                            {run.creation_date ? formatDateTime(run.creation_date) : "Unknown date"}
                          </p>
                        </TableCell>
                        <TableCell className="text-xs font-mono min-w-48">
                          {configLabel(run)}
                        </TableCell>
                        <TableCell className="text-xs">
                          {run.llm_evaluated ? "Answers evaluated" : "Retrieval only"}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {formatTrendMetric(
                            metricValue(
                              run.id === selected.id ? selectedMetrics : run.summary_metrics,
                              metric.key,
                            ),
                            metric,
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
