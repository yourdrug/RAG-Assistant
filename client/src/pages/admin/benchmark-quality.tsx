import { ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import toast from "react-hot-toast";
import { useBenchmarkDatasets, useBenchmarkRuns, useCurrentUser } from "@/shared/api/hooks";
import {
  type ReviewQuestionParams,
  useBenchmarkReviewQuestion,
  useBenchmarkReviewQuestions,
} from "@/shared/api/hooks/use-benchmark";
import {
  type AcceptanceThresholds,
  type AgreementMetric,
  DIAGNOSIS_CLASSES,
  type DiagnosisTransition,
  type HumanLabel,
  type LabelMetric,
  useDatasetComposition,
  useDatasetPolicy,
  useDeleteLabel,
  useJudgeAgreement,
  useJudgeCalibrations,
  useMeasureNoise,
  useRunLabels,
  useSaveCalibration,
  useSaveLabel,
  useUpdateThresholds,
} from "@/shared/api/hooks/use-benchmark-quality";
import type { BenchmarkQuestionResult, BenchmarkReviewQuestion } from "@/shared/api/types";
import { getErrorMessage } from "@/shared/lib/errors";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

type QuestionRow = BenchmarkQuestionResult & {
  outcome?: string | null;
  diagnosis?: string | null;
  run?: number;
  case_id?: string | null;
  history?: { role: string; content: string }[] | null;
  as_of_date?: string | null;
  evidence?: {
    context?: string | null;
    attempts?: { checks?: GateCheck[] | null }[] | null;
  } | null;
};

type GateCheck = {
  stage?: string;
  context?: string | null;
  result?: { is_relevant?: boolean; reason?: string } | null;
};

/** Last relevance-gate check: when the gate refused, it holds the only recorded context. */
function lastGateCheck(row: QuestionRow): GateCheck | null {
  const checks = (row.evidence?.attempts ?? []).flatMap((attempt) => attempt.checks ?? []);
  const relevance = checks.filter((check) => check.stage === "relevance" && check.context);
  return relevance.at(-1) ?? null;
}

const DIAGNOSIS_INFO: Record<
  string,
  {
    label: string;
    hint: string;
    variant: "success" | "warning" | "destructive" | "secondary";
  }
> = {
  ok: { label: "OK", hint: "All stages passed", variant: "success" },
  retrieval_miss: {
    label: "Retrieval miss",
    hint: "Expected fragments were not found by search",
    variant: "destructive",
  },
  false_refusal: {
    label: "False refusal",
    hint: "Answerable question was refused",
    variant: "destructive",
  },
  context_loss: {
    label: "Context loss",
    hint: "Found by search, lost before the prompt",
    variant: "warning",
  },
  generation_error: {
    label: "Generation",
    hint: "Full context, but correctness or faithfulness < 7",
    variant: "warning",
  },
  false_answer: {
    label: "False answer",
    hint: "Expected a refusal, got an answer",
    variant: "destructive",
  },
  unlabeled: {
    label: "Unlabeled",
    hint: "No annotations or scores to decide",
    variant: "secondary",
  },
};

export function DiagnosisBadge({ value }: { value: string | null | undefined }) {
  if (!value) return <span className="text-muted-foreground">-</span>;
  const info = DIAGNOSIS_INFO[value] ?? {
    label: value,
    hint: value,
    variant: "secondary" as const,
  };
  return (
    <Badge variant={info.variant} className="text-xs whitespace-nowrap" title={info.hint}>
      {info.label}
    </Badge>
  );
}

export function DiagnosisCounts({ counts }: { counts: Record<string, number> | null | undefined }) {
  if (!counts) return null;
  const present = DIAGNOSIS_CLASSES.filter((name) => (counts[name] ?? 0) > 0);
  if (present.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="text-muted-foreground">Failure classes:</span>
      {present.map((name) => (
        <span key={name} className="flex items-center gap-1">
          <DiagnosisBadge value={name} />
          <span className="font-mono">{counts[name]}</span>
        </span>
      ))}
    </div>
  );
}

export function DiagnosisTransitions({
  transitions,
}: {
  transitions: DiagnosisTransition[] | undefined;
}) {
  if (!transitions || transitions.length === 0) return null;
  return (
    <div className="space-y-3 mt-3">
      {transitions.map((transition) => {
        const sources = DIAGNOSIS_CLASSES.filter((name) => transition.matrix[name]);
        const targets = DIAGNOSIS_CLASSES.filter((name) =>
          sources.some((source) => transition.matrix[source]?.[name]),
        );
        return (
          <div key={`${transition.from_run_id}-${transition.to_run_id}`} className="space-y-1">
            <h4 className="text-xs font-medium">
              Classes: run #{transition.from_run_id} → #{transition.to_run_id} (
              {transition.compared} shared questions, {transition.changes.length} changed)
            </h4>
            {sources.length > 0 && (
              <div className="overflow-auto border rounded-md">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">was ↓ / now →</TableHead>
                      {targets.map((name) => (
                        <TableHead key={name} className="text-xs">
                          <DiagnosisBadge value={name} />
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sources.map((source) => (
                      <TableRow key={source}>
                        <TableCell className="text-xs">
                          <DiagnosisBadge value={source} />
                        </TableCell>
                        {targets.map((target) => {
                          const count = transition.matrix[source]?.[target] ?? 0;
                          return (
                            <TableCell
                              key={target}
                              className={`text-xs font-mono ${count && source !== target ? "font-semibold" : "text-muted-foreground"}`}
                            >
                              {count || "·"}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
            {transition.changes.length > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-muted-foreground">
                  Changed questions
                </summary>
                <ul className="mt-1 space-y-0.5">
                  {transition.changes.map((change) => (
                    <li key={change.key} className="flex items-center gap-1.5">
                      <DiagnosisBadge value={change.from} />→
                      <DiagnosisBadge value={change.to} />
                      <span className="truncate">{change.question}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

function formatScore(value: number | null | undefined, digits = 2) {
  return value == null ? "-" : value.toFixed(digits);
}

function LabelInput({
  runId,
  questionKey,
  metric,
  label,
}: {
  runId: number;
  questionKey: string;
  metric: LabelMetric;
  label: HumanLabel | undefined;
}) {
  const inputId = useId();
  const save = useSaveLabel(runId);
  const remove = useDeleteLabel(runId);
  const [draft, setDraft] = useState<string | null>(null);
  const value = draft ?? (label ? String(label.score) : "");
  const pending = save.isPending || remove.isPending;
  const title = metric === "correctness" ? "Correctness" : "Faithfulness";

  return (
    <form
      className="space-y-2 rounded-lg border bg-background p-4"
      onSubmit={async (event) => {
        event.preventDefault();
        const score = Number(value);
        if (!value.trim() || !Number.isFinite(score) || score < 0 || score > 10) {
          toast.error("Enter a score between 0 and 10");
          return;
        }
        try {
          await save.mutateAsync({ question_key: questionKey, metric, score, note: label?.note });
          setDraft(null);
        } catch (error) {
          toast.error(getErrorMessage(error, "Could not save the score"));
        }
      }}
    >
      <Label htmlFor={inputId} className="text-base font-semibold">
        {title}
      </Label>
      <p className="text-sm text-muted-foreground">
        {metric === "correctness"
          ? "Does the answer match the reference?"
          : "Is the answer supported by the recorded context?"}
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Input
          id={inputId}
          type="number"
          min={0}
          max={10}
          step="any"
          required
          inputMode="decimal"
          className="h-10 w-24 text-base"
          placeholder="0–10"
          disabled={pending}
          value={value}
          onChange={(event) => setDraft(event.target.value)}
        />
        <Button type="submit" disabled={pending || !value.trim() || Number(value) === label?.score}>
          {save.isPending ? "Saving…" : "Save score"}
        </Button>
        {label && (
          <Button
            type="button"
            variant="ghost"
            disabled={pending}
            onClick={async () => {
              try {
                await remove.mutateAsync({ question_key: questionKey, metric });
                setDraft(null);
              } catch (error) {
                toast.error(getErrorMessage(error, "Could not remove the score"));
              }
            }}
          >
            {remove.isPending ? "Removing…" : "Remove"}
          </Button>
        )}
      </div>
      <p className="text-sm text-muted-foreground" role="status">
        {label ? `Saved score · ${label.score}/10` : "No score saved"}
      </p>
    </form>
  );
}

function AgreementSummary({ runId }: { runId: number }) {
  const agreementQuery = useJudgeAgreement(runId);
  const { data: agreement } = agreementQuery;
  const calibrate = useSaveCalibration(runId);
  if (agreementQuery.isError)
    return <QueryError query={agreementQuery} fallback="Could not load judge agreement." />;
  if (!agreement) return <p role="status">Loading judge agreement…</p>;
  const labelled = Object.values(agreement.metrics).some((metric) => metric.n > 0);

  const describe = (name: string, metric: AgreementMetric) => {
    const weak =
      metric.kappa != null && metric.min_kappa != null && metric.kappa < metric.min_kappa;
    return (
      <span key={name} className={weak ? "text-destructive" : undefined}>
        {name}: n={metric.n}, MAE {formatScore(metric.mae)}, Spearman {formatScore(metric.spearman)}
        , κ {formatScore(metric.kappa)}
        {weak && " — fix the judge before the pipeline"}
      </span>
    );
  };

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
      <span>
        Judge {agreement.judge_model ?? "?"} (rubric {agreement.judge_rubric_version ?? "?"}) vs
        human labels:
      </span>
      {Object.entries(agreement.metrics).map(([name, metric]) => describe(name, metric))}
      <Button
        variant="outline"
        size="sm"
        className="h-7"
        disabled={!labelled || calibrate.isPending}
        onClick={async () => {
          try {
            await calibrate.mutateAsync();
            toast.success(`Calibration saved for ${agreement.dataset}`);
          } catch (error) {
            toast.error(getErrorMessage(error, "Could not save the calibration"));
          }
        }}
      >
        Save as dataset calibration
      </Button>
    </div>
  );
}

function ContextBlock({ row }: { row: QuestionRow }) {
  if (row.evidence?.context) {
    return (
      <div className="mt-3 space-y-2 border-t pt-3">
        Context given to the model:
        <pre className="whitespace-pre-wrap break-words font-sans text-base leading-relaxed">
          {row.evidence.context}
        </pre>
      </div>
    );
  }

  const gate = lastGateCheck(row);

  if (!gate) {
    return (
      <div className="mt-3 space-y-2 border-t pt-3">
        Context given to the model: <span>not recorded for this run</span>
      </div>
    );
  }

  return (
    <div className="mt-3 space-y-2 border-t pt-3">
      <p>
        {gate.result?.is_relevant === false
          ? `No prompt was built: the relevance gate rejected the question.${
              gate.result.reason ? ` Reason: ${gate.result.reason}` : ""
            }`
          : "No prompt context was recorded for this answer."}
      </p>
      <p className="mt-1">Context seen by the relevance gate:</p>
      <pre className="whitespace-pre-wrap break-words font-sans text-base leading-relaxed">
        {gate.context}
      </pre>
    </div>
  );
}

function RecordedContext({ row }: { row: QuestionRow }) {
  const [open, setOpen] = useState(false);
  const contentId = useId();
  const toggle = useRef<HTMLButtonElement>(null);

  function toggleContext() {
    setOpen(!open);
    if (open) {
      // Keep the control in view when the content under the scroll position disappears.
      requestAnimationFrame(() => toggle.current?.scrollIntoView({ block: "nearest" }));
    }
  }

  return (
    <section className="rounded-lg border bg-background">
      <div className="sticky top-0 z-10 flex items-center justify-between gap-2 rounded-t-lg bg-background px-4 py-2">
        <h4 className="font-semibold">Recorded context</h4>
        <Button
          ref={toggle}
          type="button"
          variant="outline"
          size="sm"
          aria-label={open ? "Collapse recorded context" : "Expand recorded context"}
          aria-expanded={open}
          aria-controls={contentId}
          onClick={toggleContext}
        >
          {open ? "Collapse" : "Expand"}
        </Button>
      </div>
      <div id={contentId} hidden={!open} className="select-text px-4 pb-4">
        {open && <ContextBlock row={row} />}
      </div>
    </section>
  );
}

const REVIEW_PAGE_SIZE = 25;
const REVIEW_SELECT = "h-10 w-full rounded-md border bg-background px-2.5 text-sm";
const REVIEW_ROW =
  "grid grid-cols-2 items-center gap-x-4 gap-y-2 px-4 py-3 sm:grid-cols-[minmax(0,1fr)_8rem_7rem_7rem]";

export function RunQuestionResults({ runId }: { runId: number }) {
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [metric, setMetric] = useState<LabelMetric>("correctness");
  const [comparison, setComparison] =
    useState<NonNullable<ReviewQuestionParams["comparison"]>>("eq");
  const [score, setScore] = useState("");
  const [sort, setSort] = useState<NonNullable<ReviewQuestionParams["sort"]>>("position");
  const [page, setPage] = useState(0);
  const reviewTop = useRef<HTMLDivElement>(null);
  const questionDetailsPrefix = useId();
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set());
  const [blind, setBlind] = useState(false);
  const [showAgreement, setShowAgreement] = useState(false);
  const userQuery = useCurrentUser();
  const labelsQuery = useRunLabels(runId);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQuery(search.trim());
      setPage(0);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [search]);

  const questionsQuery = useBenchmarkReviewQuestions(runId, {
    query: query || undefined,
    diagnosis: !blind && diagnosis ? diagnosis : undefined,
    metric: blind ? undefined : metric,
    comparison: blind ? undefined : comparison,
    score: !blind && score !== "" && comparison !== "unscored" ? Number(score) : undefined,
    sort: blind ? undefined : sort,
    limit: REVIEW_PAGE_SIZE,
    offset: page * REVIEW_PAGE_SIZE,
  });
  const { data } = questionsQuery;
  const canLabel = Boolean(userQuery.data && labelsQuery.data);
  const ownLabels = useMemo(() => {
    const grouped = new Map<string, Partial<Record<LabelMetric, HumanLabel>>>();
    for (const label of labelsQuery.data ?? []) {
      if (label.labeler_id === userQuery.data?.id) {
        const scores = grouped.get(label.question_key) ?? {};
        scores[label.metric] = label;
        grouped.set(label.question_key, scores);
      }
    }
    return grouped;
  }, [labelsQuery.data, userQuery.data?.id]);

  function toggleQuestion(position: number) {
    setExpanded((open) => {
      const next = new Set(open);
      if (next.has(position)) next.delete(position);
      else next.add(position);
      return next;
    });
  }

  function goToPage(nextPage: number) {
    setPage(nextPage);
    reviewTop.current?.scrollIntoView({ block: "start" });
  }

  function selectExactTen() {
    setMetric("correctness");
    setComparison("eq");
    setScore("10");
    setPage(0);
  }

  function resetFilters() {
    setSearch("");
    setQuery("");
    setDiagnosis("");
    setMetric("correctness");
    setComparison("eq");
    setScore("");
    setSort("position");
    setPage(0);
  }

  const rangeStart = data?.questions.length ? page * REVIEW_PAGE_SIZE + 1 : 0;
  const rangeEnd = data?.questions.length ? page * REVIEW_PAGE_SIZE + data.questions.length : 0;
  const hasFilters = Boolean(
    search ||
      (!blind && (diagnosis || score !== "" || comparison === "unscored" || sort !== "position")),
  );
  const reviewedCount = [...ownLabels.values()].filter(
    (scores) => scores.correctness && scores.faithfulness,
  ).length;

  return (
    <div ref={reviewTop} className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground" role="status">
          {data && canLabel
            ? `${reviewedCount} of ${data.total} fully labelled`
            : "Review and label answers"}
          {blind && " · Blind review"}
        </p>
        <Button
          variant="outline"
          aria-pressed={blind}
          onClick={() => {
            setBlind(!blind);
            setPage(0);
          }}
        >
          {blind ? "Show judge scores" : "Hide judge scores"}
        </Button>
      </div>
      <div className="space-y-3 rounded-lg border bg-muted/25 p-4">
        <div className="flex flex-wrap items-end gap-2">
          <div className="min-w-44 flex-1 space-y-1.5">
            <Label htmlFor="review-question-search">Find a question</Label>
            <Input
              id="review-question-search"
              type="search"
              placeholder="Question text or number"
              maxLength={500}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          {!blind && (
            <Button variant="outline" onClick={selectExactTen}>
              Correctness = 10
            </Button>
          )}
          {hasFilters && (
            <Button variant="ghost" onClick={resetFilters}>
              Clear filters
            </Button>
          )}
        </div>
        {!blind && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            <div className="space-y-1.5">
              <Label htmlFor="review-score-metric">Judge metric</Label>
              <select
                id="review-score-metric"
                className={REVIEW_SELECT}
                value={metric}
                onChange={(event) => {
                  setMetric(event.target.value as LabelMetric);
                  setPage(0);
                }}
              >
                <option value="correctness">Correctness</option>
                <option value="faithfulness">Faithfulness</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="review-score-condition">Score condition</Label>
              <select
                id="review-score-condition"
                className={REVIEW_SELECT}
                value={comparison}
                onChange={(event) => {
                  setComparison(
                    event.target.value as NonNullable<ReviewQuestionParams["comparison"]>,
                  );
                  setPage(0);
                }}
              >
                <option value="eq">Equals</option>
                <option value="gte">At least</option>
                <option value="lt">Below</option>
                <option value="unscored">Unscored</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="review-score-filter">Score (0–10)</Label>
              <Input
                id="review-score-filter"
                type="number"
                inputMode="decimal"
                min={0}
                max={10}
                step="any"
                placeholder="Any score"
                value={score}
                disabled={comparison === "unscored"}
                onChange={(event) => {
                  const { value, validity } = event.currentTarget;
                  if (validity.valid) {
                    setScore(value);
                    setPage(0);
                  }
                }}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="review-diagnosis">Failure class</Label>
              <select
                id="review-diagnosis"
                className={REVIEW_SELECT}
                value={diagnosis}
                onChange={(event) => {
                  setDiagnosis(event.target.value);
                  setPage(0);
                }}
              >
                <option value="">All classes</option>
                {DIAGNOSIS_CLASSES.map((name) => (
                  <option key={name} value={name}>
                    {DIAGNOSIS_INFO[name].label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="review-sort">Sort questions</Label>
              <select
                id="review-sort"
                className={REVIEW_SELECT}
                value={sort}
                onChange={(event) => {
                  setSort(event.target.value as NonNullable<ReviewQuestionParams["sort"]>);
                  setPage(0);
                }}
              >
                <option value="position">Question order</option>
                <option value="score_desc">Highest score first</option>
                <option value="score_asc">Lowest score first</option>
              </select>
            </div>
          </div>
        )}
      </div>
      {labelsQuery.isError && (
        <QueryError query={labelsQuery} fallback="Could not load saved scores." />
      )}
      {userQuery.isError && (
        <QueryError query={userQuery} fallback="Could not load the reviewer profile." />
      )}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground" aria-live="polite">
          {data
            ? `${rangeStart}–${rangeEnd} of ${data.matched} matching · ${data.total} in this run`
            : questionsQuery.isPending
              ? "Loading question list…"
              : "Question list unavailable"}
        </p>
        {expanded.size > 0 && (
          <Button variant="ghost" size="sm" onClick={() => setExpanded(new Set())}>
            Collapse all
          </Button>
        )}
      </div>
      {questionsQuery.isError && (
        <QueryError query={questionsQuery} fallback="Could not load this question list." />
      )}
      {questionsQuery.isPending && <Skeleton className="h-36 w-full" />}
      {data?.questions.length === 0 && (
        <p className="rounded-lg border p-5 text-muted-foreground">
          {data.total === 0
            ? "No questions were recorded for this run."
            : "No questions match these filters."}
        </p>
      )}
      {data && data.questions.length > 0 && (
        <div className="rounded-lg border">
          <div
            className={`${REVIEW_ROW} hidden rounded-t-lg bg-muted/40 text-sm font-medium sm:grid`}
            aria-hidden="true"
          >
            <span>Question</span>
            <span>{blind ? "" : "Judge correct."}</span>
            <span>{blind ? "" : "Judge faithful."}</span>
            <span>Your scores</span>
          </div>
          <ul className="divide-y" aria-label="Run questions">
            {data.questions.map((question) => {
              const open = expanded.has(question.position);
              const detailsId = `${questionDetailsPrefix}-${question.position}`;
              const labels = ownLabels.get(question.key);
              const saved =
                Number(Boolean(labels?.correctness)) + Number(Boolean(labels?.faithfulness));
              return (
                <li key={question.position}>
                  <div className={`${REVIEW_ROW} select-text`}>
                    <div className="col-span-2 flex min-w-0 items-start gap-2 sm:col-span-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="-ml-2 -mt-1 size-9 shrink-0"
                        aria-label={`${open ? "Collapse" : "Expand"} question ${question.position}`}
                        aria-expanded={open}
                        aria-controls={detailsId}
                        onClick={() => toggleQuestion(question.position)}
                      >
                        {open ? <ChevronDown aria-hidden /> : <ChevronRight aria-hidden />}
                      </Button>
                      <div className="min-w-0 space-y-1">
                        <p className="break-words font-medium">
                          #{question.position} · {question.question || "Untitled question"}
                        </p>
                        {!blind && <DiagnosisBadge value={question.diagnosis} />}
                      </div>
                    </div>
                    <span className="text-sm tabular-nums">
                      {!blind && (
                        <>
                          <span className="sm:sr-only">Correctness </span>
                          {question.correctness ?? "—"}
                        </>
                      )}
                    </span>
                    <span className="text-sm tabular-nums">
                      {!blind && (
                        <>
                          <span className="sm:sr-only">Faithfulness </span>
                          {question.faithfulness ?? "—"}
                        </>
                      )}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {canLabel ? `${saved}/2 saved` : "—"}
                    </span>
                  </div>
                  <div id={detailsId} hidden={!open}>
                    {open && (
                      <ReviewQuestionDetails
                        key={question.key}
                        runId={runId}
                        question={question}
                        labels={labels}
                        canLabel={canLabel}
                        refreshQuestions={() => void questionsQuery.refetch()}
                      />
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {(page > 0 || (data?.matched ?? 0) > REVIEW_PAGE_SIZE) && (
        <nav
          className="flex flex-wrap items-center justify-between gap-2"
          aria-label="Question pagination"
        >
          <span className="text-sm text-muted-foreground">
            Page {page + 1}
            {data && ` of ${Math.max(1, Math.ceil(data.matched / REVIEW_PAGE_SIZE))}`}
          </span>
          <div className="flex gap-2">
            <Button variant="outline" disabled={page === 0} onClick={() => goToPage(page - 1)}>
              Previous
            </Button>
            <Button
              variant="outline"
              disabled={!data || rangeEnd >= data.matched}
              onClick={() => goToPage(page + 1)}
            >
              Next
            </Button>
          </div>
        </nav>
      )}
      {!blind && (
        <details
          className="rounded-lg border p-4"
          onToggle={(event) => setShowAgreement(event.currentTarget.open)}
        >
          <summary className="cursor-pointer font-medium">Judge agreement and calibration</summary>
          <div className="mt-3">{showAgreement && <AgreementSummary runId={runId} />}</div>
        </details>
      )}
    </div>
  );
}

function ReviewQuestionDetails({
  runId,
  question,
  labels,
  canLabel,
  refreshQuestions,
}: {
  runId: number;
  question: BenchmarkReviewQuestion;
  labels: Partial<Record<LabelMetric, HumanLabel>> | undefined;
  canLabel: boolean;
  refreshQuestions: () => void;
}) {
  const detailQuery = useBenchmarkReviewQuestion(runId, question.position, question.key);
  const row: QuestionRow | undefined = detailQuery.data;

  if (row?.question_key !== undefined && row.question_key !== question.key) {
    return (
      <div className="flex flex-wrap items-center gap-3 p-4">
        <p className="text-sm" role="alert">
          This run changed while it was open.
        </p>
        <Button variant="outline" onClick={refreshQuestions}>
          Refresh questions
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4 border-t bg-muted/20 px-4 py-4 sm:px-7">
      {detailQuery.isError && (
        <QueryError query={detailQuery} fallback="Could not load the answer and evidence." />
      )}
      {detailQuery.isPending && (
        <div role="status">
          <Skeleton className="h-24 w-full" />
          <span className="sr-only">Loading answer and evidence</span>
        </div>
      )}
      {row && (
        <>
          {row.as_of_date && (
            <p className="text-sm text-muted-foreground">As of {row.as_of_date}</p>
          )}
          {row.history && row.history.length > 0 && (
            <details className="rounded-lg border bg-background p-4">
              <summary className="cursor-pointer font-medium">Conversation history</summary>
              <div className="mt-3 space-y-3">
                {row.history.map((turn, index) => (
                  <p key={index} className="whitespace-pre-wrap break-words leading-relaxed">
                    <strong>{turn.role}</strong> {turn.content}
                  </p>
                ))}
              </div>
            </details>
          )}
          {row.history === undefined && (
            <p className="text-sm text-muted-foreground">
              History and as-of date were not recorded for this run.
            </p>
          )}
          <div className="grid select-text gap-4 lg:grid-cols-2">
            <section className="min-w-0 rounded-lg border bg-background p-4">
              <h4 className="mb-2 font-semibold">Model answer</h4>
              <p className="whitespace-pre-wrap break-words text-base leading-relaxed">
                {row.answer || "No answer recorded."}
              </p>
            </section>
            <section className="min-w-0 rounded-lg border bg-background p-4">
              <h4 className="mb-2 font-semibold">Reference answer</h4>
              <p className="whitespace-pre-wrap break-words text-base leading-relaxed">
                {row.expected_answer || "No reference answer recorded."}
              </p>
            </section>
          </div>
          <RecordedContext row={row} />
          {canLabel && (
            <section
              className="grid gap-4 lg:grid-cols-2"
              aria-label={`Your scores for question ${question.position}`}
            >
              {(["correctness", "faithfulness"] as const).map((metric) => (
                <LabelInput
                  key={metric}
                  runId={runId}
                  questionKey={question.key}
                  metric={metric}
                  label={labels?.[metric]}
                />
              ))}
            </section>
          )}
          <details className="rounded-lg border bg-background p-4">
            <summary className="cursor-pointer font-semibold">Retrieval and timing</summary>
            <dl className="mt-3 grid gap-x-5 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
              {[
                ["Outcome", row.outcome ?? "—"],
                ["Fragment recall", formatScore(row.evidence_metrics?.fragment_recall_at_k)],
                ["Context fragments", formatScore(row.evidence_metrics?.context_fragment_recall)],
                ["Context facts", formatScore(row.evidence_metrics?.context_fact_coverage)],
                ["Latency", row.latency_sec == null ? "—" : `${row.latency_sec.toFixed(1)}s`],
              ].map(([name, value]) => (
                <div key={name}>
                  <dt className="text-muted-foreground">{name}</dt>
                  <dd className="font-medium">
                    {value}
                    {name === "Latency" && row.llm_replayed && " (replayed LLM calls)"}
                  </dd>
                </div>
              ))}
            </dl>
          </details>
        </>
      )}
    </div>
  );
}

// ─── Acceptance tab ───────────────────────────────────────────────────────

const THRESHOLD_FIELDS: Array<{
  key: keyof AcceptanceThresholds;
  label: string;
  max: number;
  hint: string;
}> = [
  {
    key: "min_fragment_recall_at_k",
    label: "Min fragment recall@k",
    max: 1,
    hint: "0–1, retrieval",
  },
  {
    key: "min_context_fragment_recall",
    label: "Min context fragment recall",
    max: 1,
    hint: "0–1, prompt context",
  },
  {
    key: "min_correctness",
    label: "Min correctness",
    max: 10,
    hint: "0–10, judge",
  },
  {
    key: "max_false_answer_rate",
    label: "Max false answer rate",
    max: 1,
    hint: "0–1, of expected refusals",
  },
];

const COMPOSITION_LABELS: Record<string, string> = {
  factual_with_fragments: "Factual with fragments (≥ 50%)",
  multi_fact: "Several required facts (≥ 15%)",
  abstention: "Correct refusal (≥ 15%)",
  with_history: "With conversation history (≥ 10%)",
  dated: "With as-of date (law: ≥ 10%)",
};

function ThresholdsForm({
  dataset,
  thresholds,
}: {
  dataset: string;
  thresholds: AcceptanceThresholds;
}) {
  const update = useUpdateThresholds(dataset);
  const [draft, setDraft] = useState<Record<string, string>>({});

  useEffect(() => {
    setDraft(
      Object.fromEntries(
        THRESHOLD_FIELDS.map(({ key }) => [
          key,
          thresholds[key] == null ? "" : String(thresholds[key]),
        ]),
      ),
    );
  }, [thresholds]);

  const submit = async () => {
    const parsed = {} as AcceptanceThresholds;
    for (const field of THRESHOLD_FIELDS) {
      const text = (draft[field.key] ?? "").trim();
      const value = text === "" ? null : Number(text);
      if (value != null && (!Number.isFinite(value) || value < 0 || value > field.max)) {
        toast.error(`${field.label}: expected 0–${field.max}`);
        return;
      }
      parsed[field.key] = value;
    }
    try {
      await update.mutateAsync(parsed);
      toast.success("Thresholds saved");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not save thresholds"));
    }
  };

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        {THRESHOLD_FIELDS.map((field) => (
          <div key={field.key} className="space-y-1">
            <Label htmlFor={`threshold-${field.key}`} className="text-xs">
              {field.label} <span className="text-muted-foreground">({field.hint})</span>
            </Label>
            <Input
              id={`threshold-${field.key}`}
              inputMode="decimal"
              placeholder="not set"
              value={draft[field.key] ?? ""}
              onChange={(event) => setDraft({ ...draft, [field.key]: event.target.value })}
            />
          </div>
        ))}
      </div>
      <Button size="sm" onClick={submit} disabled={update.isPending}>
        Save thresholds
      </Button>
    </div>
  );
}

function CalibrationHistory({ dataset }: { dataset: string }) {
  const history = useJudgeCalibrations(dataset);
  const { data: calibrations } = history;
  if (history.isError)
    return <QueryError query={history} fallback="Could not load calibrations." />;
  if (history.isPending) return <p role="status">Loading calibrations…</p>;

  if (!calibrations?.length) return <p className="text-muted-foreground">Not calibrated yet.</p>;

  return (
    <div className="space-y-2">
      {calibrations.map((calibration, index) => (
        <div
          key={calibration.id ?? calibration.measured_at}
          className={index === 0 ? undefined : "border-t pt-2 text-muted-foreground"}
        >
          <p className={index === 0 ? "text-muted-foreground" : undefined}>
            {index === 0 && <Badge className="mr-1">current</Badge>}
            {calibration.run_id != null ? `Run #${calibration.run_id}` : "Deleted run"}, judge{" "}
            {calibration.judge_model ?? "?"} (rubric {calibration.judge_rubric_version ?? "?"}),{" "}
            {new Date(calibration.measured_at).toLocaleString()}
          </p>
          {Object.entries(calibration.metrics).map(([metric, value]) => (
            <p key={metric} className="font-mono">
              {metric}: n={value.n} MAE={formatScore(value.mae)} Spearman=
              {formatScore(value.spearman)} κ={formatScore(value.kappa)}
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}

function NoisePanel({ dataset }: { dataset: string }) {
  const policyQuery = useDatasetPolicy(dataset);
  const runsQuery = useBenchmarkRuns({ dataset, limit: 20 });
  const { data: runs } = runsQuery;
  const measure = useMeasureNoise(dataset);
  const [selected, setSelected] = useState<number[]>([]);
  const noise = policyQuery.data?.noise;

  return (
    <div className="space-y-3 text-sm">
      {runsQuery.isError && (
        <QueryError query={runsQuery} fallback="Could not load finished runs." />
      )}
      {policyQuery.isError ? (
        <QueryError query={policyQuery} fallback="Could not load the noise measurement." />
      ) : policyQuery.isPending ? (
        <p role="status">Loading noise measurement…</p>
      ) : noise ? (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">
            Measured {new Date(noise.measured_at).toLocaleString()} on runs{" "}
            {noise.run_ids.join(", ")} with judge {noise.judge_model ?? "?"} (rubric{" "}
            {noise.judge_rubric_version ?? "?"}). A change is significant above 2σ.
          </p>
          <div className="flex flex-wrap gap-3 font-mono text-xs">
            {Object.entries(noise.sigma).map(([metric, sigma]) => (
              <span key={metric}>
                {metric}: σ={sigma} (2σ={(2 * sigma).toFixed(4)})
              </span>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          Not measured. Pick 3–10 finished runs of one configuration and judge (or run{" "}
          <code>benchmark noise -d {dataset} --repeat 3</code>, which is paid).
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {(runs?.runs ?? []).map((run) => (
          <label key={run.id} className="flex items-center gap-1 text-xs border rounded px-2 py-1">
            <input
              type="checkbox"
              checked={selected.includes(run.id)}
              onChange={() =>
                setSelected((current) =>
                  current.includes(run.id)
                    ? current.filter((id) => id !== run.id)
                    : [...current, run.id],
                )
              }
            />
            #{run.id}
            <span className="text-muted-foreground">
              {run.creation_date ? new Date(run.creation_date).toLocaleDateString() : ""}
            </span>
          </label>
        ))}
      </div>
      <Button
        size="sm"
        variant="outline"
        disabled={selected.length < 3 || measure.isPending}
        onClick={async () => {
          try {
            await measure.mutateAsync(selected);
            toast.success("Noise stored");
          } catch (error) {
            toast.error(getErrorMessage(error, "Could not measure noise"));
          }
        }}
      >
        Measure noise from {selected.length} runs
      </Button>
    </div>
  );
}

export function AcceptanceTab() {
  const datasetsQuery = useBenchmarkDatasets();
  const { data: datasets } = datasetsQuery;
  const [dataset, setDataset] = useState("");
  const active = dataset || datasets?.find((name) => name.endsWith("-main")) || datasets?.[0] || "";
  const policyQuery = useDatasetPolicy(active);
  const compositionQuery = useDatasetComposition(active);
  const { data: policy } = policyQuery;
  const { data: composition } = compositionQuery;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Label className="text-sm">Dataset</Label>
        <Select value={active} onValueChange={setDataset}>
          <SelectTrigger className="w-64" aria-label="Dataset">
            <SelectValue
              placeholder={datasetsQuery.isPending ? "Loading datasets…" : "Select dataset"}
            />
          </SelectTrigger>
          <SelectContent>
            {(datasets ?? []).map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {datasetsQuery.isError && (
        <QueryError query={datasetsQuery} fallback="Could not load datasets." />
      )}
      {datasets?.length === 0 && <p className="text-muted-foreground">No datasets.</p>}
      {active && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Acceptance thresholds</CardTitle>
              <CardDescription>
                Checked on the whole run by the regression check; empty means not enforced.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {policyQuery.isError && (
                <QueryError query={policyQuery} fallback="Could not load acceptance thresholds." />
              )}
              {policy && <ThresholdsForm dataset={active} thresholds={policy.thresholds} />}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                Composition
                {composition && (
                  <Badge variant={composition.passed ? "success" : "warning"}>
                    {composition.passed ? "meets minimum" : "incomplete"}
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                Active questions of a `-main` set: 40–60 with the shares below.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-1 text-sm">
              {compositionQuery.isError && (
                <QueryError
                  query={compositionQuery}
                  fallback="Could not load dataset composition."
                />
              )}
              {composition && (
                <>
                  <p>Active questions: {composition.total}</p>
                  {Object.entries(COMPOSITION_LABELS).map(([key, label]) => (
                    <p key={key} className="flex justify-between gap-2 text-xs">
                      <span>{label}</span>
                      <span className="font-mono">
                        {composition.counts[key] ?? 0} (
                        {Math.round((composition.shares[key] ?? 0) * 100)}%)
                      </span>
                    </p>
                  ))}
                  {composition.failures.length > 0 && (
                    <ul className="text-xs text-destructive list-disc pl-4 pt-1">
                      {composition.failures.map((failure) => (
                        <li key={failure}>{failure}</li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Noise (σ of repeated runs)</CardTitle>
              <CardDescription>
                Valid only for the judge model and rubric it was measured with.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <NoisePanel dataset={active} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Judge calibration</CardTitle>
              <CardDescription>
                Label 30–50 answers in Leaderboard → Review questions. Save the calibration under
                Judge agreement and calibration.
              </CardDescription>
            </CardHeader>
            <CardContent className="text-xs space-y-1">
              <CalibrationHistory dataset={active} />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
