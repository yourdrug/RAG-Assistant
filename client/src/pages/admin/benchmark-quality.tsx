import { ListChecks } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import toast from "react-hot-toast";
import { useBenchmarkDatasets, useBenchmarkRuns, useCurrentUser } from "@/shared/api/hooks";
import {
  type AcceptanceThresholds,
  type AgreementMetric,
  DIAGNOSIS_CLASSES,
  type DiagnosisTransition,
  type HumanLabel,
  type LabelMetric,
  resultKey,
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
import type { BenchmarkResultDetail } from "@/shared/api/types";
import { getErrorMessage } from "@/shared/lib/errors";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

type QuestionRow = NonNullable<BenchmarkResultDetail["per_question_results"]>[number] & {
  outcome?: string | null;
  diagnosis?: string | null;
  run?: number;
  case_id?: string | null;
  history?: { role: string; content: string }[] | null;
  as_of_date?: string | null;
  evidence?: { context?: string | null } | null;
};

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
  const save = useSaveLabel(runId);
  const remove = useDeleteLabel(runId);
  const [value, setValue] = useState(label ? String(label.score) : "");

  useEffect(() => setValue(label ? String(label.score) : ""), [label]);

  const commit = async () => {
    const text = value.trim();
    try {
      if (text === "") {
        if (label) await remove.mutateAsync({ question_key: questionKey, metric });
        return;
      }
      const score = Number(text);
      if (!Number.isFinite(score) || score < 0 || score > 10) {
        toast.error("Score must be between 0 and 10");
        setValue(label ? String(label.score) : "");
        return;
      }
      if (label?.score !== score)
        await save.mutateAsync({ question_key: questionKey, metric, score });
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not save the label"));
    }
  };

  return (
    <Input
      aria-label={`Human ${metric}`}
      inputMode="decimal"
      className="h-7 w-14 px-1.5 text-xs"
      placeholder="–"
      value={value}
      onChange={(event) => setValue(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") (event.target as HTMLInputElement).blur();
      }}
    />
  );
}

function AgreementSummary({ runId }: { runId: number }) {
  const { data: agreement } = useJudgeAgreement(runId);
  const calibrate = useSaveCalibration(runId);
  if (!agreement) return null;
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
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
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

export function RunQuestionResults({ runId, rows }: { runId: number; rows: QuestionRow[] }) {
  const [diagnosisFilter, setDiagnosisFilter] = useState("all");
  const [labelling, setLabelling] = useState(false);
  // Labels are given before seeing the judge, so they do not anchor on its scores.
  const [blind, setBlind] = useState(true);
  const hideJudge = labelling && blind;
  const { data: user } = useCurrentUser();
  const { data: labels } = useRunLabels(labelling ? runId : null);

  const ownLabels = useMemo(() => {
    const result = new Map<string, HumanLabel>();
    for (const label of labels ?? []) {
      if (label.labeler_id === user?.id) result.set(`${label.question_key}|${label.metric}`, label);
    }
    return result;
  }, [labels, user?.id]);

  const visible = rows.filter(
    (row) => hideJudge || diagnosisFilter === "all" || row.diagnosis === diagnosisFilter,
  );

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-sm font-medium flex items-center gap-1">
          <ListChecks className="h-4 w-4" /> Per-question results ({visible.length}/{rows.length})
        </h4>
        {!hideJudge && (
          <Select value={diagnosisFilter} onValueChange={setDiagnosisFilter}>
            <SelectTrigger className="h-7 w-44 text-xs" aria-label="Failure class">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All classes</SelectItem>
              {DIAGNOSIS_CLASSES.map((name) => (
                <SelectItem key={name} value={name}>
                  {DIAGNOSIS_INFO[name].label} (
                  {rows.filter((row) => row.diagnosis === name).length})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <Button
          variant={labelling ? "default" : "outline"}
          size="sm"
          className="h-7"
          onClick={() => setLabelling(!labelling)}
        >
          {labelling ? "Hide labelling" : "Label answers"}
        </Button>
        {labelling && (
          <Button variant="outline" size="sm" className="h-7" onClick={() => setBlind(!blind)}>
            {blind ? "Show judge scores" : "Hide judge scores"}
          </Button>
        )}
      </div>
      {labelling && !blind && <AgreementSummary runId={runId} />}
      <div className="overflow-auto max-h-[420px] border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">#</TableHead>
              <TableHead className="text-xs">Question</TableHead>
              {!hideJudge && <TableHead className="text-xs">Class</TableHead>}
              <TableHead className="text-xs">Outcome</TableHead>
              <TableHead className="text-xs">Fragment recall</TableHead>
              <TableHead className="text-xs">Context fragments</TableHead>
              <TableHead className="text-xs">Context facts</TableHead>
              {!hideJudge && <TableHead className="text-xs">Correct</TableHead>}
              {!hideJudge && <TableHead className="text-xs">Faith</TableHead>}
              {labelling && <TableHead className="text-xs">Human correct</TableHead>}
              {labelling && <TableHead className="text-xs">Human faith</TableHead>}
              <TableHead className="text-xs">Latency</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map((row, index) => {
              const key = resultKey(row);
              return (
                <TableRow key={key}>
                  <TableCell className="text-xs text-muted-foreground">{index + 1}</TableCell>
                  <TableCell className="text-xs max-w-[320px]">
                    <div className="truncate" title={row.question}>
                      {row.question}
                    </div>
                    {labelling && (
                      <details className="text-muted-foreground">
                        <summary className="cursor-pointer">Answer, reference and context</summary>
                        {row.as_of_date && <p className="mt-1">As of: {row.as_of_date}</p>}
                        {row.history && row.history.length > 0 && (
                          <div className="mt-1 border-t pt-1 space-y-0.5">
                            {row.history.map((turn, turnIndex) => (
                              <p key={turnIndex} className="whitespace-pre-wrap">
                                <span className="font-medium">{turn.role}:</span> {turn.content}
                              </p>
                            ))}
                          </div>
                        )}
                        <p className="whitespace-pre-wrap mt-1 border-t pt-1">{row.answer}</p>
                        {row.expected_answer && (
                          <p className="whitespace-pre-wrap mt-1 border-t pt-1">
                            Reference: {row.expected_answer}
                          </p>
                        )}
                        <div className="mt-1 border-t pt-1">
                          Context given to the model:
                          {row.evidence?.context ? (
                            <pre className="whitespace-pre-wrap font-sans max-h-64 overflow-auto mt-1">
                              {row.evidence.context}
                            </pre>
                          ) : (
                            <span> not recorded for this run</span>
                          )}
                        </div>
                        {row.history === undefined && (
                          <p className="mt-1 italic">
                            History and as-of date were not recorded for this run.
                          </p>
                        )}
                      </details>
                    )}
                  </TableCell>
                  {!hideJudge && (
                    <TableCell className="text-xs">
                      <DiagnosisBadge value={row.diagnosis} />
                    </TableCell>
                  )}
                  <TableCell className="text-xs font-mono">{row.outcome ?? "-"}</TableCell>
                  <TableCell className="text-xs font-mono">
                    {formatScore(row.evidence_metrics?.fragment_recall_at_k)}
                  </TableCell>
                  <TableCell className="text-xs font-mono">
                    {formatScore(row.evidence_metrics?.context_fragment_recall)}
                  </TableCell>
                  <TableCell className="text-xs font-mono">
                    {formatScore(row.evidence_metrics?.context_fact_coverage)}
                  </TableCell>
                  {!hideJudge && (
                    <TableCell className="text-xs font-mono">
                      {formatScore(row.correctness, 1)}
                    </TableCell>
                  )}
                  {!hideJudge && (
                    <TableCell className="text-xs font-mono">
                      {formatScore(row.faithfulness, 1)}
                    </TableCell>
                  )}
                  {labelling &&
                    (["correctness", "faithfulness"] as const).map((metric) => (
                      <TableCell key={metric} className="text-xs">
                        <LabelInput
                          runId={runId}
                          questionKey={key}
                          metric={metric}
                          label={ownLabels.get(`${key}|${metric}`)}
                        />
                      </TableCell>
                    ))}
                  <TableCell className="text-xs text-muted-foreground">
                    {row.latency_sec != null ? `${row.latency_sec.toFixed(1)}s` : "-"}
                    {row.llm_replayed && (
                      <span title="Есть повторённые вызовы LLM: латентность не является замером">
                        {" "}
                        (повтор)
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
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
                Label 30–50 answers in a run (Leaderboard → run → Label answers), then save its
                agreement here.
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
