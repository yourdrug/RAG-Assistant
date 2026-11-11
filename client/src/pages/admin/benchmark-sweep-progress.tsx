import { useJob, useSweep, useSweepProgress } from "@/shared/api/hooks";
import type { SweepResponse } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import { Progress } from "@/shared/ui/progress";

export function SweepProgressDialog({
  initialSweep,
  onClose,
}: {
  initialSweep: SweepResponse;
  onClose: () => void;
}) {
  const { data: current, isError: sweepError } = useSweep(initialSweep.id);
  const sweep = current ?? initialSweep;
  const active = sweep.status === "pending" || sweep.status === "running";
  const {
    data: progress,
    isLoading,
    isError,
    refetch,
    dataUpdatedAt,
  } = useSweepProgress(sweep.id, active);
  const { data: job } = useJob(sweep.job_id ?? null);
  const evaluated = progress?.evaluated ?? sweep.evaluated_configs;
  const total = progress?.total ?? sweep.total_configs;
  const percent = total > 0 ? Math.min(100, (evaluated / total) * 100) : 0;
  const full = progress?.phase === "full_evaluation";
  const phase =
    sweep.status === "pending"
      ? "Waiting for worker"
      : !active
        ? `Sweep ${sweep.status}`
        : !progress
          ? "Waiting for progress details"
          : full
            ? "2. Answer generation & judge evaluation"
            : "1. Retrieval evaluation";

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Sweep #{sweep.id} · Live progress</DialogTitle>
          <DialogDescription>
            {active ? "Updates automatically every 3 seconds." : "Execution has stopped."}
            {dataUpdatedAt > 0 && ` Last checked: ${new Date(dataUpdatedAt).toLocaleTimeString()}.`}
          </DialogDescription>
        </DialogHeader>

        {(isError || sweepError) && (
          <div role="alert" className="text-sm text-destructive">
            Could not refresh progress. Displayed values may be out of date.
            <Button
              size="sm"
              variant="outline"
              className="ml-2"
              onClick={() => {
                void refetch();
              }}
            >
              Retry
            </Button>
          </div>
        )}

        <section className="space-y-3 rounded-lg border p-4" aria-live="polite">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-semibold">{phase}</h3>
            <Badge variant="secondary">{sweep.status}</Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            {sweep.status === "pending"
              ? "The sweep is queued. Evaluation begins when a worker picks up the job."
              : !active
                ? "Final status is shown above. Completed results are available in the Leaderboard."
                : !progress
                  ? isLoading
                    ? "Loading execution details…"
                    : "No detailed update has been recorded yet. Preparing data or waiting for the next configuration to finish."
                  : full
                    ? "For each selected configuration, RAG generates answers for the dataset and the judge scores them. One configuration can take several minutes."
                    : "Testing search configurations against the dataset and measuring retrieval quality before generating answers."}
          </p>
          <Progress value={percent} />
          <p className="text-sm">
            {evaluated} / {total || "—"} configurations completed in the current phase
            {total > 0 && ` · ${percent.toFixed(0)}%`}
          </p>
          {active && (
            <p className="text-xs text-muted-foreground">
              The counter updates when a configuration finishes and resets for the next phase.
              {progress?.updated_at &&
                ` Last configuration update: ${new Date(progress.updated_at).toLocaleString()}.`}{" "}
              Time without a new result alone does not indicate a stalled worker.
            </p>
          )}
        </section>

        <dl className="grid grid-cols-2 gap-3 text-sm">
          {[
            ["Dataset", sweep.dataset],
            ["Strategy", sweep.strategy],
            ["Evaluation mode", sweep.evaluation_mode],
            [
              "Full evaluation scope",
              sweep.evaluation_mode === "full"
                ? "All configurations"
                : sweep.top_n_llm > 0
                  ? `Up to ${sweep.top_n_llm} retrieval finalists`
                  : "Retrieval only",
            ],
            [
              "Judge model",
              sweep.evaluation_mode === "fast" && sweep.top_n_llm === 0
                ? "Not used"
                : sweep.judge_model || "Default",
            ],
            [
              "Worker job",
              job ? `#${job.id} · ${job.status}` : sweep.job_id ? `#${sweep.job_id}` : "Not linked",
            ],
            ["Started", job?.started_at ? new Date(job.started_at).toLocaleString() : "—"],
            ["Best run", sweep.best_run_id ? `#${sweep.best_run_id}` : "Not selected yet"],
          ].map(([label, value]) => (
            <div key={label} className="min-w-0">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="break-words">{value}</dd>
            </div>
          ))}
        </dl>

        {progress?.latest?.config && (
          <section className="space-y-2">
            <h3 className="text-sm font-semibold">Last completed configuration</h3>
            <pre className="max-h-48 overflow-auto rounded-md bg-muted p-3 text-xs">
              {JSON.stringify(progress.latest.config, null, 2)}
            </pre>
            <div className="flex flex-wrap gap-4 text-sm">
              {[
                ["Hit rate", progress.latest.avg_hit_rate],
                ["MRR", progress.latest.avg_mrr],
                ["Composite score", progress.latest.composite_score],
              ].map(
                ([label, value]) =>
                  typeof value === "number" && (
                    <span key={label}>
                      {label}: {value.toFixed(3)}
                    </span>
                  ),
              )}
            </div>
          </section>
        )}
        {sweep.status === "failed" && job?.error_message && (
          <pre className="whitespace-pre-wrap break-words text-xs text-destructive">
            {job.error_message}
          </pre>
        )}
        <details className="text-sm">
          <summary className="cursor-pointer">Search space & objective weights</summary>
          <pre className="mt-2 max-h-56 overflow-auto rounded-md bg-muted p-3 text-xs">
            {JSON.stringify(
              { search_space: sweep.search_space, objective_weights: sweep.objective_weights },
              null,
              2,
            )}
          </pre>
        </details>
      </DialogContent>
    </Dialog>
  );
}
