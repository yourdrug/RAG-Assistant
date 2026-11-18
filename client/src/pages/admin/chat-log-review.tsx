"use client";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  useBenchmarkDraft,
  useReviewChatLog,
  useTransferToBenchmark,
} from "@/shared/api/hooks/use-chat-logs";
import type { BenchmarkDraft, ChatLogEntry, ReviewStatus } from "@/shared/api/types";
import { Button } from "@/shared/ui/button";
import { Label } from "@/shared/ui/label";
import { Textarea } from "@/shared/ui/textarea";

export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  new: "New",
  benchmark_candidate: "Benchmark candidate",
  in_benchmark: "In benchmark",
  content_gap: "Content gap",
  pipeline_issue: "Pipeline issue",
  wont_fix: "Won’t fix",
};

const MANUAL_STATUSES = (Object.keys(REVIEW_STATUS_LABELS) as ReviewStatus[]).filter(
  (status) => status !== "in_benchmark",
);

function apiError(error: unknown): string {
  const status = (error as { response?: { status?: number } })?.response?.status;
  if (status === 409) return "This entry has already been transferred to the benchmark.";
  if (status === 422) return "Check the form fields.";
  if (status === 403) return "Insufficient permissions.";
  return "Could not save changes.";
}

export function ReviewForm({ entry }: { entry: ChatLogEntry }) {
  const mutation = useReviewChatLog(entry.id);
  const [status, setStatus] = useState<ReviewStatus>(entry.review?.status ?? "new");
  const [note, setNote] = useState(entry.review?.note ?? "");
  const transferred = entry.review?.status === "in_benchmark";
  useEffect(() => {
    setStatus(entry.review?.status ?? "new");
    setNote(entry.review?.note ?? "");
  }, [entry.review?.status, entry.review?.note]);
  return (
    <div className="space-y-4 rounded-xl border bg-muted/20 p-5">
      <div className="space-y-1">
        <p className="font-medium">Review</p>
        <p className="text-sm text-muted-foreground">
          Save a triage status and an internal note. Creating a benchmark question is a separate
          action below.
        </p>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <Label className="flex min-w-48 flex-col gap-2">
          <span>Status</span>
          <select
            className="block h-10 rounded-md border border-input bg-background px-3 text-sm"
            disabled={mutation.isPending}
            value={status}
            onChange={(event) => setStatus(event.target.value as ReviewStatus)}
          >
            {transferred && (
              <option value="in_benchmark">{REVIEW_STATUS_LABELS.in_benchmark}</option>
            )}
            {MANUAL_STATUSES.map((value) => (
              <option key={value} value={value}>
                {REVIEW_STATUS_LABELS[value]}
              </option>
            ))}
          </select>
        </Label>
        <Button
          variant="outline"
          disabled={mutation.isPending || status === "in_benchmark"}
          onClick={() => mutation.mutate({ status, note: note.trim() || null })}
        >
          {mutation.isPending ? "Saving…" : "Save review"}
        </Button>
      </div>
      <Textarea
        aria-label="Review note"
        placeholder="Note"
        maxLength={2000}
        value={note}
        onChange={(event) => setNote(event.target.value)}
      />
      {entry.review?.benchmark_question_id != null && (
        <p className="text-sm">
          Transferred to benchmark question #{entry.review.benchmark_question_id}.
        </p>
      )}
      {mutation.isError && <p role="alert">{apiError(mutation.error)}</p>}
      {mutation.isSuccess && (
        <p role="status" className="text-sm text-emerald-700 dark:text-emerald-300">
          {mutation.data.status === "benchmark_candidate"
            ? "Saved as a benchmark candidate. An administrator can now create the question."
            : "Review saved."}
        </p>
      )}
    </div>
  );
}

/** ADMIN only: the server also requires the benchmark capability. */
export function BenchmarkTransferForm({ entry }: { entry: ChatLogEntry }) {
  const [open, setOpen] = useState(false);
  const draft = useBenchmarkDraft(entry.id, open);
  const transfer = useTransferToBenchmark(entry.id);
  const [form, setForm] = useState<BenchmarkDraft | null>(null);
  const [dataset, setDataset] = useState("");

  useEffect(() => {
    if (draft.data) {
      setForm(draft.data);
      setDataset(draft.data.datasets[0] ?? "");
    }
  }, [draft.data]);

  if (entry.review?.benchmark_question_id != null)
    return (
      <div className="space-y-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-5">
        <p className="font-medium">Benchmark question #{entry.review.benchmark_question_id}</p>
        <p className="text-sm text-muted-foreground">
          The question has been created. Complete its expected answer and annotations in Benchmark →
          Questions before activating it.
        </p>
        <Button asChild variant="outline">
          <Link to="/admin/benchmark?tab=questions">Open benchmark questions</Link>
        </Button>
      </div>
    );
  if (!open) {
    return (
      <section className="space-y-3 rounded-xl border border-primary/20 bg-muted/20 p-5">
        <div className="space-y-1">
          <h3 className="font-medium">Create a benchmark question</h3>
          <p className="text-sm text-muted-foreground">
            Review the question and conversation history, then choose a dataset. The new question
            starts inactive.
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>Add to benchmark</Button>
      </section>
    );
  }
  return (
    <div className="space-y-4 rounded-xl border bg-muted/20 p-5">
      <p className="font-medium">Transfer to benchmark</p>
      <p role="note" className="text-sm text-amber-600 dark:text-amber-400">
        Personal data has been automatically redacted. Review the text before adding it to a
        benchmark dataset, where it will be visible to benchmark administrators.
      </p>
      {draft.isLoading && <p>Loading…</p>}
      {draft.isError && <p role="alert">{apiError(draft.error)}</p>}
      {form && (
        <>
          {form.datasets.length === 0 ? (
            <p role="alert">This entry has no pack and cannot be transferred.</p>
          ) : (
            <Label className="flex flex-col gap-2">
              <span>Dataset</span>
              <select
                className="block h-10 rounded-md border border-input bg-background px-3 text-sm"
                value={dataset}
                onChange={(event) => setDataset(event.target.value)}
              >
                {form.datasets.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </Label>
          )}
          <Label className="flex flex-col gap-2">
            <span>Question</span>
            <Textarea
              value={form.question}
              onChange={(event) => setForm({ ...form, question: event.target.value })}
            />
          </Label>
          {form.history.map((turn, index) => (
            <Label key={index} className="flex flex-col gap-2">
              <span className="flex items-center justify-between">
                <span>
                  History {index + 1}: {turn.role === "user" ? "question" : "answer"}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setForm({ ...form, history: form.history.filter((_, item) => item !== index) })
                  }
                >
                  Remove
                </Button>
              </span>
              <Textarea
                value={turn.content}
                onChange={(event) =>
                  setForm({
                    ...form,
                    history: form.history.map((item, position) =>
                      position === index ? { ...item, content: event.target.value } : item,
                    ),
                  })
                }
              />
            </Label>
          ))}
          <p className="text-sm text-muted-foreground">
            Date: {form.as_of_date ?? "—"} · Sources: {form.source_hint ?? "—"}. The question
            remains inactive until its expected answer and annotations are complete.
          </p>
          <div className="flex gap-2">
            <Button
              disabled={transfer.isPending || !dataset || !form.question.trim()}
              onClick={() =>
                transfer.mutate({ dataset, question: form.question, history: form.history })
              }
            >
              {transfer.isPending ? "Creating question…" : "Create question"}
            </Button>
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
          </div>
          {transfer.isError && <p role="alert">{apiError(transfer.error)}</p>}
          {transfer.isSuccess && (
            <p className="text-sm">
              Created question #{transfer.data.question_id} in dataset {transfer.data.dataset}.
            </p>
          )}
        </>
      )}
    </div>
  );
}
