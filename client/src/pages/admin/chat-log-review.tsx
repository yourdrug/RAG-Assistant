"use client";
import { BookOpen, ClipboardList, Plus } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { Link } from "react-router-dom";
import { useReviewChatLog } from "@/shared/api/hooks/use-chat-logs";
import type { ChatLogEntry, ReviewStatus } from "@/shared/api/types";
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

export function chatLogReviewError(error: unknown): string {
  const response = (
    error as {
      response?: { status?: number; data?: { message?: string; errors?: Record<string, string> } };
    }
  )?.response;
  const status = response?.status;
  if (status === 409) return "This entry has already been transferred to the benchmark.";
  if (status === 422) {
    const errors = Object.entries(response?.data?.errors ?? {});
    return errors.length
      ? errors.map(([field, message]) => `${field}: ${message}`).join(". ")
      : (response?.data?.message ?? "Check the form fields.");
  }
  if (status === 403) return "Insufficient permissions.";
  return "Could not save changes.";
}

export function ReviewForm({ entry, canTransfer }: { entry: ChatLogEntry; canTransfer: boolean }) {
  const statusId = useId();
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
        <p className="text-sm text-muted-foreground">Save a triage status and an internal note.</p>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex min-w-48 flex-col gap-2">
          <Label htmlFor={statusId}>Status</Label>
          <select
            id={statusId}
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
        </div>
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
          {canTransfer && (
            <>
              {" "}
              <Link className="underline underline-offset-2" to="/admin/benchmark?tab=questions">
                Open benchmark questions
              </Link>
            </>
          )}
        </p>
      )}
      {mutation.isError && <p role="alert">{chatLogReviewError(mutation.error)}</p>}
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

export function ReviewAction({
  entry,
  canTransfer,
}: {
  entry: ChatLogEntry;
  canTransfer: boolean;
}) {
  const status = entry.review?.status;
  if (status !== "content_gap" && status !== "benchmark_candidate") return null;
  if (status === "benchmark_candidate" && entry.review?.benchmark_question_id != null) return null;

  const isContentGap = status === "content_gap";
  const Icon = isContentGap ? BookOpen : ClipboardList;
  const canOpen = isContentGap || canTransfer;
  const action = isContentGap ? "Improve knowledge" : "Add to benchmark";
  const path = isContentGap ? "knowledge" : "benchmark-question";

  return (
    <section className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-primary/20 bg-primary/5 p-5">
      <div className="flex items-start gap-3">
        <Icon aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
        <div>
          <h3 className="font-semibold">
            {isContentGap ? "Close the knowledge gap" : "Create a benchmark question"}
          </h3>
          <p className="mt-1 text-sm text-muted-foreground">
            {isContentGap
              ? "Edit source chunks, add missing information or create a new knowledge note."
              : canTransfer
                ? "Review the question and conversation history, then choose a dataset."
                : "An administrator can create the question from this candidate."}
          </p>
        </div>
      </div>
      {canOpen && (
        <Button variant="outline" asChild>
          <Link to={`/admin/chat-logs/${entry.id}/${path}`}>
            <Plus aria-hidden="true" /> {action}
          </Link>
        </Button>
      )}
    </section>
  );
}
