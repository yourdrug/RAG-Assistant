"use client";
import { useQueryClient } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import { useLocation, useSearchParams } from "react-router-dom";
import { useCurrentUser } from "@/shared/api/hooks/use-auth";
import {
  type ChatLogsParams,
  downloadChatLogs,
  useChatLog,
  useChatLogs,
} from "@/shared/api/hooks/use-chat-logs";
import { queryKeys } from "@/shared/api/query-keys";
import type { AnswerOutcome, ChatLogEntry, FeedbackReason, ReviewStatus } from "@/shared/api/types";
import { formatDateTime } from "@/shared/lib/date-time";
import { cn } from "@/shared/lib/utils";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { FEEDBACK_REASON_LABELS, OUTCOME_LABELS } from "./chat-log-labels";
import { REVIEW_STATUS_LABELS, ReviewAction, ReviewForm } from "./chat-log-review";
import { ChatLogStatsPanel } from "./chat-log-stats";
import { ChatLogTrace } from "./chat-log-trace";

const outcomes: AnswerOutcome[] = [
  "generated",
  "grounded",
  "cache_hit",
  "not_found",
  "out_of_domain",
  "input_blocked",
  "output_blocked",
  "error",
  "cancelled",
];

function LogCard({ entry, canTransfer }: { entry: ChatLogEntry; canTransfer: boolean }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"answer" | "trace">("answer");
  const { data, isLoading, isError } = useChatLog(entry.id, open);
  const log = data?.entry ?? entry;
  const rating = entry.feedback?.rating;
  return (
    <details
      className={cn(
        "group rounded-xl border p-5 transition-colors",
        rating === "up" && "border-emerald-500/30 bg-emerald-500/5 hover:border-emerald-500/50",
        rating === "down" && "border-rose-500/30 bg-rose-500/5 hover:border-rose-500/50",
        !rating && "bg-card hover:border-muted-foreground/40",
      )}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer list-none flex flex-wrap items-center gap-2.5">
        <Badge variant={entry.outcome === "error" ? "destructive" : "secondary"}>
          {OUTCOME_LABELS[entry.outcome]}
        </Badge>
        {entry.feedback && (
          <Badge variant={rating === "up" ? "success" : "destructive"}>
            {entry.feedback.rating === "down" ? "Negative" : "Positive"}
            {entry.feedback.comment ? " · Comment" : ""}
          </Badge>
        )}
        {entry.review && entry.review.status !== "new" && (
          <Badge variant="outline">{REVIEW_STATUS_LABELS[entry.review.status]}</Badge>
        )}
        <span className="order-first flex w-full gap-4 break-words text-base font-medium leading-relaxed">
          <span className="min-w-0 flex-1">{entry.question}</span>
          <ChevronDown
            aria-hidden="true"
            className="mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
          />
        </span>
        <span className="text-xs text-muted-foreground">
          {formatDateTime(entry.creation_date)} · {entry.user_email ?? `#${entry.user_id ?? "—"}`} ·{" "}
          {entry.latency_ms ?? "—"} ms
        </span>
      </summary>
      {open && (
        <div className="mt-5 space-y-6 border-t pt-5">
          {isLoading && <p>Loading answer…</p>}
          {isError && <p role="alert">Could not load this answer.</p>}
          <div className="flex gap-2" role="tablist" aria-label="Answer details">
            <Button
              role="tab"
              aria-selected={tab === "answer"}
              variant={tab === "answer" ? "default" : "outline"}
              onClick={() => setTab("answer")}
            >
              Answer
            </Button>
            <Button
              role="tab"
              aria-selected={tab === "trace"}
              variant={tab === "trace" ? "default" : "outline"}
              onClick={() => setTab("trace")}
            >
              Trace
            </Button>
          </div>
          {tab === "trace" ? (
            <ChatLogTrace key={entry.id} entry={log} />
          ) : (
            <>
              {data && data.context.length > 0 && (
                <details>
                  <summary className="cursor-pointer text-sm">
                    Conversation history ({data.context.length})
                  </summary>
                  <ol className="mt-2 space-y-2 text-sm">
                    {data.context.map((turn, index) => (
                      <li key={turn.id ?? index} className="rounded bg-muted/50 p-2">
                        <span className="text-xs text-muted-foreground">
                          {turn.role === "user" ? "Question" : "Answer"}
                        </span>
                        <p className="whitespace-pre-wrap">{turn.content}</p>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
              <p className="whitespace-pre-wrap">{log.answer || "No answer delivered"}</p>
              {log.feedback && (
                <div
                  className={`space-y-3 rounded-lg border p-4 text-sm ${log.feedback.rating === "down" ? "border-rose-500/30 bg-rose-500/5" : "border-emerald-500/30 bg-emerald-500/5"}`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-muted-foreground">User feedback</span>
                    <Badge
                      variant="outline"
                      className={
                        log.feedback.rating === "down"
                          ? "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300"
                          : "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                      }
                    >
                      {log.feedback.rating === "down" ? "Negative" : "Positive"}
                    </Badge>
                  </div>
                  {log.feedback.reasons.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-muted-foreground">Reasons</span>
                      {log.feedback.reasons.map((reason) => (
                        <Badge
                          key={reason}
                          variant="outline"
                          className="border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-300"
                        >
                          {FEEDBACK_REASON_LABELS[reason]}
                        </Badge>
                      ))}
                    </div>
                  )}
                  {log.feedback.comment && (
                    <div className="border-t border-amber-500/15 pt-3">
                      <p className="mb-1 text-xs font-medium text-muted-foreground">Comment</p>
                      <p className="whitespace-pre-wrap leading-relaxed">{log.feedback.comment}</p>
                    </div>
                  )}
                </div>
              )}
              <ReviewForm key={log.id} entry={log} canTransfer={canTransfer} />
              <ReviewAction entry={log} canTransfer={canTransfer} />
              <details className="rounded-lg border p-4">
                <summary className="cursor-pointer text-sm font-medium">Technical details</summary>
                <dl className="mt-4 grid grid-cols-1 gap-x-8 gap-y-3 text-sm sm:grid-cols-[minmax(180px,1fr)_2fr] [&_dt]:text-muted-foreground [&_dd]:break-words">
                  <dt>Space / pack</dt>
                  <dd>
                    {log.space_id} / {log.pack_key ?? "—"}
                  </dd>
                  <dt>Query domain / breadth</dt>
                  <dd>
                    {log.domain ?? "—"} / {log.breadth ?? "—"} (requested:{" "}
                    {log.depth_requested ?? "auto"})
                    <p className="mt-1 text-xs text-muted-foreground">
                      Classified from the question; source documents may have a different domain.
                    </p>
                  </dd>
                  <dt>Answer style / finish reason</dt>
                  <dd>
                    {log.answer_style ?? "—"} / {log.finish_reason ?? "—"}
                  </dd>
                  <dt>Model / profile</dt>
                  <dd>
                    {log.model_used ?? "—"} / {log.evidence_profile ?? "—"}
                  </dd>
                  <dt>Confidence / error code</dt>
                  <dd>
                    {log.confidence ?? "—"} / {log.error_code ?? "—"}
                  </dd>
                  <dt>Sources / candidates / top rerank</dt>
                  <dd>
                    {log.sources_count ?? "—"} / {log.candidates_count ?? "—"} /{" "}
                    {log.top_rerank_score ?? "—"}
                  </dd>
                  <dt>Tokens: input / output / auxiliary</dt>
                  <dd>
                    {log.input_tokens ?? "—"} / {log.output_tokens ?? "—"} / {log.aux_tokens ?? "—"}
                  </dd>
                  <dt>TTFT / retrieval / rerank / generation, ms</dt>
                  <dd>
                    {log.ttft_ms ?? "—"} / {log.retrieval_ms ?? "—"} / {log.rerank_ms ?? "—"} /{" "}
                    {log.generation_ms ?? "—"}
                  </dd>
                  <dt>Message / request</dt>
                  <dd>
                    {log.message_id ?? "—"} / {log.request_id ?? "—"}
                  </dd>
                  <dt>Question date</dt>
                  <dd>{log.as_of_date ?? "—"}</dd>
                  <dt>Condensed / refined query</dt>
                  <dd>
                    {log.condensed_query ?? "—"} / {log.refined_query ?? "—"}
                  </dd>
                  <dt>Subqueries</dt>
                  <dd>{log.sub_queries?.join("; ") || "—"}</dd>
                </dl>
              </details>
              <div className="flex flex-wrap gap-3 text-sm">
                {log.sources?.map((source, index) =>
                  source.document_id ? (
                    <a
                      className="underline"
                      key={`${source.document_id}-${index}`}
                      href={`/documents/${source.document_id}`}
                    >
                      {source.source}
                    </a>
                  ) : (
                    <span key={`${source.source}-${index}`}>{source.source}</span>
                  ),
                )}
              </div>
              <details>
                <summary className="cursor-pointer">
                  Configuration: {log.config_hash ?? "no snapshot"}
                </summary>
                <p className="text-sm">Process version: {log.config_version ?? "—"}</p>
                <p>Changes from current configuration</p>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(data?.snapshot.diff ?? "Comparison unavailable", null, 2)}
                </pre>
                <p>Configuration at answer time</p>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(data?.snapshot.data, null, 2)}
                </pre>
              </details>
            </>
          )}
        </div>
      )}
    </details>
  );
}

function isoDateTime(value: string | null) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : "";
}

function dateTimeInput(value: string) {
  if (!value) return "";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "";
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

export function AdminChatLogsPage() {
  const location = useLocation();
  return <ChatLogsPage key={location.search} />;
}

function ChatLogsPage() {
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState({
    search: "",
    domain: "",
    space_id: searchParams.get("space_id") ?? "",
    outcome: String(outcomes.find((outcome) => outcome === searchParams.get("outcome")) ?? ""),
    error_code: searchParams.get("error_code") ?? "",
    model: searchParams.get("model") ?? "",
    breadth: "",
    answer_style: "",
    finish_reason: "",
    config_hash: "",
    rating: "",
    reason: searchParams.get("reason") ?? "",
    has_comment: "",
    review_status: "",
    user_id: "",
    date_from: isoDateTime(searchParams.get("date_from")),
    date_to: isoDateTime(searchParams.get("date_to")),
  });
  const [sortBy, setSortBy] = useState<"date" | "latency" | "confidence">("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const [needsReview, setNeedsReview] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const { data: currentUser } = useCurrentUser();
  const isAdmin = currentUser?.role === "admin";
  const pageSize = 20;
  const params: ChatLogsParams = {
    ...filters,
    needs_review: needsReview,
    space_id: filters.space_id ? Number(filters.space_id) : undefined,
    user_id: filters.user_id ? Number(filters.user_id) : undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit: pageSize,
    offset: page * pageSize,
  };
  const { data, isFetching, isError } = useChatLogs(params);
  const update = (key: keyof typeof filters, value: string) => {
    setFilters((previous) => ({ ...previous, [key]: value }));
    setPage(0);
  };
  const showQueue = (queue: "triage" | "candidates" | "all") => {
    setNeedsReview(queue === "triage");
    setFilters((previous) => ({
      ...previous,
      review_status: queue === "candidates" ? "benchmark_candidate" : "",
    }));
    setSortBy("date");
    setSortOrder("desc");
    setPage(0);
  };
  const exportLogs = (format: "csv" | "json") => {
    setExportError(null);
    downloadChatLogs(params, format).catch(() => setExportError("Could not export logs."));
  };
  return (
    <div className="mx-auto max-w-screen-2xl space-y-8 p-4 sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-1 text-xs font-medium uppercase tracking-widest text-muted-foreground">
            Quality monitoring
          </p>
          <h1 className="text-3xl font-semibold tracking-tight">Answer logs</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Explore responses, investigate issues and build better benchmarks.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => exportLogs("csv")}>
            CSV
          </Button>
          <Button variant="outline" onClick={() => exportLogs("json")}>
            JSON
          </Button>
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: queryKeys.chatLogs.all })}
            disabled={isFetching}
          >
            Refresh
          </Button>
        </div>
      </div>
      {exportError && <p role="alert">{exportError}</p>}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Review queues">
        <Button
          variant={needsReview ? "default" : "outline"}
          aria-pressed={needsReview}
          onClick={() => showQueue("triage")}
          title="Negative feedback, no answer or errors awaiting review"
        >
          Needs review
        </Button>
        <Button
          variant={filters.review_status === "benchmark_candidate" ? "default" : "outline"}
          aria-pressed={filters.review_status === "benchmark_candidate"}
          onClick={() => showQueue("candidates")}
        >
          Benchmark candidates
        </Button>
        <Button
          variant={
            !needsReview && filters.review_status !== "benchmark_candidate" ? "default" : "outline"
          }
          aria-pressed={!needsReview && filters.review_status !== "benchmark_candidate"}
          onClick={() => showQueue("all")}
        >
          All entries
        </Button>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <Label className="flex min-w-0 flex-1 flex-col gap-2.5 sm:max-w-xl">
          <span>Search responses</span>
          <Input
            type="search"
            placeholder="Search questions and answers…"
            value={filters.search}
            onChange={(event) => update("search", event.target.value)}
          />
        </Label>
        <Button
          variant="ghost"
          disabled={!Object.values(filters).some(Boolean)}
          onClick={() => {
            setFilters(
              (previous) =>
                Object.fromEntries(
                  Object.keys(previous).map((key) => [key, ""]),
                ) as typeof previous,
            );
            setPage(0);
          }}
        >
          Clear filters
        </Button>
      </div>
      <details className="rounded-xl border bg-card p-5" open={searchParams.size > 0}>
        <summary className="cursor-pointer text-sm font-semibold">
          Advanced filters{" "}
          <span className="ml-2 font-normal text-muted-foreground">
            {Object.entries(filters).filter(([key, value]) => key !== "search" && value).length}{" "}
            active
          </span>
        </summary>
        <div className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {(
            [
              ["domain", "Domain"],
              ["space_id", "Space ID"],
              ["model", "Model"],
              ["error_code", "Error code"],
              ["config_hash", "Configuration hash"],
              ["user_id", "User ID"],
              ["date_from", "From"],
              ["date_to", "To"],
            ] as const
          ).map(([key, label]) => (
            <Label key={key} className="flex min-w-0 flex-col gap-2.5">
              <span>{label}</span>
              {key === "date_from" || key === "date_to" ? (
                <Input
                  type="datetime-local"
                  step={1}
                  value={dateTimeInput(filters[key])}
                  onChange={(event) => update(key, isoDateTime(event.target.value))}
                />
              ) : (
                <Input
                  type={key.endsWith("_id") ? "number" : "text"}
                  value={filters[key]}
                  onChange={(event) => update(key, event.target.value)}
                />
              )}
            </Label>
          ))}
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Outcome</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.outcome}
              onChange={(event) => update("outcome", event.target.value)}
            >
              <option value="">All</option>
              {outcomes.map((outcome) => (
                <option key={outcome} value={outcome}>
                  {OUTCOME_LABELS[outcome]}
                </option>
              ))}
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Feedback</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.rating}
              onChange={(event) => update("rating", event.target.value)}
            >
              <option value="">All</option>
              <option value="down">Negative</option>
              <option value="up">Positive</option>
              <option value="none">Not rated</option>
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Feedback reason</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.reason}
              onChange={(event) => update("reason", event.target.value)}
            >
              <option value="">All</option>
              {(Object.keys(FEEDBACK_REASON_LABELS) as FeedbackReason[]).map((reason) => (
                <option key={reason} value={reason}>
                  {FEEDBACK_REASON_LABELS[reason]}
                </option>
              ))}
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Comment</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.has_comment}
              onChange={(event) => update("has_comment", event.target.value)}
            >
              <option value="">All</option>
              <option value="true">With comment</option>
              <option value="false">Without comment</option>
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Review status</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.review_status}
              onChange={(event) => update("review_status", event.target.value)}
            >
              <option value="">All</option>
              {(Object.keys(REVIEW_STATUS_LABELS) as ReviewStatus[]).map((status) => (
                <option key={status} value={status}>
                  {REVIEW_STATUS_LABELS[status]}
                </option>
              ))}
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Search breadth</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.breadth}
              onChange={(event) => update("breadth", event.target.value)}
            >
              <option value="">All</option>
              <option>narrow</option>
              <option>broad</option>
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Answer style</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.answer_style}
              onChange={(event) => update("answer_style", event.target.value)}
            >
              <option value="">All</option>
              <option>concise</option>
              <option>adaptive</option>
              <option>detailed</option>
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Finish reason</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={filters.finish_reason}
              onChange={(event) => update("finish_reason", event.target.value)}
            >
              <option value="">All</option>
              <option>stop</option>
              <option>length</option>
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Sort by</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={sortBy}
              onChange={(event) => {
                setSortBy(event.target.value as typeof sortBy);
                setPage(0);
              }}
            >
              <option value="date">Date</option>
              <option value="latency">Latency</option>
              <option value="confidence">Confidence</option>
            </select>
          </Label>
          <Label className="flex min-w-0 flex-col gap-2.5">
            <span>Order</span>
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={sortOrder}
              onChange={(event) => {
                setSortOrder(event.target.value as typeof sortOrder);
                setPage(0);
              }}
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </Label>
        </div>
      </details>
      <ChatLogStatsPanel params={params} />
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Responses</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {data?.total ?? 0} matching entries{isFetching ? " · Updating…" : ""}
          </p>
        </div>
        <span className="text-sm text-muted-foreground">Page {page + 1}</span>
      </div>
      {isError && <p role="alert">Could not load logs. Check your filters and try again.</p>}
      <div className="space-y-3" aria-busy={isFetching}>
        {isFetching && !data && (
          <p role="status" className="py-8 text-center text-sm text-muted-foreground">
            Loading responses…
          </p>
        )}
        {!isFetching && !isError && data?.logs.length === 0 && (
          <div className="rounded-xl border border-dashed p-12 text-center">
            <h3 className="font-medium">No matching responses</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              Try another queue or adjust your filters.
            </p>
          </div>
        )}
        {data?.logs.map((entry) => (
          <LogCard key={entry.id} entry={entry} canTransfer={isAdmin} />
        ))}
      </div>
      <div className="flex gap-4 items-center">
        <Button variant="outline" disabled={page === 0} onClick={() => setPage(page - 1)}>
          Previous
        </Button>
        <span>{page + 1}</span>
        <Button
          variant="outline"
          disabled={!data || (page + 1) * pageSize >= data.total}
          onClick={() => setPage(page + 1)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
