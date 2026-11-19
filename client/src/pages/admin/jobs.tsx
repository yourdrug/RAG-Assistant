import {
  AlertCircle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  ListFilter,
  RefreshCw,
  RotateCcw,
  X,
} from "lucide-react";
import { Fragment, useState } from "react";
import { useJobAction, useJobStats, useJobs } from "@/shared/api/hooks";
import type { JobResponse } from "@/shared/api/types";
import { formatDateTime } from "@/shared/lib/date-time";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/button";
import { Card, CardContent } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

const pageSize = 50;
const statuses = ["pending", "running", "done", "failed", "cancelled"] as const;
const statusStyles: Record<JobResponse["status"], string> = {
  pending: "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  running: "bg-sky-50 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  done: "bg-teal-50 text-teal-800 dark:bg-teal-950 dark:text-teal-300",
  failed: "bg-rose-50 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  cancelled: "bg-muted text-muted-foreground",
};

function formatDuration(start?: string | null, end?: string | null) {
  if (!start) return "—";
  const ms = (end ? new Date(end).getTime() : Date.now()) - new Date(start).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 60000).toFixed(1)} min`;
}

function JobDetails({ job }: { job: JobResponse }) {
  const metadata = [
    ["Attempt", job.attempt],
    ["Request ID", job.request_id || "—"],
    ["Created", formatDateTime(job.creation_date)],
    ["Started", formatDateTime(job.started_at)],
    ["Finished", formatDateTime(job.finished_at)],
    ["Duration", formatDuration(job.started_at, job.finished_at)],
  ];

  return (
    <div id={`job-details-${job.id}`} className="space-y-4">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 @min-[38rem]:grid-cols-3">
        {metadata.map(([label, value]) => (
          <div key={label} className="min-w-0 space-y-1">
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="font-mono text-xs leading-relaxed wrap-anywhere">{value}</dd>
          </div>
        ))}
      </dl>
      {job.error_message && (
        <div className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-3">
          <p className="flex items-center gap-2 text-xs font-semibold text-rose-800 dark:text-rose-300">
            <AlertCircle aria-hidden="true" className="size-3.5" />
            Error
          </p>
          <p className="mt-2 font-mono text-xs leading-relaxed whitespace-pre-wrap wrap-anywhere">
            {job.error_message}
          </p>
        </div>
      )}
    </div>
  );
}

export function JobsPage() {
  const [page, setPage] = useState(0);
  const [status, setStatus] = useState("");
  const [expandedJobId, setExpandedJobId] = useState<number | null>(null);
  const retry = useJobAction("retry");
  const cancel = useJobAction("cancel");
  const { data, isLoading, isFetching, isError, refetch } = useJobs({
    limit: pageSize,
    offset: page * pageSize,
    status: status || undefined,
  });
  const { data: statsData } = useJobStats();
  const jobs = data?.jobs ?? [];
  const total = data?.total ?? 0;
  const first = page * pageSize + 1;

  function changePage(next: number) {
    setExpandedJobId(null);
    setPage(next);
  }

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Jobs</h1>
          <p className="text-muted-foreground">Background task monitoring</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void refetch()} disabled={isFetching}>
          <RefreshCw aria-hidden="true" className={isFetching ? "motion-safe:animate-spin" : ""} />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{statsData?.total ?? "—"}</div>
            <p className="text-xs text-muted-foreground">Total</p>
          </CardContent>
        </Card>
        {statuses.map((value) => (
          <Card key={value}>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold">{statsData?.by_status?.[value] ?? 0}</div>
              <p className="text-xs text-muted-foreground capitalize">{value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <section
        aria-labelledby="jobs-table-title"
        className="@container overflow-hidden rounded-xl border bg-card shadow-xs"
      >
        <header className="flex flex-wrap items-center justify-between gap-3 border-b px-3 py-3.5 text-card-foreground @min-[32rem]:px-4">
          <div className="flex items-center gap-2.5">
            <span className="grid size-8 place-items-center rounded-md bg-muted text-foreground">
              <Clock aria-hidden="true" className="size-4" />
            </span>
            <h2 id="jobs-table-title" className="text-sm font-semibold">
              Recent jobs
            </h2>
          </div>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <ListFilter aria-hidden="true" className="size-4" />
            <span className="sr-only">Job status</span>
            <select
              value={status}
              className="h-8 rounded-md border border-input bg-background px-2 text-xs font-medium text-foreground focus-visible:outline-2 focus-visible:outline-ring"
              onChange={(event) => {
                setStatus(event.target.value);
                changePage(0);
              }}
            >
              <option value="">All statuses</option>
              {statuses.map((value) => (
                <option key={value} value={value}>
                  {value[0].toUpperCase() + value.slice(1)}
                </option>
              ))}
            </select>
          </label>
        </header>
        {isError && data && (
          <p role="status" className="border-b bg-amber-500/10 px-4 py-2 text-xs">
            Refresh failed. Showing the last loaded jobs.
          </p>
        )}
        <Table aria-labelledby="jobs-table-title" aria-busy={isLoading} className="table-fixed">
          <TableHeader>
            <TableRow className="bg-muted/45 hover:bg-muted/45">
              <TableHead scope="col" className="pl-3 text-xs @min-[32rem]:pl-4">
                Job
              </TableHead>
              <TableHead
                scope="col"
                className="w-[5.65rem] px-1.5 text-xs @min-[32rem]:w-28 @min-[32rem]:px-2"
              >
                Status
              </TableHead>
              <TableHead scope="col" className="hidden w-44 text-xs @min-[38rem]:table-cell">
                Created / duration
              </TableHead>
              <TableHead
                scope="col"
                className="w-[4.75rem] pr-3 text-right text-xs @min-[32rem]:w-40 @min-[32rem]:pr-4"
              >
                Actions
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              ["100%", "90%", "95%", "85%"].map((width) => (
                <TableRow key={width}>
                  <TableCell colSpan={4} className="px-4 py-3">
                    <Skeleton className="h-7" style={{ width }} />
                  </TableCell>
                </TableRow>
              ))
            ) : jobs.length === 0 ? (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={4} className="px-4 py-12 text-center">
                  <div className="mx-auto mb-3 grid size-9 place-items-center rounded-lg bg-muted text-muted-foreground">
                    {isError ? (
                      <AlertCircle aria-hidden="true" className="size-4" />
                    ) : (
                      <Clock aria-hidden="true" className="size-4" />
                    )}
                  </div>
                  <p className="font-semibold">
                    {isError ? "Could not load jobs" : "No jobs found"}
                  </p>
                  {isError && (
                    <p className="mt-1 text-xs text-muted-foreground">Try refreshing the list.</p>
                  )}
                </TableCell>
              </TableRow>
            ) : (
              jobs.map((job) => {
                const expanded = expandedJobId === job.id;
                const canCancel =
                  ["ingest", "ingest_file", "benchmark"].includes(job.job_type) &&
                  ["pending", "running"].includes(job.status);
                return (
                  <Fragment key={job.id}>
                    <TableRow className={cn(expanded && "bg-teal-500/8 hover:bg-teal-500/8")}>
                      <TableCell className="p-0">
                        <button
                          type="button"
                          aria-expanded={expanded}
                          aria-controls={expanded ? `job-details-${job.id}` : undefined}
                          onClick={() => setExpandedJobId(expanded ? null : job.id)}
                          className="flex w-full items-center gap-1.5 py-2.5 pr-1 pl-3 text-left focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring @min-[32rem]:gap-2.5 @min-[32rem]:pl-4"
                        >
                          {expanded ? (
                            <ChevronDown
                              aria-hidden="true"
                              className="size-3.5 shrink-0 text-muted-foreground"
                            />
                          ) : (
                            <ChevronRight
                              aria-hidden="true"
                              className="size-3.5 shrink-0 text-muted-foreground"
                            />
                          )}
                          <span className="min-w-0">
                            <span className="block text-sm leading-snug font-medium wrap-anywhere first-letter:uppercase">
                              {job.job_type.replaceAll("_", " ")}
                            </span>
                            <span className="mt-0.5 block truncate font-mono text-[11px] text-muted-foreground">
                              #{job.id}
                            </span>
                          </span>
                        </button>
                      </TableCell>
                      <TableCell className="px-1.5 @min-[32rem]:px-2">
                        <span
                          className={cn(
                            "inline-flex max-w-full items-center gap-1 rounded-md px-1.5 py-1 text-[11px] leading-none font-semibold capitalize @min-[32rem]:gap-1.5 @min-[32rem]:px-2",
                            statusStyles[job.status],
                          )}
                        >
                          <span className="size-1.5 shrink-0 rounded-full bg-current" />
                          {job.status}
                        </span>
                      </TableCell>
                      <TableCell className="hidden font-mono text-xs tabular-nums @min-[38rem]:table-cell">
                        <div>{formatDateTime(job.creation_date)}</div>
                        <div className="mt-0.5 text-muted-foreground">
                          {formatDuration(job.started_at, job.finished_at)}
                        </div>
                      </TableCell>
                      <TableCell className="pr-3 pl-0 text-right @min-[32rem]:pr-4">
                        <div className="flex flex-wrap justify-end gap-1">
                          {job.retryable && (
                            <Button
                              size="sm"
                              variant="outline"
                              disabled={retry.isPending}
                              onClick={() => retry.mutate(job.id)}
                              className="gap-1.5 px-2"
                            >
                              <RotateCcw aria-hidden="true" className="hidden @min-[32rem]:block" />
                              Retry
                            </Button>
                          )}
                          {canCancel && (
                            <Button
                              size="sm"
                              variant="outline"
                              disabled={cancel.isPending}
                              onClick={() => cancel.mutate(job.id)}
                              className="gap-1.5 px-2"
                            >
                              <X aria-hidden="true" className="hidden @min-[32rem]:block" />
                              Cancel
                            </Button>
                          )}
                          {!job.retryable && !canCancel && (
                            <span className="px-2 text-xs text-muted-foreground">—</span>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                    {expanded && (
                      <TableRow className="bg-muted/30 hover:bg-muted/30">
                        <TableCell colSpan={3} className="px-4 py-4 @min-[32rem]:px-7">
                          <JobDetails job={job} />
                        </TableCell>
                        <TableCell className="hidden @min-[38rem]:table-cell" />
                      </TableRow>
                    )}
                  </Fragment>
                );
              })
            )}
          </TableBody>
        </Table>
        <footer className="flex flex-wrap items-center justify-between gap-2 border-t bg-muted/40 px-3 py-2.5 @min-[32rem]:px-4">
          <span className="text-xs text-muted-foreground tabular-nums">
            {data &&
              (jobs.length
                ? `${first}–${first + jobs.length - 1} of ${total}`
                : `0 shown · ${total} total`)}
          </span>
          <nav aria-label="Jobs pagination" className="ml-auto flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={page === 0}
              onClick={() => changePage(page - 1)}
            >
              <ChevronLeft aria-hidden="true" /> Previous
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={(page + 1) * pageSize >= total}
              onClick={() => changePage(page + 1)}
            >
              Next <ChevronRight aria-hidden="true" />
            </Button>
          </nav>
        </footer>
      </section>
    </div>
  );
}
