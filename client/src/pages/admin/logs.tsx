import {
  Activity,
  AlertCircle,
  ArrowDown,
  ChevronRight,
  Copy,
  ListFilter,
  Pause,
  Play,
  RefreshCw,
  ScrollText,
  Search,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";
import toast from "react-hot-toast";
import { useLogs } from "@/shared/api/hooks";
import { LOGS_REFRESH_INTERVAL_MS } from "@/shared/api/hooks/use-logs";
import type { LogEntry } from "@/shared/api/types";
import { formatDate, formatDateTime, formatTime } from "@/shared/lib/date-time";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Skeleton } from "@/shared/ui/skeleton";

const logLimit = 200;
const levelStyles: Record<string, string> = {
  INFO: "bg-sky-50 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  WARNING: "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  ERROR: "bg-rose-50 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  CRITICAL: "bg-rose-100 text-rose-900 dark:bg-rose-900 dark:text-rose-100",
  DEBUG: "bg-muted text-muted-foreground",
};
const filterLevels = ["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"] as const;
const eventColumns =
  "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 @min-[560px]:grid-cols-[6rem_5.75rem_minmax(0,1fr)_1rem]";

function LevelBadge({ value }: { value: string }) {
  const level = value?.toUpperCase() || "INFO";
  return (
    <span
      className={cn(
        "inline-flex max-w-full items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-semibold leading-none",
        levelStyles[level] ?? "bg-muted text-muted-foreground",
      )}
    >
      <span className="size-1.5 shrink-0 rounded-full bg-current" />
      <span className="truncate">{level}</span>
    </span>
  );
}

async function copyText(value: string, label: string) {
  try {
    await navigator.clipboard.writeText(value);
    toast.success(`${label} copied`);
  } catch {
    toast.error(`Could not copy ${label.toLowerCase()}`);
  }
}

function EventDetails({ log }: { log: LogEntry }) {
  const requestId = log.request_id && log.request_id !== "-" ? log.request_id : null;
  const source = log.filename
    ? `${log.filename}${log.lineno != null ? `:${log.lineno}` : ""}`
    : null;

  return (
    <>
      <div className="border-b border-border/70 p-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="text-xs font-semibold text-muted-foreground">Message</h3>
          <Button variant="ghost" size="sm" onClick={() => void copyText(log.message, "Message")}>
            <Copy aria-hidden="true" /> Copy
          </Button>
        </div>
        <p className="rounded-lg border border-border/70 bg-muted/50 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap wrap-anywhere">
          {log.message}
        </p>
      </div>
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-x-4 gap-y-3 p-4 text-xs">
        <dt className="pt-0.5 text-muted-foreground">Time</dt>
        <dd className="font-mono leading-5">
          <time dateTime={log.timestamp}>{formatDateTime(log.timestamp)}</time>
        </dd>
        <dt className="pt-0.5 text-muted-foreground">Level</dt>
        <dd>
          <LevelBadge value={log.level} />
        </dd>
        <dt className="pt-0.5 text-muted-foreground">Logger</dt>
        <dd className="font-mono leading-5 wrap-anywhere">{log.logger || "—"}</dd>
        <dt className="pt-0.5 text-muted-foreground">User ID</dt>
        <dd className="font-mono leading-5">{log.user_id ?? "—"}</dd>
        <dt className="pt-0.5 text-muted-foreground">Login</dt>
        <dd className="font-mono leading-5 wrap-anywhere">{log.user_login ?? "—"}</dd>
        <dt className="pt-0.5 text-muted-foreground">Source</dt>
        <dd className="font-mono leading-5 wrap-anywhere">{source ?? "—"}</dd>
        <dt className="pt-0.5 text-muted-foreground">Request ID</dt>
        <dd className="flex min-w-0 items-start gap-1 font-mono leading-5">
          <span className="min-w-0 wrap-anywhere">{requestId ?? "—"}</span>
          {requestId && (
            <Button
              variant="ghost"
              size="icon"
              className="size-6 shrink-0 [&_svg]:size-3.5"
              aria-label="Copy request ID"
              onClick={() => void copyText(requestId, "Request ID")}
            >
              <Copy aria-hidden="true" />
            </Button>
          )}
        </dd>
      </dl>
    </>
  );
}

export function LogsPage() {
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const { data, dataUpdatedAt, isLoading, isError, isFetching, isPlaceholderData, refetch } =
    useLogs(
      { limit: logLimit, search: search || undefined, level: levelFilter || undefined },
      {
        autoRefresh,
      },
    );

  const entries = useMemo(() => {
    const seen = new Map<string, number>();
    return (data?.logs ?? []).map((log) => {
      // The API has no event ID. Keep selection stable when polling prepends records.
      const fingerprint = JSON.stringify([
        log.timestamp,
        log.level,
        log.logger,
        log.request_id,
        log.message,
        log.filename,
        log.lineno,
        log.user_id,
      ]);
      const occurrence = seen.get(fingerprint) ?? 0;
      seen.set(fingerprint, occurrence + 1);
      return { key: `${fingerprint}:${occurrence}`, log };
    });
  }, [data]);

  const selected = isPlaceholderData ? undefined : entries.find(({ key }) => key === selectedKey);
  const preview = selected ?? (isPlaceholderData ? undefined : entries[0]);
  const filtered = Boolean(search || levelFilter);
  const pending = isLoading || isPlaceholderData;

  function resetFilters() {
    setSearchInput("");
    setSearch("");
    setLevelFilter("");
    setSelectedKey(null);
  }

  return (
    <div className="@container flex min-h-full flex-col bg-muted/35 p-4 sm:p-6">
      <div className="mx-auto flex w-full max-w-[100rem] flex-1 flex-col gap-4 @min-[900px]:min-h-[40rem]">
        <header className="flex flex-wrap items-center justify-between gap-x-8 gap-y-5 rounded-xl border bg-card px-5 py-5 text-card-foreground shadow-xs sm:px-7 sm:py-6">
          <div className="flex items-center gap-4">
            <div className="grid size-11 shrink-0 place-items-center rounded-xl bg-muted text-foreground sm:size-12">
              <ScrollText aria-hidden="true" className="size-5" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight sm:text-[1.75rem]">Actions</h1>
              <p className="mt-1 text-sm text-muted-foreground">Recent application activity</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 pr-2 text-xs font-medium text-muted-foreground">
              <span
                className={cn(
                  "size-2 rounded-full",
                  autoRefresh ? "bg-teal-600 dark:bg-teal-400" : "bg-muted-foreground",
                )}
              />
              {autoRefresh ? `Live · ${LOGS_REFRESH_INTERVAL_MS / 1000}s` : "Paused"}
            </span>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setAutoRefresh(!autoRefresh);
                if (!autoRefresh) void refetch();
              }}
            >
              {autoRefresh ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
              {autoRefresh ? "Pause" : "Resume"}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => void refetch()}
              disabled={isFetching}
            >
              <RefreshCw
                aria-hidden="true"
                className={isFetching ? "motion-safe:animate-spin" : ""}
              />
              Refresh
            </Button>
          </div>
        </header>

        <section
          aria-label="Action log"
          className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border bg-card shadow-xs"
        >
          <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-b p-3 sm:p-4">
            <form
              role="search"
              className="flex min-w-0 flex-1 basis-full items-center gap-2 @min-[760px]:basis-auto"
              onSubmit={(event) => {
                event.preventDefault();
                setSearch(searchInput.trim());
                setSelectedKey(null);
              }}
            >
              <div className="relative min-w-0 flex-1">
                <Search
                  aria-hidden="true"
                  className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
                />
                <Input
                  aria-label="Search messages"
                  placeholder="Search messages…"
                  value={searchInput}
                  onChange={(event) => {
                    setSearchInput(event.target.value);
                    if (!event.target.value && search) {
                      setSearch("");
                      setSelectedKey(null);
                    }
                  }}
                  className="bg-background pr-9 pl-9"
                />
                {searchInput && (
                  <Button
                    type="button"
                    aria-label="Clear search"
                    variant="ghost"
                    size="icon"
                    className="absolute top-1/2 right-1 size-7 -translate-y-1/2"
                    onClick={() => {
                      setSearchInput("");
                      setSearch("");
                      setSelectedKey(null);
                    }}
                  >
                    <X aria-hidden="true" />
                  </Button>
                )}
              </div>
              <Button type="submit" variant="outline">
                Search
              </Button>
            </form>
            <label className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
              <ListFilter aria-hidden="true" className="size-4" />
              Level
              <select
                value={levelFilter}
                onChange={(event) => {
                  setLevelFilter(event.target.value);
                  setSelectedKey(null);
                }}
                className="h-9 rounded-md border border-input bg-background px-2.5 text-sm font-medium text-foreground focus-visible:outline-2 focus-visible:outline-ring"
              >
                <option value="">All levels</option>
                {filterLevels.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </label>
            {filtered && (
              <Button type="button" variant="ghost" onClick={resetFilters}>
                Reset filters
              </Button>
            )}
          </div>
          {isError && data && (
            <p role="status" className="border-b bg-amber-500/10 px-4 py-2 text-xs">
              Refresh failed. Showing the last loaded events.
            </p>
          )}

          <div className="grid min-h-0 flex-1 @min-[900px]:grid-cols-[minmax(0,1fr)_minmax(17rem,30%)]">
            <div className="flex min-h-0 min-w-0 flex-col">
              <div className="flex items-center justify-between gap-3 px-4 py-3">
                <h2 className="text-sm font-semibold">Event stream</h2>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {pending ? "Loading…" : data && `${entries.length} shown`}
                </span>
              </div>
              <div
                className={cn(
                  eventColumns,
                  "hidden border-y bg-muted/45 px-4 py-2 text-xs font-medium text-muted-foreground @min-[560px]:grid",
                )}
                aria-hidden="true"
              >
                <span className="flex items-center gap-1">
                  Time <ArrowDown className="size-3" />
                </span>
                <span>Level</span>
                <span>Event</span>
              </div>
              <div
                className="scrollbar-thin max-h-[34rem] min-h-0 flex-1 overflow-y-auto overscroll-y-contain @min-[900px]:max-h-[clamp(24rem,calc(100dvh-25.75rem),64rem)]"
                aria-busy={pending}
              >
                {pending ? (
                  <div className="space-y-2 px-4 py-3">
                    {["100%", "85%", "95%", "75%", "90%"].map((width) => (
                      <Skeleton key={width} className="h-12" style={{ width }} />
                    ))}
                  </div>
                ) : entries.length > 0 ? (
                  <ul aria-label="Events" className="divide-y divide-border/70">
                    {entries.map(({ key, log }) => (
                      <li key={key}>
                        <button
                          type="button"
                          aria-pressed={selected?.key === key}
                          onClick={() => setSelectedKey(selectedKey === key ? null : key)}
                          className={cn(
                            eventColumns,
                            "relative w-full gap-y-1 px-4 py-2.5 text-left transition-colors hover:bg-muted/65 focus-visible:z-10 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring",
                            selected?.key === key
                              ? "bg-teal-500/8 shadow-[inset_2px_0] shadow-teal-600 dark:shadow-teal-400"
                              : preview?.key === key && "@min-[900px]:bg-teal-500/8",
                          )}
                        >
                          <time
                            dateTime={log.timestamp}
                            className="flex items-baseline gap-2 font-mono text-xs tabular-nums @min-[560px]:block"
                          >
                            {formatTime(log.timestamp)}
                            <span className="block text-[11px] text-muted-foreground">
                              {formatDate(log.timestamp)}
                            </span>
                          </time>
                          <span>
                            <LevelBadge value={log.level} />
                          </span>
                          <span className="col-span-full min-w-0 @min-[560px]:col-span-1">
                            <span className="block truncate text-sm">{log.message}</span>
                            <span className="mt-0.5 block truncate font-mono text-[11px] text-muted-foreground">
                              {log.logger}
                              {log.user_id != null && (
                                <>
                                  {" · User #"}
                                  {log.user_id}
                                  {log.user_login && ` · ${log.user_login}`}
                                </>
                              )}
                            </span>
                          </span>
                          <ChevronRight
                            aria-hidden="true"
                            className="hidden size-4 text-muted-foreground @min-[560px]:block"
                          />
                        </button>
                        {selected?.key === key && (
                          <div className="border-t bg-muted/30 @min-[900px]:hidden">
                            <EventDetails log={log} />
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="flex min-h-56 flex-col items-center justify-center gap-2 px-4 py-8 text-center">
                    <span className="mb-1 grid size-10 place-items-center rounded-lg bg-muted text-muted-foreground">
                      {isError ? (
                        <AlertCircle aria-hidden="true" className="size-5" />
                      ) : (
                        <Activity aria-hidden="true" className="size-5" />
                      )}
                    </span>
                    <p className="text-sm font-semibold">
                      {isError
                        ? "Could not load events"
                        : filtered
                          ? "No matching events"
                          : "No events yet"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {isError
                        ? "Try refreshing the event stream."
                        : filtered
                          ? "Try a different search or level."
                          : "New actions will appear here."}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <aside
              aria-label="Event details"
              className="hidden min-w-0 flex-col border-l bg-muted/20 @min-[900px]:flex"
            >
              <h2 className="border-b border-border/70 px-4 py-3 text-sm font-semibold">
                Event details
              </h2>
              <div className="scrollbar-thin max-h-[clamp(26rem,calc(100dvh-23.75rem),66rem)] overflow-y-auto overscroll-y-contain">
                {preview ? (
                  <EventDetails log={preview.log} />
                ) : (
                  <p className="px-4 py-5 text-sm text-muted-foreground">No event to inspect.</p>
                )}
              </div>
            </aside>
          </div>

          <footer className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-t bg-muted/40 px-4 py-2.5 text-xs text-muted-foreground">
            <span>
              {entries.length === logLimit && !pending
                ? `Latest ${logLimit} events · Newest first`
                : "Newest first"}
            </span>
            {dataUpdatedAt > 0 && (
              <span className="tabular-nums">Updated {formatTime(dataUpdatedAt)}</span>
            )}
          </footer>
        </section>
      </div>
    </div>
  );
}
