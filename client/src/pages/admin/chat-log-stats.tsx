"use client";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { apiClient } from "@/shared/api/client";
import { type ChatLogsParams, useChatLogStats } from "@/shared/api/hooks/use-chat-logs";
import { useSpaces } from "@/shared/api/hooks/use-spaces";
import type { ChatLogGroupStats, ConfigSnapshot } from "@/shared/api/types";
import { formatDate } from "@/shared/lib/date-time";
import { Button } from "@/shared/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import { FEEDBACK_REASON_LABELS, OUTCOME_LABELS } from "./chat-log-labels";

const share = (part: number, total: number) =>
  total ? `${Math.round((part / total) * 100)}%` : "—";
const ms = (value: number | null) =>
  value == null ? "—" : `${Math.round(value).toLocaleString("en-US")} ms`;
const count = (value: number) => value.toLocaleString("en-US");

function Tile({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-xl border bg-card p-5">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-3 text-2xl font-semibold tracking-tight tabular-nums">{value}</p>
      <p className="mt-2 text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

function Counts({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, value]) => value), 1);
  return (
    <section className="rounded-xl border bg-card p-5">
      <h3 className="text-sm font-semibold">{title}</h3>
      <div className="mt-5 space-y-4">
        {!entries.length && (
          <p className="text-sm text-muted-foreground">No data for this selection.</p>
        )}
        {entries.map(([key, value]) => (
          <div key={key}>
            <div className="mb-2 flex justify-between gap-4 text-sm">
              <span className="break-words">{key}</span>
              <span className="tabular-nums text-muted-foreground">{count(value)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-slate-400"
                style={{ width: `${(value / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function GroupTable({
  title,
  rows,
  label,
  onSelect,
}: {
  title: string;
  rows: ChatLogGroupStats[];
  label: (row: ChatLogGroupStats) => string;
  onSelect?: (key: string) => void;
}) {
  return (
    <div className="overflow-hidden rounded-xl border bg-card">
      <div className="border-b px-5 py-4">
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-muted/30 text-xs text-muted-foreground">
            <tr>
              {["Value", "Responses", "Negative", "No answer", "Errors", "Median latency"].map(
                (label, index) => (
                  <th
                    key={label}
                    className={`whitespace-nowrap px-5 py-3 font-medium ${index ? "text-right" : "text-left"}`}
                  >
                    {label}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={String(row.key)}
                className={`relative border-t ${onSelect && row.key ? "hover:bg-muted/40 focus-within:bg-muted/40" : ""}`}
              >
                <td className="max-w-64 truncate px-5 py-4 font-medium" title={label(row)}>
                  {onSelect && row.key ? (
                    <button
                      type="button"
                      className="text-left text-primary underline underline-offset-4 after:absolute after:inset-0 focus-visible:outline-none focus-visible:after:rounded focus-visible:after:ring-2 focus-visible:after:ring-ring"
                      onClick={() => onSelect(String(row.key))}
                    >
                      {label(row)}
                      <span className="mt-1 block text-xs font-normal text-muted-foreground">
                        View saved configuration
                      </span>
                    </button>
                  ) : (
                    label(row)
                  )}
                </td>
                {[row.total, row.down, row.not_found, row.errors, ms(row.latency_p50)].map(
                  (value, index) => (
                    <td key={index} className="whitespace-nowrap px-5 py-4 text-right tabular-nums">
                      {value}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && (
          <p className="p-5 text-sm text-muted-foreground">No data for this selection.</p>
        )}
      </div>
    </div>
  );
}

function ConfigurationDetails({ hash, onClose }: { hash: string | null; onClose: () => void }) {
  const snapshot = useQuery({
    queryKey: ["chatLogs", "snapshot", hash],
    queryFn: async () =>
      (await apiClient.get<ConfigSnapshot>(`/admin/rag-config-snapshots/${hash}`)).data,
    enabled: Boolean(hash),
    retry: false,
  });
  const saved = snapshot.data?.data;
  const modelSettings = saved?.LLMSection as Record<string, unknown> | undefined;
  const spaceSettings = saved?.space as Record<string, unknown> | undefined;
  return (
    <Dialog
      open={Boolean(hash)}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Saved configuration</DialogTitle>
          <DialogDescription className="break-all">{hash}</DialogDescription>
        </DialogHeader>
        {snapshot.isPending && (
          <p role="status" className="text-sm text-muted-foreground">
            Loading configuration…
          </p>
        )}
        {snapshot.isError && (
          <p role="alert" className="text-sm text-destructive">
            Could not load this configuration. It may be unavailable or outside your access scope.
          </p>
        )}
        {snapshot.data && (
          <div className="space-y-6">
            <dl className="grid grid-cols-1 gap-4 rounded-lg border bg-muted/20 p-4 sm:grid-cols-2">
              <div>
                <dt className="text-xs text-muted-foreground">Model</dt>
                <dd className="mt-1 break-words text-sm font-medium">
                  {String(modelSettings?.llm_model ?? "—")}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Domain pack</dt>
                <dd className="mt-1 text-sm font-medium">
                  {String(spaceSettings?.pack_key ?? "—")}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Temperature</dt>
                <dd className="mt-1 text-sm font-medium">
                  {String(modelSettings?.llm_temperature ?? "—")}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Evidence profile</dt>
                <dd className="mt-1 break-words text-sm font-medium">
                  {String(saved?.evidence_profile ?? "—")}
                </dd>
              </div>
            </dl>
            <section className="space-y-3">
              <h3 className="font-medium">Changes from current settings</h3>
              {snapshot.data.diff === null ? (
                <p className="text-sm text-muted-foreground">Comparison unavailable.</p>
              ) : snapshot.data.diff.length === 0 ? (
                <p className="text-sm text-muted-foreground">Matches current settings.</p>
              ) : (
                <div className="overflow-x-auto rounded-lg border">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-muted/40">
                      <tr>
                        <th className="p-3">Setting</th>
                        <th className="p-3">Saved</th>
                        <th className="p-3">Current</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.data.diff.map((item) => (
                        <tr key={item.field} className="border-t align-top">
                          <td className="break-all p-3 font-mono text-xs">{item.field}</td>
                          <td className="break-all p-3 font-mono text-xs">
                            {JSON.stringify(item.saved)}
                          </td>
                          <td className="break-all p-3 font-mono text-xs">
                            {JSON.stringify(item.current)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
            <section className="space-y-3">
              <h3 className="font-medium">Settings at answer time</h3>
              {snapshot.data.data ? (
                Object.entries(snapshot.data.data).map(([section, value]) => (
                  <details key={section} className="rounded-lg border p-4">
                    <summary className="cursor-pointer text-sm font-medium">{section}</summary>
                    <pre className="mt-3 overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  </details>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">No saved settings.</p>
              )}
            </section>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Aggregates over the same filters and curator scope as the journal below. */
export function ChatLogStatsPanel({ params }: { params: ChatLogsParams }) {
  const { limit: _limit, offset: _offset, ...filters } = params;
  const { data, isLoading, isError } = useChatLogStats(filters);
  const { data: spaces } = useSpaces();
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [group, setGroup] = useState<"space" | "model" | "config">("space");
  if (isLoading)
    return (
      <div className="rounded-xl border p-6 text-sm text-muted-foreground" role="status">
        Loading overview…
      </div>
    );
  if (isError || !data) return <p role="alert">Could not load the overview.</p>;
  const reasons = Object.fromEntries(
    Object.entries(data.reasons).map(([key, value]) => [
      FEEDBACK_REASON_LABELS[key as keyof typeof FEEDBACK_REASON_LABELS] ?? key,
      value,
    ]),
  );
  const outcomes = Object.fromEntries(
    Object.entries(data.outcomes).map(([key, value]) => [
      OUTCOME_LABELS[key as keyof typeof OUTCOME_LABELS] ?? key,
      value,
    ]),
  );
  const series = [
    { key: "total", label: "Responses", color: "#64748b" },
    { key: "down", label: "Negative feedback", color: "#d97706" },
    { key: "not_found", label: "No answer", color: "#0891b2" },
    { key: "errors", label: "Errors", color: "#e11d48" },
  ];
  return (
    <section className="space-y-5" aria-label="Response overview">
      <div>
        <h2 className="text-lg font-semibold">Overview</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Metrics across all responses matching the current filters.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Tile
          label="Responses"
          value={count(data.total)}
          hint={`${share(data.cache_hits, data.total)} served from cache`}
        />
        <Tile
          label="Negative feedback"
          value={count(data.down)}
          hint={`${share(data.down, data.total)} of responses · ${count(data.up)} positive`}
        />
        <Tile
          label="Median response time"
          value={ms(data.latency_p50)}
          hint={`95th percentile: ${ms(data.latency_p95)}`}
        />
        <Tile
          label="Time to first token"
          value={ms(data.ttft_p50)}
          hint={`95th percentile: ${ms(data.ttft_p95)}`}
        />
      </div>
      <section className="rounded-xl border bg-card p-5 sm:p-6" aria-label="Daily response volume">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h3 className="font-semibold">Response activity</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Daily volume and signals that may need attention.
            </p>
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-2">
            {series.map((item) => (
              <span
                key={item.key}
                className="flex items-center gap-2 text-xs text-muted-foreground"
              >
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                {item.label}
              </span>
            ))}
          </div>
        </div>
        {data.by_day.length ? (
          <div className="mt-8 h-60 w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.by_day}
                barGap={3}
                barCategoryGap="28%"
                margin={{ top: 8, right: 8, bottom: 0, left: -16 }}
              >
                <CartesianGrid vertical={false} stroke="currentColor" strokeOpacity={0.08} />
                <XAxis
                  dataKey="key"
                  axisLine={false}
                  tickLine={false}
                  tickMargin={12}
                  tick={{ fontSize: 12, fill: "#94a3b8" }}
                  minTickGap={28}
                  tickFormatter={formatDate}
                />
                <YAxis
                  allowDecimals={false}
                  axisLine={false}
                  tickLine={false}
                  tickMargin={8}
                  tick={{ fontSize: 12, fill: "#94a3b8" }}
                />
                <Tooltip
                  position={{ x: 48, y: 8 }}
                  isAnimationActive={false}
                  labelFormatter={(value) => formatDate(String(value))}
                  cursor={false}
                  wrapperStyle={{ pointerEvents: "none", outline: "none", zIndex: 10 }}
                  content={({ active, payload, label }) =>
                    active && payload?.length ? (
                      <div className="pointer-events-none min-w-48 rounded-lg border bg-popover p-3 text-popover-foreground shadow-md">
                        <p className="mb-2 text-xs font-semibold">{formatDate(String(label))}</p>
                        <div className="space-y-1.5">
                          {payload.map((item) => (
                            <div
                              key={String(item.dataKey)}
                              className="flex items-center justify-between gap-6 text-xs"
                            >
                              <span className="flex items-center gap-2">
                                <span
                                  className="h-2 w-2 rounded-full"
                                  style={{ backgroundColor: item.color }}
                                />
                                {item.name}
                              </span>
                              <span className="font-medium tabular-nums">{item.value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null
                  }
                />
                {series.map((item) => (
                  <Bar
                    key={item.key}
                    dataKey={item.key}
                    name={item.label}
                    fill={item.color}
                    isAnimationActive={false}
                    maxBarSize={28}
                    radius={[4, 4, 0, 0]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="py-14 text-center text-sm text-muted-foreground">
            No activity for this selection.
          </p>
        )}
        <p className="mt-4 text-xs text-muted-foreground">
          Feedback, unanswered requests and errors are separate signals; they can overlap.
        </p>
      </section>
      <details className="rounded-xl border bg-card p-5">
        <summary className="cursor-pointer text-sm font-semibold">
          Diagnostics & breakdowns{" "}
          <span className="ml-2 font-normal text-muted-foreground">
            Outcomes, segments and token usage
          </span>
        </summary>
        <div className="mt-6 space-y-6">
          <div className="grid gap-4 lg:grid-cols-3">
            <Counts title="Answer outcomes" counts={outcomes} />
            <Counts title="Error codes" counts={data.error_codes} />
            <Counts title="Negative feedback reasons" counts={reasons} />
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            <Tile label="Input tokens" value={count(data.input_tokens)} hint="Total input usage" />
            <Tile
              label="Output tokens"
              value={count(data.output_tokens)}
              hint="Total generated output"
            />
            <Tile
              label="Auxiliary tokens"
              value={count(data.aux_tokens)}
              hint="Supporting pipeline operations"
            />
          </div>
          <div className="flex flex-wrap gap-2" role="group" aria-label="Group metrics by">
            {(
              [
                ["space", "By space"],
                ["model", "By model"],
                ["config", "By configuration"],
              ] as const
            ).map(([value, label]) => (
              <Button
                key={value}
                size="sm"
                variant={group === value ? "default" : "outline"}
                aria-pressed={group === value}
                onClick={() => setGroup(value)}
              >
                {label}
              </Button>
            ))}
          </div>
          <GroupTable
            title={group === "space" ? "Spaces" : group === "model" ? "Models" : "Configurations"}
            label={(row) =>
              group === "space"
                ? (spaces?.find((space) => space.id === Number(row.key))?.name ??
                  `Space #${row.key ?? "unknown"}`)
                : String(
                    row.key ?? (group === "model" ? "Unknown model" : "No saved configuration"),
                  )
            }
            onSelect={group === "config" ? setSelectedConfig : undefined}
            rows={
              group === "space"
                ? data.by_space
                : group === "model"
                  ? data.by_model
                  : data.by_config_hash
            }
          />
        </div>
      </details>
      <ConfigurationDetails hash={selectedConfig} onClose={() => setSelectedConfig(null)} />
    </section>
  );
}
