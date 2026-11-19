import { useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useSpaces } from "@/shared/api/hooks/use-spaces";
import type { ChatLogStats, FeedbackReason } from "@/shared/api/types";
import { formatDate } from "@/shared/lib/date-time";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { FEEDBACK_REASON_LABELS } from "./chat-log-labels";

type DateRange = { date_from: string; date_to: string };

const dailySeries = [
  { key: "total", label: "Requests", color: "var(--color-primary)" },
  { key: "not_found", label: "Not found", color: "#d97706" },
  { key: "errors", label: "Errors", color: "#e11d48" },
];

export function chatLogsUrl(dates: DateRange, filters: Record<string, string> = {}) {
  return `/admin/chat-logs?${new URLSearchParams({ ...dates, ...filters })}`;
}

function rate(part: number, total: number) {
  return total
    ? (part / total).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 })
    : "—";
}

function milliseconds(value: number | null) {
  return value == null ? "—" : `${Math.round(value).toLocaleString()} ms`;
}

function dailyActivity(stats: ChatLogStats, dates: DateRange) {
  const byDay = new Map(stats.by_day.map((row) => [row.key, row]));
  const day = new Date(`${dates.date_from.slice(0, 10)}T00:00:00Z`);
  const end = dates.date_to.slice(0, 10);
  const rows = [];

  for (let key = day.toISOString().slice(0, 10); key <= end; key = day.toISOString().slice(0, 10)) {
    const row = byDay.get(key);
    rows.push({
      key,
      total: row?.total ?? 0,
      not_found: row?.not_found ?? 0,
      errors: row?.errors ?? 0,
    });
    day.setUTCDate(day.getUTCDate() + 1);
  }
  return rows;
}

function CountLinks({
  title,
  counts,
  href,
  label,
}: {
  title: string;
  counts: Record<string, number>;
  href: (key: string) => string;
  label?: (key: string) => string;
}) {
  const top = Object.entries(counts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);
  return (
    <Card className="min-w-0">
      <CardHeader className="p-4">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {top.length ? (
          <ul className="divide-y">
            {top.map(([key, count]) => (
              <li key={key}>
                <Link
                  to={href(key)}
                  className="flex justify-between gap-4 py-2 text-sm text-primary hover:underline"
                >
                  <span className="wrap-anywhere">{label?.(key) ?? key}</span>
                  <span className="tabular-nums">{count.toLocaleString()}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">No entries in this period</p>
        )}
      </CardContent>
    </Card>
  );
}

export function DashboardRag({ stats, dates }: { stats: ChatLogStats; dates: DateRange }) {
  const notFound = stats.outcomes.not_found ?? 0;
  const errors = stats.outcomes.error ?? 0;
  const ratings = stats.up + stats.down;
  const metrics = [
    {
      label: "Not found",
      value: rate(notFound, stats.total),
      detail: `${notFound.toLocaleString()} requests`,
      href: chatLogsUrl(dates, { outcome: "not_found" }),
    },
    {
      label: "Errors",
      value: rate(errors, stats.total),
      detail: `${errors.toLocaleString()} requests`,
      href: chatLogsUrl(dates, { outcome: "error" }),
    },
    {
      label: "Cache hit rate",
      value: rate(stats.cache_hits, stats.total),
      detail: `${stats.cache_hits.toLocaleString()} hits`,
    },
    {
      label: "Positive / negative feedback",
      value: `${rate(stats.up, ratings)} / ${rate(stats.down, ratings)}`,
      detail: `${ratings.toLocaleString()} ratings`,
    },
  ];
  const operationalMetrics = [
    {
      label: "Latency · p50 / p95",
      value: `${milliseconds(stats.latency_p50)} / ${milliseconds(stats.latency_p95)}`,
    },
    {
      label: "TTFT · p50 / p95",
      value: `${milliseconds(stats.ttft_p50)} / ${milliseconds(stats.ttft_p95)}`,
    },
    { label: "Input tokens", value: stats.input_tokens.toLocaleString() },
    { label: "Output tokens", value: stats.output_tokens.toLocaleString() },
    { label: "Auxiliary tokens", value: stats.aux_tokens.toLocaleString() },
  ];
  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map(({ label, value, detail, href }) => {
          const content = (
            <>
              <p className="text-xs font-medium text-muted-foreground">{label}</p>
              <p className="text-xl font-semibold tabular-nums">{value}</p>
              <p className="text-xs text-muted-foreground">{detail}</p>
            </>
          );
          return (
            <Card key={label}>
              {href ? (
                <Link
                  to={href}
                  className="block rounded-xl p-4 hover:bg-muted/50 focus-visible:outline-ring"
                >
                  {content}
                </Link>
              ) : (
                <div className="p-4">{content}</div>
              )}
            </Card>
          );
        })}
      </div>
      <Card className="min-w-0">
        <CardHeader className="p-4">
          <CardTitle className="text-base">Daily requests (UTC)</CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4">
          <div className="h-64 w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={dailyActivity(stats, dates)}
                accessibilityLayer
                margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                <XAxis
                  dataKey="key"
                  minTickGap={24}
                  tickFormatter={(date: string) => date.slice(5)}
                  tick={{ fontSize: 12 }}
                />
                <YAxis allowDecimals={false} tickCount={5} tick={{ fontSize: 12 }} />
                <Tooltip
                  content={({ active, label, payload }) =>
                    active && payload?.length ? (
                      <div className="rounded-md border bg-popover p-3 text-xs text-popover-foreground shadow-md">
                        <p>{formatDate(String(label))}</p>
                        {payload.map(({ dataKey, name, value }) => (
                          <p key={String(dataKey)}>
                            {name}: {value?.toLocaleString()}
                          </p>
                        ))}
                      </div>
                    ) : null
                  }
                />
                <Legend iconType="plainline" />
                {dailySeries.map(({ key, label, color }) => (
                  <Line
                    key={key}
                    type="linear"
                    dataKey={key}
                    name={label}
                    stroke={color}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <dl className="mt-4 grid gap-3 border-t pt-4 sm:grid-cols-2 xl:grid-cols-5">
            {operationalMetrics.map(({ label, value }) => (
              <div key={label}>
                <dt className="text-xs text-muted-foreground">{label}</dt>
                <dd className="mt-1 text-sm font-semibold tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        </CardContent>
      </Card>
      <div className="grid gap-3 md:grid-cols-2">
        <CountLinks
          title="Top error codes"
          counts={stats.error_codes}
          href={(error_code) => chatLogsUrl(dates, { error_code })}
        />
        <CountLinks
          title="Top feedback reasons"
          counts={stats.reasons}
          href={(reason) => chatLogsUrl(dates, { reason })}
          label={(reason) => FEEDBACK_REASON_LABELS[reason as FeedbackReason] ?? reason}
        />
      </div>
    </>
  );
}

export function DashboardRagBreakdown({ stats, dates }: { stats: ChatLogStats; dates: DateRange }) {
  const [group, setGroup] = useState<"space" | "model">("space");
  const { data: spaces } = useSpaces();
  const rows = group === "space" ? stats.by_space : stats.by_model;
  return (
    <Card className="min-w-0">
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 space-y-0 p-4">
        <CardTitle>RAG breakdown</CardTitle>
        <div className="flex gap-1" role="group" aria-label="RAG breakdown by">
          {(["space", "model"] as const).map((value) => (
            <Button
              key={value}
              size="sm"
              variant={group === value ? "secondary" : "ghost"}
              aria-pressed={group === value}
              onClick={() => setGroup(value)}
            >
              By {value}
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {rows.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr>
                  <th scope="col" className="py-2 capitalize">
                    {group}
                  </th>
                  <th scope="col" className="px-2 py-2 text-right">
                    Requests
                  </th>
                  <th scope="col" className="px-2 py-2 text-right">
                    Not found
                  </th>
                  <th scope="col" className="py-2 text-right">
                    Negative
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ key, total, not_found, down }) => {
                  const label =
                    key == null
                      ? "Unknown"
                      : group === "space"
                        ? (spaces?.find((space) => space.id === Number(key))?.name ??
                          `Space #${key}`)
                        : String(key);
                  return (
                    <tr key={String(key)} className="border-t">
                      <td className="max-w-48 wrap-anywhere py-2">
                        {key == null ? (
                          label
                        ) : (
                          <Link
                            className="text-primary hover:underline"
                            to={chatLogsUrl(dates, {
                              [group === "space" ? "space_id" : "model"]: String(key),
                            })}
                          >
                            {label}
                          </Link>
                        )}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {total.toLocaleString()}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {not_found.toLocaleString()}
                      </td>
                      <td className="py-2 text-right tabular-nums">{down.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No requests in this period</p>
        )}
      </CardContent>
    </Card>
  );
}
