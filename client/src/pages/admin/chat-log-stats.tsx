"use client";
import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { type ChatLogsParams, useChatLogStats } from "@/shared/api/hooks/use-chat-logs";
import type { ChatLogGroupStats } from "@/shared/api/types";
import { FEEDBACK_REASON_LABELS } from "@/widgets/chat/message-feedback";

const share = (part: number, total: number) =>
  total ? `${Math.round((part / total) * 100)}%` : "—";
const ms = (value: number | null) => (value == null ? "—" : `${Math.round(value)} мс`);

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded border p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

function Counts({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts);
  return (
    <div className="space-y-1">
      <p className="text-sm font-medium">{title}</p>
      {entries.length === 0 && <p className="text-xs text-muted-foreground">Нет данных</p>}
      {entries.map(([key, value]) => (
        <div key={key} className="flex justify-between gap-4 text-sm">
          <span>{key}</span>
          <span className="tabular-nums">{value}</span>
        </div>
      ))}
    </div>
  );
}

function GroupTable({ title, rows }: { title: string; rows: ChatLogGroupStats[] }) {
  return (
    <div className="space-y-1 overflow-x-auto">
      <p className="text-sm font-medium">{title}</p>
      <table className="w-full text-sm">
        <thead className="text-xs text-muted-foreground">
          <tr>
            <th className="text-left font-normal">Значение</th>
            <th className="text-right font-normal">Всего</th>
            <th className="text-right font-normal">👎</th>
            <th className="text-right font-normal">not_found</th>
            <th className="text-right font-normal">error</th>
            <th className="text-right font-normal">p50</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.key)}>
              <td className="max-w-40 truncate" title={String(row.key ?? "—")}>
                {row.key ?? "—"}
              </td>
              <td className="text-right tabular-nums">{row.total}</td>
              <td className="text-right tabular-nums">{row.down}</td>
              <td className="text-right tabular-nums">{row.not_found}</td>
              <td className="text-right tabular-nums">{row.errors}</td>
              <td className="text-right tabular-nums">{ms(row.latency_p50)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Aggregates over the same filters and curator scope as the journal below. */
export function ChatLogStatsPanel({ params }: { params: ChatLogsParams }) {
  const { limit: _limit, offset: _offset, ...filters } = params;
  const { data, isLoading, isError } = useChatLogStats(filters);
  if (isLoading) return <p>Загрузка сводки…</p>;
  if (isError || !data) return <p role="alert">Не удалось загрузить сводку.</p>;
  const reasons = Object.fromEntries(
    Object.entries(data.reasons).map(([key, value]) => [
      FEEDBACK_REASON_LABELS[key as keyof typeof FEEDBACK_REASON_LABELS] ?? key,
      value,
    ]),
  );
  return (
    <details className="rounded border p-4" open>
      <summary className="cursor-pointer font-medium">Сводка по фильтру</summary>
      <div className="mt-4 space-y-4">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
          <Tile label="Попыток" value={String(data.total)} />
          <Tile
            label="👎 / 👍"
            value={`${data.down} / ${data.up}`}
            hint={`👎 ${share(data.down, data.total)} от всех`}
          />
          <Tile label="Cache hit" value={share(data.cache_hits, data.total)} />
          <Tile
            label="Латентность p50 / p95"
            value={ms(data.latency_p50)}
            hint={ms(data.latency_p95)}
          />
          <Tile label="TTFT p50 / p95" value={ms(data.ttft_p50)} hint={ms(data.ttft_p95)} />
          <Tile
            label="Токены вход / выход"
            value={`${data.input_tokens} / ${data.output_tokens}`}
            hint={`вспомогательные: ${data.aux_tokens}`}
          />
        </div>
        {data.by_day.length > 0 && (
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.by_day} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <XAxis dataKey="key" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={32} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="total" name="Всего" fill="#94a3b8" />
                <Bar dataKey="down" name="👎" fill="#ef4444" />
                <Bar dataKey="not_found" name="not_found" fill="#f59e0b" />
                <Bar dataKey="errors" name="error" fill="#7c3aed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <div className="grid gap-4 md:grid-cols-3">
          <Counts title="Исходы" counts={data.outcomes} />
          <Counts title="Коды ошибок" counts={data.error_codes} />
          <Counts title="Причины 👎" counts={reasons} />
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          <GroupTable title="По пространству" rows={data.by_space} />
          <GroupTable title="По модели" rows={data.by_model} />
          <GroupTable title="По хешу настроек" rows={data.by_config_hash} />
        </div>
      </div>
    </details>
  );
}
