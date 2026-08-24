import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DryRunPageResult } from "@/shared/api/types";

const CHART_COLORS: Record<string, string> = {
  text: "#22c55e",
  table: "#3b82f6",
  scan: "#eab308",
  garbled: "#ef4444",
  empty: "#9ca3af",
};

interface DryRunCharsChartProps {
  pages: DryRunPageResult[];
  selectedPage: number | null;
  onSelectPage: (page: number) => void;
}

export function DryRunCharsChart({ pages, selectedPage, onSelectPage }: DryRunCharsChartProps) {
  const data = pages.map((p) => ({
    page: p.page,
    chars: p.chars,
    type: p.type,
    label: p.type,
  }));

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-muted-foreground">Characters per page</span>
        <span className="text-xs text-muted-foreground">{pages.length} pages</span>
      </div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 2, right: 2, bottom: 0, left: 0 }}
            onClick={(data) => {
              if (data?.activePayload?.[0]) {
                onSelectPage(data.activePayload[0].payload.page);
              }
            }}
          >
            <XAxis
              dataKey="page"
              tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
              interval={Math.max(0, Math.floor(pages.length / 20))}
            />
            <YAxis
              tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
              width={40}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-popover text-popover-foreground px-2 py-1 rounded shadow-md text-xs">
                    p.{d.page} &middot; {d.chars.toLocaleString()} chars &middot; {d.label}
                  </div>
                );
              }}
            />
            <Bar dataKey="chars" radius={[2, 2, 0, 0]} maxBarSize={20}>
              {data.map((entry) => (
                <Cell
                  key={entry.page}
                  fill={CHART_COLORS[entry.type] || "#9ca3af"}
                  opacity={selectedPage === entry.page ? 1 : 0.75}
                  stroke={selectedPage === entry.page ? "hsl(var(--foreground))" : undefined}
                  strokeWidth={selectedPage === entry.page ? 2 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
