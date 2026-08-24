import { Badge } from "@/shared/ui/badge";
import type { DryRunResponse } from "@/shared/api/types";

const TYPE_COLORS: Record<string, string> = {
  text: "bg-green-500",
  table: "bg-blue-500",
  scan: "bg-yellow-500",
  garbled: "bg-red-500",
  empty: "bg-gray-300 dark:bg-gray-600",
};

const TYPE_LABELS: Record<string, string> = {
  text: "Text",
  scan: "Scan (no OCR)",
  garbled: "Garbled",
  empty: "Empty",
  table: "Table",
};

export { TYPE_COLORS, TYPE_LABELS };

export function DryRunSummaryBar({ result }: { result: DryRunResponse }) {
  const { summary, total_pages, total_chars, quality_score } = result;
  const okCount = summary.text + summary.table;
  const okPct = total_pages > 0 ? Math.round((okCount / total_pages) * 100) : 0;
  const avgChars = total_pages > 0 ? Math.round(total_chars / total_pages) : 0;
  const estimatedChunks = Math.ceil(total_chars / 1500);
  const badPct = Math.round(quality_score * 100);
  const problemCount = total_pages - okCount;

  return (
    <div className="p-3 rounded-lg border space-y-2">
      <div className="flex items-center gap-3 flex-wrap">
        <span
          className={`text-2xl font-bold ${
            badPct <= 10
              ? "text-green-600 dark:text-green-400"
              : badPct <= 30
                ? "text-yellow-600 dark:text-yellow-400"
                : "text-red-600 dark:text-red-400"
          }`}
        >
          {okPct}%
        </span>
        <span className="text-sm text-muted-foreground">OK</span>
        <span className="text-muted-foreground">&middot;</span>
        <span className="text-sm">
          {total_pages} pages &middot; {total_chars.toLocaleString()} chars
        </span>
        <span className="text-muted-foreground">&middot;</span>
        <span className="text-sm text-muted-foreground">
          avg {avgChars.toLocaleString()} chars/page
        </span>
        <span className="text-muted-foreground">&middot;</span>
        <span className="text-sm text-muted-foreground">
          ~{estimatedChunks.toLocaleString()} chunks est.
        </span>
        {badPct > 0 && (
          <>
            <span className="text-muted-foreground">&middot;</span>
            <Badge variant={badPct <= 10 ? "warning" : "destructive"}>
              bad: {(quality_score * 100).toFixed(1)}% &middot; {problemCount} pages
            </Badge>
          </>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5 text-xs">
        {summary.text > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            {summary.text} text
          </span>
        )}
        {summary.table > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            {summary.table} table{summary.table > 1 ? "s" : ""}
          </span>
        )}
        {summary.scan > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-yellow-50 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300">
            <span className="w-1.5 h-1.5 rounded-full bg-yellow-500" />
            {summary.scan} scan
          </span>
        )}
        {summary.garbled > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
            {summary.garbled} garbled
          </span>
        )}
        {summary.empty > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
            <span className="w-1.5 h-1.5 rounded-full bg-gray-400" />
            {summary.empty} empty
          </span>
        )}
      </div>
    </div>
  );
}
