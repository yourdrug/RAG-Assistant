import { TYPE_COLORS } from "./DryRunSummaryBar";
import type { DryRunPageResult } from "@/shared/api/types";
import { RefreshCw } from "lucide-react";

interface DryRunPageListProps {
  pages: DryRunPageResult[];
  selectedPage: number | null;
  onSelectPage: (page: number) => void;
}

export function DryRunPageList({ pages, selectedPage, onSelectPage }: DryRunPageListProps) {
  return (
    <div className="p-1 space-y-0.5">
      {pages.map((p) => {
        const color = TYPE_COLORS[p.type] || "bg-gray-300";
        const isActive = selectedPage === p.page;
        return (
          <button
            key={p.page}
            onClick={() => onSelectPage(p.page)}
            className={`w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded transition-colors text-left ${
              isActive ? "bg-accent text-accent-foreground" : "hover:bg-muted/50"
            }`}
          >
            <span className={`w-2 h-2 rounded-full shrink-0 ${color}`} />
            <span className="font-mono text-muted-foreground w-8 shrink-0">{p.page}</span>
            <span className="truncate flex-1">
              {p.type}
              {p.previous_type && <RefreshCw className="inline h-3 w-3 ml-1 text-blue-500" />}
            </span>
            <span className="text-muted-foreground shrink-0">{p.chars.toLocaleString()}</span>
          </button>
        );
      })}
    </div>
  );
}
