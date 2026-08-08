"use client";
import { useState, useEffect, useRef } from "react";
import { useLogs } from "@/shared/api/hooks";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Skeleton } from "@/shared/ui/skeleton";
import { ScrollText, RefreshCw, Search, ChevronDown, ChevronRight, Clock, User, FileText, AlertCircle, Info, AlertTriangle, XCircle } from "lucide-react";

const levelConfig: Record<string, { color: string; icon: React.ReactNode; bg: string }> = {
  INFO: { color: "text-blue-600 dark:text-blue-400", icon: <Info className="h-3.5 w-3.5" />, bg: "bg-blue-500/10" },
  WARNING: { color: "text-amber-600 dark:text-amber-400", icon: <AlertTriangle className="h-3.5 w-3.5" />, bg: "bg-amber-500/10" },
  ERROR: { color: "text-red-600 dark:text-red-400", icon: <XCircle className="h-3.5 w-3.5" />, bg: "bg-red-500/10" },
  CRITICAL: { color: "text-red-700 dark:text-red-300", icon: <AlertCircle className="h-3.5 w-3.5" />, bg: "bg-red-600/10" },
  DEBUG: { color: "text-gray-500 dark:text-gray-400", icon: <FileText className="h-3.5 w-3.5" />, bg: "bg-gray-500/10" },
};

function formatTimestamp(ts: string) {
  try {
    const d = new Date(ts);
    const time = d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const date = d.toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
    return { time, date };
  } catch {
    return { time: ts, date: "" };
  }
}

function LogEntryRow({ log }: { log: any }) {
  const [expanded, setExpanded] = useState(false);
  const level = log.level?.toUpperCase() || "INFO";
  const config = levelConfig[level] || levelConfig.INFO;
  const { time } = formatTimestamp(log.timestamp);

  return (
    <div
      className={`border-b border-border/30 last:border-0 transition-colors hover:bg-muted/30 ${expanded ? "bg-muted/20" : ""}`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-2 px-3 py-2 text-left"
      >
        {/* Expand icon */}
        <span className="mt-0.5 shrink-0 text-muted-foreground">
          {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        </span>

        {/* Level badge */}
        <span className={`shrink-0 mt-0.5 ${config.color}`}>
          {config.icon}
        </span>

        {/* Timestamp */}
        <span className="shrink-0 text-xs text-muted-foreground font-mono w-[70px]">{time}</span>

        {/* Message */}
        <span className="flex-1 text-sm font-mono leading-relaxed break-all line-clamp-2">
          {log.message}
        </span>

        {/* Request ID */}
        {log.request_id && log.request_id !== "-" && (
          <span className="shrink-0 text-[10px] text-muted-foreground/60 font-mono max-w-[80px] truncate" title={log.request_id}>
            {log.request_id.slice(0, 8)}
          </span>
        )}
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="px-3 pb-3 pt-0 ml-7 space-y-2">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <Clock className="h-3 w-3" />
              <span>{log.timestamp}</span>
            </div>
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <AlertCircle className="h-3 w-3" />
              <span>Level: <span className={config.color}>{level}</span></span>
            </div>
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <User className="h-3 w-3" />
              <span>Logger: <span className="text-foreground">{log.logger}</span></span>
            </div>
            {log.request_id && log.request_id !== "-" && (
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <FileText className="h-3 w-3" />
                <span>Request: <span className="text-foreground font-mono text-[10px]">{log.request_id}</span></span>
              </div>
            )}
          </div>
          <div className="bg-muted/50 rounded-md p-2 font-mono text-xs text-foreground break-all">
            {log.message}
          </div>
          {log.filename && (
            <div className="text-xs text-muted-foreground">
              <FileText className="h-3 w-3 inline mr-1" />
              {log.filename}{log.lineno ? `:${log.lineno}` : ""}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function LogsPage() {
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [levelFilter, setLevelFilter] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { data: logsData, isLoading, refetch } = useLogs({
    limit: 200,
    search: search || undefined,
    level: levelFilter || undefined,
  });

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => refetch(), 5000);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, refetch]);

  const handleSearch = () => {
    setSearch(searchInput);
  };

  const levels = ["INFO", "WARNING", "ERROR", "DEBUG"];

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <ScrollText className="h-6 w-6 text-muted-foreground" />
            Logs
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {logsData?.total ?? 0} entries
            {autoRefresh && <span className="ml-2 text-green-600 dark:text-green-400">• auto-refreshing</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? "Auto: ON" : "Auto: OFF"}
          </Button>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex-1 flex gap-2 min-w-[200px]">
          <Input
            placeholder="Search logs..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="max-w-sm"
          />
          <Button variant="outline" size="sm" onClick={handleSearch}>
            <Search className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex gap-1">
          <Button
            variant={levelFilter === null ? "default" : "outline"}
            size="sm"
            onClick={() => setLevelFilter(null)}
          >
            All
          </Button>
          {levels.map((lvl) => {
            const config = levelConfig[lvl];
            return (
              <Button
                key={lvl}
                variant={levelFilter === lvl ? "default" : "outline"}
                size="sm"
                onClick={() => setLevelFilter(levelFilter === lvl ? null : lvl)}
                className={levelFilter === lvl ? "" : config.color}
              >
                {config.icon}
                <span className="ml-1 hidden sm:inline">{lvl}</span>
              </Button>
            );
          })}
        </div>
      </div>

      {/* Log entries */}
      <div className="border rounded-lg bg-card">
        {isLoading && !logsData ? (
          <div className="p-4 space-y-2">
            {Array.from({ length: 15 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        ) : (
          <ScrollArea className="h-[calc(100vh-280px)]">
            <div className="divide-y divide-border/30">
              {logsData?.logs?.length === 0 && (
                <p className="text-center text-muted-foreground py-12">No logs found</p>
              )}
              {logsData?.logs?.map((log, i) => (
                <LogEntryRow key={i} log={log} />
              ))}
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  );
}
