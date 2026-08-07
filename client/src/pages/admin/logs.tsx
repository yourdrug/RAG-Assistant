"use client";
import { useState } from "react";
import { useLogs } from "@/shared/api/hooks";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Skeleton } from "@/shared/ui/skeleton";
import { ScrollText, RefreshCw, Search } from "lucide-react";

export function LogsPage() {
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);

  const { data: logsData, isLoading, refetch } = useLogs({
    limit: 200,
    search: search || undefined,
  });

  const handleSearch = () => {
    setSearch(searchInput);
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">User Actions</h1>
          <p className="text-muted-foreground">Audit trail of user operations (auto-refresh 5s)</p>
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
            <RefreshCw className="h-4 w-4 mr-2" /> Refresh
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-3 items-center">
        <div className="flex-1 flex gap-2">
          <Input
            placeholder="Search actions..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="max-w-sm"
          />
          <Button variant="outline" size="sm" onClick={handleSearch}>
            <Search className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Log entries */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ScrollText className="h-5 w-5" />
            Actions
            <Badge variant="secondary" className="ml-auto">
              {logsData?.total ?? 0} entries
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} className="h-6 w-full" />
              ))}
            </div>
          ) : (
            <ScrollArea className="h-[600px]">
              <div className="font-mono text-xs space-y-0.5">
                {logsData?.logs?.length === 0 && (
                  <p className="text-center text-muted-foreground py-8">No actions recorded yet</p>
                )}
                {logsData?.logs?.map((log, i) => {
                  const parts = log.message.split(" | ");
                  const action = parts[0] || log.message;
                  const details = parts.slice(1).join(" | ");
                  return (
                    <div key={i} className="flex gap-2 py-1 hover:bg-muted/50 px-2 rounded border-b border-border/50">
                      <span className="text-muted-foreground shrink-0 w-[160px]">{log.timestamp}</span>
                      <Badge variant="secondary" className="shrink-0 text-[10px] px-1.5 py-0">{action}</Badge>
                      {details && (
                        <span className="flex-1 text-muted-foreground truncate">{details}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
