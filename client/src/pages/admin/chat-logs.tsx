"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { Clock, MessageSquare, RefreshCw } from "lucide-react";
import { useState } from "react";
import { useChatLogs } from "@/shared/api/hooks";
import type { ChatLogEntry } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";

export function AdminChatLogsPage() {
  const [search, setSearch] = useState("");
  const [domain, setDomain] = useState<string>("all");
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const { data, refetch, isFetching } = useChatLogs({
    search: search || undefined,
    domain: domain === "all" ? undefined : domain,
    limit: pageSize,
    offset: page * pageSize,
  });

  const columns: ColumnDef<ChatLogEntry>[] = [
    {
      accessorKey: "creation_date",
      header: "Time",
      cell: ({ row }) => {
        const d = new Date(row.original.creation_date);
        return (
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {d.toLocaleString()}
          </span>
        );
      },
    },
    {
      accessorKey: "user_id",
      header: "User",
      cell: ({ row }) => (
        <span className="text-muted-foreground">#{row.original.user_id ?? "—"}</span>
      ),
    },
    {
      accessorKey: "question",
      header: "Question",
      cell: ({ row }) => (
        <span className="text-sm line-clamp-2 max-w-md">{row.original.question}</span>
      ),
    },
    {
      accessorKey: "answer",
      header: "Answer",
      cell: ({ row }) => (
        <span className="text-sm text-muted-foreground line-clamp-2 max-w-lg">
          {row.original.answer}
        </span>
      ),
    },
    {
      accessorKey: "domain",
      header: "Domain",
      cell: ({ row }) => (
        <Badge variant={row.original.domain === "legal" ? "default" : "secondary"}>
          {row.original.domain ?? "—"}
        </Badge>
      ),
    },
    {
      accessorKey: "breadth",
      header: "Breadth",
      cell: ({ row }) => <Badge variant="outline">{row.original.breadth ?? "—"}</Badge>,
    },
    {
      accessorKey: "latency_ms",
      header: "Latency",
      cell: ({ row }) => {
        const ms = row.original.latency_ms;
        if (!ms) return "—";
        return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
      },
    },
    {
      accessorKey: "retrieval_count",
      header: "Chunks",
      cell: ({ row }) => row.original.retrieval_count ?? "—",
    },
    {
      accessorKey: "reranker_score",
      header: "Score",
      cell: ({ row }) => {
        const s = row.original.reranker_score;
        return s != null ? s.toFixed(3) : "—";
      },
    },
    {
      accessorKey: "input_tokens",
      header: "Tokens",
      cell: ({ row }) => {
        const inp = row.original.input_tokens;
        const out = row.original.output_tokens;
        if (inp == null && out == null) return "—";
        const format = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
        return (
          <span className="text-xs whitespace-nowrap" title={`Input: ${inp ?? 0} / Output: ${out ?? 0}`}>
            {inp != null ? format(inp) : "0"} / {out != null ? format(out) : "0"}
          </span>
        );
      },
    },
  ];

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <MessageSquare className="h-6 w-6" /> Chat Logs
          </h1>
          <p className="text-muted-foreground">Q&A history for quality tracking</p>
        </div>
        <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <Label>Search</Label>
              <Input
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setPage(0);
                }}
                placeholder="Search questions or answers..."
              />
            </div>
            <div className="w-40">
              <Label>Domain</Label>
              <Select
                value={domain}
                onValueChange={(v) => {
                  setDomain(v);
                  setPage(0);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="general">General</SelectItem>
                  <SelectItem value="legal">Legal</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-4 w-4" /> Q&A History
          </CardTitle>
          <CardDescription>{data?.total ?? 0} total entries</CardDescription>
        </CardHeader>
        <CardContent>
          <DataTable
            columns={columns}
            data={data?.logs || []}
            searchKey="question"
            searchPlaceholder="Filter..."
          />
          {totalPages > 1 && (
            <div className="flex justify-between items-center mt-4">
              <span className="text-sm text-muted-foreground">
                Page {page + 1} of {totalPages}
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 0}
                  onClick={() => setPage(page - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
