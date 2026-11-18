"use client";
import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowUpRight,
  ChevronDown,
  Database,
  Layers,
  RefreshCw,
  Server,
} from "lucide-react";
import { Fragment, useState } from "react";
import { apiClient } from "@/shared/api/client";
import { formatTime } from "@/shared/lib/date-time";
import { cn } from "@/shared/lib/utils";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

interface VectorDBCollection {
  name: string;
  points_count: number;
  vectors_count: number;
  indexed_vectors_count: number;
  segments_count: number;
  status: string;
  optimizer_status: string;
  hnsw_m: number | null;
  hnsw_ef_construct: number | null;
  on_disk_payload: boolean | null;
  vector_size: number | null;
  distance: string | null;
}

interface VectorDBInfo {
  collections: VectorDBCollection[];
  active_collection: string;
  qdrant_status: string;
}

function CollectionStatus({ status }: { status: string }) {
  const variant =
    status === "green"
      ? "success"
      : status === "red"
        ? "destructive"
        : status === "yellow"
          ? "warning"
          : "outline";
  const label =
    status === "green"
      ? "Healthy"
      : status === "yellow"
        ? "Optimizing"
        : status === "red"
          ? "Error"
          : status || "Unknown";
  return <Badge variant={variant}>{label}</Badge>;
}

export function AdminVectorDBPage() {
  const [expanded, setExpanded] = useState<string[]>([]);
  const { data, isLoading, isError, isFetching, dataUpdatedAt, refetch } = useQuery({
    queryKey: ["admin", "vectordb"],
    queryFn: async () => (await apiClient.get<VectorDBInfo>("/admin/vectordb/info")).data,
    refetchInterval: 15000,
  });
  const collections = [...(data?.collections ?? [])].sort(
    (a, b) =>
      Number(b.name === data?.active_collection) - Number(a.name === data?.active_collection) ||
      a.name.localeCompare(b.name),
  );
  const connected = data?.qdrant_status === "ok";
  const metrics = [
    {
      label: "Points / chunks",
      value: collections.reduce((sum, col) => sum + col.points_count, 0),
      icon: Database,
      description: "Stored across all collections",
    },
    {
      label: "Vectors",
      value: collections.reduce((sum, col) => sum + col.vectors_count, 0),
      icon: Layers,
      description: "Reported by Qdrant",
    },
    {
      label: "HNSW indexed",
      value: collections.reduce((sum, col) => sum + col.indexed_vectors_count, 0),
      icon: ArrowUpRight,
      description: "Vectors in the approximate search index",
    },
  ];

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 p-4 sm:p-6 lg:p-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Vector DB</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Qdrant health, collections and search indexes.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs text-muted-foreground">
            {dataUpdatedAt
              ? `Updated ${formatTime(dataUpdatedAt)} · every 15s`
              : "Auto-refresh every 15s"}
          </span>
          <Button variant="outline" disabled={isFetching} onClick={() => void refetch()}>
            <RefreshCw className={cn("h-4 w-4", isFetching && "animate-spin")} />
            {isFetching ? "Refreshing…" : "Refresh"}
          </Button>
        </div>
      </div>

      {isError && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4"
        >
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
          <div className="space-y-1 text-sm">
            <p className="font-medium">Could not refresh Vector DB information</p>
            <p className="text-muted-foreground">
              {data
                ? "Showing the last successful snapshot. Connection status may have changed."
                : "Check the API connection and try again."}
            </p>
            <Button
              variant="link"
              className="h-auto p-0"
              disabled={isFetching}
              onClick={() => void refetch()}
            >
              Try again
            </Button>
          </div>
        </div>
      )}

      {isLoading ? (
        <div aria-label="Loading Vector DB" aria-busy="true" className="space-y-6">
          <Skeleton className="h-28 w-full rounded-xl" />
          <div className="grid gap-4 sm:grid-cols-3">
            {[1, 2, 3].map((item) => (
              <Skeleton key={item} className="h-32 rounded-xl" />
            ))}
          </div>
          <Skeleton className="h-64 w-full rounded-xl" />
        </div>
      ) : data ? (
        <>
          <Card>
            <CardContent className="flex flex-wrap items-center justify-between gap-5 p-5 sm:p-6">
              <div className="flex items-center gap-4">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-muted">
                  <Server className="h-5 w-5 text-muted-foreground" />
                </div>
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold">Qdrant</span>
                    <Badge variant={isError ? "warning" : connected ? "success" : "destructive"}>
                      {isError ? "Status unverified" : connected ? "Connected" : "Unavailable"}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">Vector storage for retrieval</p>
                </div>
              </div>
              <div className="min-w-0 space-y-1 sm:text-right">
                <p className="text-xs text-muted-foreground">Active collection</p>
                <p className="break-all font-mono text-sm font-medium">
                  {data.active_collection || "Not configured"}
                </p>
              </div>
            </CardContent>
          </Card>

          {!connected && (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm"
            >
              <AlertCircle className="h-5 w-5 shrink-0 text-destructive" />
              <div>
                <p className="font-medium">Qdrant is unavailable</p>
                <p className="mt-1 text-muted-foreground">
                  Collection information may be incomplete. Check the service and refresh to retry.
                </p>
              </div>
            </div>
          )}

          <div className="grid gap-4 sm:grid-cols-3">
            {metrics.map(({ label, value, icon: Icon, description }) => (
              <Card key={label}>
                <CardContent className="p-5">
                  <div className="flex items-center justify-between gap-2 text-sm text-muted-foreground">
                    <span>{label}</span>
                    <Icon className="h-4 w-4" />
                  </div>
                  <p className="mt-3 text-3xl font-semibold tabular-nums tracking-tight">
                    {connected ? value.toLocaleString() : "—"}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">{description}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <CardTitle className="text-base">Collections</CardTitle>
                <Badge variant="secondary">{collections.length}</Badge>
              </div>
              <CardDescription>
                Active collection first. Expand a row to inspect its configuration.
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              {collections.length ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="pl-6">Collection</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Points</TableHead>
                      <TableHead className="text-right">Vectors</TableHead>
                      <TableHead className="text-right">HNSW indexed</TableHead>
                      <TableHead>Optimizer</TableHead>
                      <TableHead>
                        <span className="sr-only">Details</span>
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {collections.map((col) => {
                      const open = expanded.includes(col.name);
                      const detailId = `collection-${encodeURIComponent(col.name)}`;
                      return (
                        <Fragment key={col.name}>
                          <TableRow
                            className={cn(col.name === data.active_collection && "bg-muted/30")}
                          >
                            <TableCell className="pl-6">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="max-w-64 break-all font-mono text-xs font-medium">
                                  {col.name}
                                </span>
                                {col.name === data.active_collection && (
                                  <Badge variant="outline">Active</Badge>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <CollectionStatus status={col.status} />
                            </TableCell>
                            <TableCell className="text-right tabular-nums">
                              {col.points_count.toLocaleString()}
                            </TableCell>
                            <TableCell className="text-right tabular-nums">
                              {col.vectors_count.toLocaleString()}
                            </TableCell>
                            <TableCell className="text-right tabular-nums">
                              {col.indexed_vectors_count.toLocaleString()}
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant={col.optimizer_status === "ok" ? "success" : "destructive"}
                              >
                                {col.optimizer_status || "Unknown"}
                              </Badge>
                            </TableCell>
                            <TableCell className="pr-4">
                              <Button
                                variant="ghost"
                                size="icon"
                                aria-label={`Details for ${col.name}`}
                                aria-expanded={open}
                                aria-controls={detailId}
                                onClick={() =>
                                  setExpanded((current) =>
                                    open
                                      ? current.filter((name) => name !== col.name)
                                      : [...current, col.name],
                                  )
                                }
                              >
                                <ChevronDown
                                  className={cn(
                                    "h-4 w-4 transition-transform",
                                    open && "rotate-180",
                                  )}
                                />
                              </Button>
                            </TableCell>
                          </TableRow>
                          <TableRow id={detailId} hidden={!open} className="hover:bg-transparent">
                            <TableCell colSpan={7} className="bg-muted/20 px-6 py-5">
                              <dl className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                                {[
                                  ["Vector dimensions", col.vector_size ?? "—"],
                                  ["Distance", col.distance?.replace("Distance.", "") ?? "—"],
                                  [
                                    "HNSW · m / ef_construct",
                                    `${col.hnsw_m ?? "—"} / ${col.hnsw_ef_construct ?? "—"}`,
                                  ],
                                  [
                                    "Payload storage",
                                    col.on_disk_payload === true
                                      ? "On disk"
                                      : col.on_disk_payload === false
                                        ? "In memory"
                                        : "—",
                                  ],
                                  ["Segments", col.segments_count],
                                ].map(([label, value]) => (
                                  <div key={label}>
                                    <dt className="text-xs text-muted-foreground">{label}</dt>
                                    <dd className="mt-1 text-sm font-medium">{value}</dd>
                                  </div>
                                ))}
                              </dl>
                            </TableCell>
                          </TableRow>
                        </Fragment>
                      );
                    })}
                  </TableBody>
                </Table>
              ) : (
                <div className="flex flex-col items-center px-6 py-12 text-center">
                  <Database className="mb-3 h-8 w-8 text-muted-foreground/50" />
                  <p className="text-sm font-medium">
                    {connected ? "No collections yet" : "Collections unavailable"}
                  </p>
                  <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                    {connected
                      ? "Collections will appear here once they are created in Qdrant."
                      : "Restore the Qdrant connection to view collections."}
                  </p>
                </div>
              )}
              <p className="border-t px-6 py-3 text-xs leading-relaxed text-muted-foreground">
                HNSW indexed is not a search-readiness indicator. Qdrant can search unindexed
                vectors directly.
              </p>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}
