"use client";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/shared/api/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/shared/ui/tooltip";
import { Database, Server, Info } from "lucide-react";

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

function TooltipLabel({ label, tooltip }: { label: string; tooltip: string }) {
  return (
    <div className="flex items-center gap-1.5">
      {label}
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-64 text-xs leading-relaxed">
          {tooltip}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

const METRIC_INFO: Record<string, string> = {
  status: "Overall health of the collection (green = healthy, yellow = degraded, red = error).",
  optimizer: "Whether the background optimizer is running normally. If 'error' — manual action may be needed.",
  points: "Total number of documents (chunks) stored. Each chunk = one point with a vector and payload.",
  vectors: "Total vectors available for search. Equals points_count when no deletions occurred.",
  indexed: "Vectors that have been indexed by HNSW and are ready for fast approximate search. Should equal vectors_count when indexing is complete.",
  segments: "Number of segments the collection is split into. More segments = more parallel search. Optimal range: 1–4 for most use cases.",
  "vector size": "Dimensionality of the embedding vector (e.g. 1024 for bge-m3). All vectors in a collection must share the same size.",
  distance: "Similarity metric used for vector comparison. 'cosine' for normalized embeddings, 'dot' for raw scores, 'euclid' for L2 distance.",
  "hnsw (m / ef_construct)": "HNSW graph parameters: m = max connections per node (higher = more recall, more RAM), ef_construct = build-time beam width (higher = better graph quality, slower indexing).",
  "on disk payload": "When 'Yes', payloads are stored on disk instead of RAM. Saves memory at the cost of slightly slower query latency.",
};

export function AdminVectorDBPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["admin", "vectordb"],
    queryFn: async () => (await apiClient.get<VectorDBInfo>("/admin/vectordb/info")).data,
    refetchInterval: 15000,
  });

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div>
          <Skeleton className="h-8 w-48 mb-2" />
          <Skeleton className="h-4 w-72" />
        </div>
        <Card>
          <CardContent className="pt-6">
            <Skeleton className="h-48 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Vector DB</h1>
          <p className="text-muted-foreground">Qdrant vector database information</p>
        </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            Connection Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">Qdrant:</span>
            <Badge variant={data?.qdrant_status === "ok" ? "success" : "destructive"}>
              {data?.qdrant_status || "unknown"}
            </Badge>
            <span className="text-sm text-muted-foreground ml-4">Active collection:</span>
            <Badge variant="outline">{data?.active_collection}</Badge>
          </div>
        </CardContent>
      </Card>

      {data?.collections && data.collections.length > 0 ? (
        data.collections.map((col) => {
          const isIndexed = col.indexed_vectors_count === col.vectors_count;
          const indexPercent = col.vectors_count > 0
            ? Math.round((col.indexed_vectors_count / col.vectors_count) * 100)
            : 0;

          return (
            <Card key={col.name}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  {col.name}
                  {col.name === data.active_collection && (
                    <Badge variant="default" className="text-xs">active</Badge>
                  )}
                </CardTitle>
                <CardDescription>Collection details</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Metric</TableHead>
                      <TableHead>Value</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Status" tooltip={METRIC_INFO.status} /></TableCell>
                      <TableCell>
                        <Badge variant={col.status === "green" ? "success" : "secondary"}>
                          {col.status}
                        </Badge>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Optimizer" tooltip={METRIC_INFO.optimizer} /></TableCell>
                      <TableCell>
                        <Badge variant={col.optimizer_status === "ok" ? "success" : "destructive"}>
                          {col.optimizer_status}
                        </Badge>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Points" tooltip={METRIC_INFO.points} /></TableCell>
                      <TableCell>{col.points_count.toLocaleString()}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Vectors" tooltip={METRIC_INFO.vectors} /></TableCell>
                      <TableCell>
                        {col.vectors_count.toLocaleString()}
                        {col.vectors_count !== col.indexed_vectors_count && (
                          <span className="ml-2 text-muted-foreground">
                            ({indexPercent}% indexed)
                          </span>
                        )}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Indexed" tooltip={METRIC_INFO.indexed} /></TableCell>
                      <TableCell>
                        {col.indexed_vectors_count.toLocaleString()}
                        {isIndexed && <Badge variant="success" className="ml-2 text-xs">ready</Badge>}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Segments" tooltip={METRIC_INFO.segments} /></TableCell>
                      <TableCell>{col.segments_count}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Vector Size" tooltip={METRIC_INFO["vector size"]} /></TableCell>
                      <TableCell>{col.vector_size ?? "—"}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="Distance" tooltip={METRIC_INFO.distance} /></TableCell>
                      <TableCell>{col.distance?.replace("Distance.", "") ?? "—"}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="HNSW (m / ef_construct)" tooltip={METRIC_INFO["hnsw (m / ef_construct)"]} /></TableCell>
                      <TableCell>
                        {col.hnsw_m != null && col.hnsw_ef_construct != null
                          ? `${col.hnsw_m} / ${col.hnsw_ef_construct}`
                          : "—"}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium"><TooltipLabel label="On Disk Payload" tooltip={METRIC_INFO["on disk payload"]} /></TableCell>
                      <TableCell>
                        <Badge variant={col.on_disk_payload ? "secondary" : "outline"}>
                          {col.on_disk_payload === true ? "Yes" : col.on_disk_payload === false ? "No" : "—"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          );
        })
      ) : (
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground text-center py-8">
              {data?.qdrant_status !== "ok"
                ? "Cannot connect to Qdrant"
                : "No collections found"}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
    </TooltipProvider>
  );
}
