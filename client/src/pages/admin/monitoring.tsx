"use client";
import { useMetrics } from "@/shared/api/hooks";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { BarChart3, Database, Cpu, Activity, RefreshCw, Zap } from "lucide-react";

function formatBytes(bytes: number) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function MetricCard({ title, icon: Icon, children }: { title: string; icon: typeof BarChart3; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export function MonitoringPage() {
  const { data: metrics, isLoading, refetch } = useMetrics();

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div><h1 className="text-2xl font-bold">Monitoring</h1><p className="text-muted-foreground">System metrics</p></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-40 w-full" />)}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Monitoring</h1>
          <p className="text-muted-foreground">System metrics (auto-refresh 15s)</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" /> Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Database Pool */}
        <MetricCard title="PostgreSQL Pool" icon={Database}>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">In Use</span>
              <span className="font-mono font-bold">{metrics?.db_pool?.in_use ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Idle</span>
              <span className="font-mono font-bold">{metrics?.db_pool?.idle ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Overflow</span>
              <span className="font-mono font-bold">{metrics?.db_pool?.overflow ?? 0}</span>
            </div>
          </div>
        </MetricCard>

        {/* Qdrant */}
        <MetricCard title="Qdrant Vector DB" icon={BarChart3}>
          <div className="space-y-2">
            <div className="text-3xl font-bold font-mono">{metrics?.qdrant?.points?.toLocaleString() ?? 0}</div>
            <p className="text-xs text-muted-foreground">Total points</p>
          </div>
        </MetricCard>

        {/* BM25 */}
        <MetricCard title="BM25 Index" icon={Zap}>
          <div className="space-y-2">
            <div className="text-3xl font-bold font-mono">{metrics?.bm25?.index_size?.toLocaleString() ?? 0}</div>
            <p className="text-xs text-muted-foreground">Documents indexed</p>
          </div>
        </MetricCard>

        {/* Ollama */}
        <MetricCard title="Ollama LLM" icon={Cpu}>
          {metrics?.ollama && metrics.ollama.length > 0 ? (
            <div className="space-y-3">
              {metrics.ollama.map((m) => (
                <div key={m.model} className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">{m.model}</Badge>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">GPU</span>
                    <span className="font-mono">{formatBytes(m.gpu_bytes)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">RAM</span>
                    <span className="font-mono">{formatBytes(m.ram_bytes)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No models loaded</p>
          )}
        </MetricCard>

        {/* RAG */}
        <MetricCard title="RAG Pipeline" icon={Activity}>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Total Queries</span>
              <span className="font-mono font-bold">{(metrics?.rag?.queries_total as number) ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Not Found</span>
              <span className="font-mono font-bold text-amber-600">{(metrics?.rag?.not_found_total as number) ?? 0}</span>
            </div>
            {metrics?.rag?.stage_latency && typeof metrics.rag.stage_latency === "object" && (
              <div className="pt-2 border-t">
                <p className="text-xs text-muted-foreground mb-1">Stage Latency</p>
                {"count" in (metrics.rag.stage_latency as Record<string, unknown>) && (
                  <div className="flex justify-between text-xs">
                    <span>Samples</span>
                    <span className="font-mono">{(metrics.rag.stage_latency as Record<string, number>).count}</span>
                  </div>
                )}
                {"sum" in (metrics.rag.stage_latency as Record<string, unknown>) && (
                  <div className="flex justify-between text-xs">
                    <span>Total time</span>
                    <span className="font-mono">{Number((metrics.rag.stage_latency as Record<string, number>).sum).toFixed(2)}s</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </MetricCard>

        {/* Ingestion */}
        <MetricCard title="Ingestion" icon={Cpu}>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Documents</span>
              <span className="font-mono font-bold">{(metrics?.ingestion?.documents_total as number) ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Chunks</span>
              <span className="font-mono font-bold">{(metrics?.ingestion?.chunks_total as number) ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Files</span>
              <span className="font-mono font-bold">{(metrics?.ingestion?.files_total as number) ?? 0}</span>
            </div>
          </div>
        </MetricCard>
      </div>
    </div>
  );
}
