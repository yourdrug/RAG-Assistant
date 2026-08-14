"use client";
import { Activity, BarChart3, Cpu, Database, Globe, HardDrive, RefreshCw, Zap } from "lucide-react";
import { useMetrics } from "@/shared/api/hooks";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";

function formatBytes(bytes: number) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function MetricCard({
  title,
  icon: Icon,
  description,
  children,
}: {
  title: string;
  icon: typeof BarChart3;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          {title}
        </CardTitle>
        {description && <CardDescription className="text-xs">{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function Row({ label, value, badge }: { label: string; value: string | number; badge?: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        {badge && (
          <Badge variant="outline" className="text-[10px] px-1 py-0">
            {badge}
          </Badge>
        )}
        <span className="font-mono font-bold">{value}</span>
      </div>
    </div>
  );
}

export function MonitoringPage() {
  const { data: metrics, isLoading, refetch } = useMetrics();

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Monitoring</h1>
            <p className="text-muted-foreground">System metrics</p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full" />
          ))}
        </div>
      </div>
    );
  }

  const ingestion = metrics?.ingestion || {};
  const byStatus = (ingestion.by_status || {}) as Record<string, number>;
  const httpReqs = metrics?.http_requests || {};
  const byEndpoint = (httpReqs.by_endpoint || {}) as Record<string, number>;

  // Top endpoints sorted by count
  const topEndpoints = Object.entries(byEndpoint)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8);

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
        <MetricCard title="PostgreSQL Pool" icon={Database} description="Connection utilization">
          <div className="space-y-3">
            <Row
              label="In Use"
              value={metrics?.db_pool?.connections_in_use ?? metrics?.db_pool?.in_use ?? 0}
            />
            <Row
              label="Idle"
              value={metrics?.db_pool?.connections_idle ?? metrics?.db_pool?.idle ?? 0}
            />
            <Row label="Overflow" value={metrics?.db_pool?.overflow ?? 0} />
            {(() => {
              const inUse = Number(
                metrics?.db_pool?.connections_in_use ?? metrics?.db_pool?.in_use ?? 0,
              );
              const idle = Number(
                metrics?.db_pool?.connections_idle ?? metrics?.db_pool?.idle ?? 0,
              );
              const total = inUse + idle || 1;
              return (
                <div className="pt-2 border-t">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted-foreground">Utilization</span>
                    <span className="font-mono">{((inUse / total) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-indigo-500 transition-all"
                      style={{ width: `${(inUse / total) * 100}%` }}
                    />
                  </div>
                </div>
              );
            })()}
          </div>
        </MetricCard>

        {/* Qdrant */}
        <MetricCard title="Qdrant Vector DB" icon={HardDrive} description="Vector storage">
          <div className="space-y-3">
            <div className="text-3xl font-bold font-mono">
              {metrics?.qdrant?.points?.toLocaleString() ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">Total points</p>
          </div>
        </MetricCard>

        {/* BM25 */}
        <MetricCard title="BM25 Index" icon={Zap} description="Full-text search">
          <div className="space-y-3">
            <div className="text-3xl font-bold font-mono">
              {metrics?.bm25?.index_size?.toLocaleString() ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">Documents indexed</p>
          </div>
        </MetricCard>

        {/* Ollama */}
        <MetricCard title="Ollama LLM" icon={Cpu} description="Loaded models">
          {metrics?.ollama && metrics.ollama.length > 0 ? (
            <div className="space-y-3">
              {metrics.ollama.map((m) => (
                <div key={m.model} className="space-y-1">
                  <Badge variant="secondary" className="text-xs">
                    {m.model}
                  </Badge>
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
        <MetricCard title="RAG Pipeline" icon={Activity} description="Query performance">
          <div className="space-y-2">
            <Row label="Total Queries" value={Number(metrics?.rag?.queries_total) || 0} />
            <Row
              label="Not Found"
              value={Number(metrics?.rag?.not_found_total) || 0}
              badge="miss"
            />
            {metrics?.rag?.stage_latency && typeof metrics.rag.stage_latency === "object" && (
              <div className="pt-2 border-t">
                {"count" in (metrics.rag.stage_latency as Record<string, unknown>) &&
                  Number((metrics.rag.stage_latency as Record<string, number>).count) > 0 && (
                    <>
                      <Row
                        label="Avg Latency"
                        value={`${(Number((metrics.rag.stage_latency as Record<string, number>).sum) / Number((metrics.rag.stage_latency as Record<string, number>).count)).toFixed(2)}s`}
                      />
                      <Row
                        label="Total Time"
                        value={`${Number((metrics.rag.stage_latency as Record<string, number>).sum).toFixed(1)}s`}
                      />
                    </>
                  )}
              </div>
            )}
          </div>
        </MetricCard>

        {/* Ingestion */}
        <MetricCard title="Ingestion" icon={Cpu} description="Processing stats">
          <div className="space-y-2">
            <Row label="Documents" value={Number(ingestion.documents_total) || 0} />
            <Row label="Chunks" value={Number(ingestion.chunks_total) || 0} />
            <Row label="Files" value={Number(ingestion.files_total) || 0} />
            {Object.keys(byStatus).length > 0 && (
              <div className="pt-2 border-t">
                <p className="text-xs text-muted-foreground mb-1">By Status</p>
                {Object.entries(byStatus).map(([status, count]) => (
                  <Row key={status} label={status} value={Number(count)} />
                ))}
              </div>
            )}
            {ingestion.duration &&
              typeof ingestion.duration === "object" &&
              "sum" in (ingestion.duration as Record<string, unknown>) &&
              Number((ingestion.duration as Record<string, number>).count) > 0 && (
                <div className="pt-2 border-t">
                  <Row
                    label="Avg Duration"
                    value={`${(Number((ingestion.duration as Record<string, number>).sum) / Number((ingestion.duration as Record<string, number>).count)).toFixed(1)}s`}
                  />
                </div>
              )}
          </div>
        </MetricCard>

        {/* HTTP Requests */}
        <MetricCard
          title="HTTP Requests"
          icon={Globe}
          description={`Total: ${Number(httpReqs.total) || 0}`}
        >
          <div className="space-y-2">
            {topEndpoints.length > 0 ? (
              topEndpoints.map(([endpoint, count]) => {
                const parts = endpoint.split("_");
                const method = parts.length >= 2 ? parts[parts.length - 2] : "";
                const path = parts.slice(0, -2).join("_") || endpoint;
                return (
                  <div key={endpoint} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <Badge variant="outline" className="text-[10px] px-1 py-0 shrink-0">
                        {method.toUpperCase()}
                      </Badge>
                      <span className="truncate font-mono text-muted-foreground">{path}</span>
                    </div>
                    <span className="font-mono font-bold shrink-0">{Number(count)}</span>
                  </div>
                );
              })
            ) : (
              <p className="text-sm text-muted-foreground">No requests recorded</p>
            )}
          </div>
        </MetricCard>
      </div>
    </div>
  );
}
