"use client";
import { useHealth, useUsers, useGroups, useDocuments, useMetrics, useIngestRegistry } from "@/shared/api/hooks";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Skeleton } from "@/shared/ui/skeleton";
import {
  Activity, FileText, Server, Database, Cpu, Clock, Zap, HardDrive,
  AlertCircle, CheckCircle2, XCircle, BarChart3, Search, TrendingUp,
  Users, FolderOpen, Globe,
} from "lucide-react";

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function StatusIndicator({ status }: { status: string }) {
  const isOk = status === "ok";
  return (
    <span className={`inline-flex items-center gap-1.5 ${isOk ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
      {isOk ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
      <span className="text-sm font-medium">{isOk ? "Healthy" : "Error"}</span>
    </span>
  );
}

function ServiceCard({
  title, icon, status, latency, details
}: {
  title: string; icon: React.ReactNode; status: string;
  latency?: number | null; details?: string[];
}) {
  const isOk = status === "ok";
  return (
    <Card className={`transition-colors ${isOk ? "" : "border-red-200 dark:border-red-800"}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${isOk ? "bg-green-500/10" : "bg-red-500/10"}`}>
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            <StatusIndicator status={status} />
          </div>
        </div>
        {latency != null && (
          <div className="text-right">
            <div className="text-lg font-bold font-mono">{latency}</div>
            <div className="text-[10px] text-muted-foreground uppercase">ms</div>
          </div>
        )}
      </CardHeader>
      {details && details.length > 0 && (
        <CardContent className="pt-0">
          <div className="flex flex-wrap gap-1.5">
            {details.map((d, i) => (
              <Badge key={i} variant="secondary" className="text-xs font-mono">{d}</Badge>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function StatCard({ label, value, icon: Icon, color }: { label: string; value: string | number; icon: typeof BarChart3; color: string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${color}`}>
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <div className="text-2xl font-bold font-mono">{value}</div>
            <div className="text-xs text-muted-foreground">{label}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function AdminDashboardPage() {
  const { data: health, isLoading: hl } = useHealth();
  const { data: users } = useUsers();
  const { data: groups } = useGroups();
  const { data: documents, isLoading: dl } = useDocuments();
  const { data: metrics } = useMetrics();
  const { data: registry } = useIngestRegistry();

  const apiStatus = health?.checks?.api?.status || "—";
  const qdrantStatus = health?.checks?.qdrant?.status || "—";
  const ollamaStatus = health?.checks?.ollama?.status || "—";
  const postgresStatus = health?.checks?.postgres?.status || "—";

  const docsDone = documents?.filter((d) => d.status === "done").length || 0;
  const docsProcessing = documents?.filter((d) => d.status === "processing").length || 0;
  const docsFailed = documents?.filter((d) => d.status === "failed").length || 0;
  const totalChunks = documents?.reduce((sum, d) => sum + (d.chunks || 0), 0) || 0;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Admin Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1">System overview and health</p>
        </div>
        {health && (
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <Zap className="h-4 w-4" />
              <span>v{health.version}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Clock className="h-4 w-4" />
              <span>Uptime: {formatUptime(health.uptime_seconds)}</span>
            </div>
            {health.background_jobs.running > 0 && (
              <Badge variant="secondary">
                <Activity className="h-3 w-3 mr-1" />
                {health.background_jobs.running} job(s)
              </Badge>
            )}
          </div>
        )}
      </div>

      {/* Overall Status */}
      {health && (
        <Card className={`border-2 ${health.status === "healthy" ? "border-green-200 dark:border-green-800" : "border-amber-200 dark:border-amber-800"}`}>
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`p-3 rounded-full ${health.status === "healthy" ? "bg-green-500/10" : "bg-amber-500/10"}`}>
                  {health.status === "healthy" ? (
                    <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-400" />
                  ) : (
                    <AlertCircle className="h-6 w-6 text-amber-600 dark:text-amber-400" />
                  )}
                </div>
                <div>
                  <h2 className="text-lg font-semibold">
                    {health.status === "healthy" ? "All Systems Operational" : "System Degraded"}
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    {health.status === "healthy"
                      ? "All services are running normally"
                      : "One or more services are experiencing issues"}
                  </p>
                </div>
              </div>
              <Badge variant={health.status === "healthy" ? "success" : "destructive"} className="text-sm px-3 py-1">
                {health.status.toUpperCase()}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Service Cards */}
      {hl ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <Skeleton className="h-4 w-24" /><Skeleton className="h-4 w-4" />
              </CardHeader>
              <CardContent><Skeleton className="h-7 w-20" /></CardContent>
            </Card>
          ))}
        </div>
      ) : health ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ServiceCard title="API Server" icon={<Server className="h-5 w-5 text-blue-600 dark:text-blue-400" />} status={apiStatus} />
          <ServiceCard title="PostgreSQL" icon={<Database className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />} status={postgresStatus} latency={health.checks?.postgres?.latency_ms} />
          <ServiceCard title="Qdrant" icon={<HardDrive className="h-5 w-5 text-cyan-600 dark:text-cyan-400" />} status={qdrantStatus} latency={health.checks?.qdrant?.latency_ms} />
          <ServiceCard title="Ollama" icon={<Cpu className="h-5 w-5 text-orange-600 dark:text-orange-400" />} status={ollamaStatus} latency={health.checks?.ollama?.latency_ms} details={health.checks?.ollama?.models || undefined} />
        </div>
      ) : null}

      {/* Quick Stats */}
      {metrics && (
        <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
          <StatCard label="Users" value={users?.length || 0} icon={Users} color="bg-violet-500/10 text-violet-600 dark:text-violet-400" />
          <StatCard label="Groups" value={groups?.length || 0} icon={FolderOpen} color="bg-pink-500/10 text-pink-600 dark:text-pink-400" />
          <StatCard label="Documents" value={documents?.length || 0} icon={FileText} color="bg-blue-500/10 text-blue-600 dark:text-blue-400" />
          <StatCard label="Vector Points" value={Number(metrics.qdrant?.points || 0).toLocaleString()} icon={BarChart3} color="bg-cyan-500/10 text-cyan-600 dark:text-cyan-400" />
          <StatCard label="BM25 Index" value={Number(metrics.bm25?.index_size || 0).toLocaleString()} icon={Search} color="bg-purple-500/10 text-purple-600 dark:text-purple-400" />
          <StatCard label="Total Chunks" value={totalChunks.toLocaleString()} icon={TrendingUp} color="bg-green-500/10 text-green-600 dark:text-green-400" />
        </div>
      )}

      {/* Documents + RAG Pipeline */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Documents */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" />Documents</CardTitle>
            <CardDescription>Indexed document status</CardDescription>
          </CardHeader>
          <CardContent>
            {dl ? (
              <div className="space-y-3">{[1, 2].map((i) => <Skeleton key={i} className="h-10 w-full" />)}</div>
            ) : documents && documents.length > 0 ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center p-3 rounded-lg bg-green-500/5">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">{docsDone}</div>
                    <div className="text-xs text-muted-foreground">Done</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-amber-500/5">
                    <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">{docsProcessing}</div>
                    <div className="text-xs text-muted-foreground">Processing</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-red-500/5">
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">{docsFailed}</div>
                    <div className="text-xs text-muted-foreground">Failed</div>
                  </div>
                </div>
                <div className="space-y-2">
                  {documents.slice(0, 5).map((d) => (
                    <div key={d.id} className="flex items-center justify-between rounded-md border p-2.5 hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-2 min-w-0">
                        <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                        <span className="text-sm font-medium truncate">{d.filename}</span>
                      </div>
                      <Badge variant={d.status === "done" ? "success" : d.status === "failed" ? "destructive" : "secondary"}>{d.status}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">No documents yet</p>
            )}
          </CardContent>
        </Card>

        {/* RAG Pipeline */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" />RAG Pipeline</CardTitle>
            <CardDescription>Query statistics and performance</CardDescription>
          </CardHeader>
          <CardContent>
            {metrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-3 rounded-lg bg-blue-500/5">
                    <div className="text-2xl font-bold">{Number(metrics.rag?.queries_total) || 0}</div>
                    <div className="text-xs text-muted-foreground">Total Queries</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-amber-500/5">
                    <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">{Number(metrics.rag?.not_found_total) || 0}</div>
                    <div className="text-xs text-muted-foreground">Not Found</div>
                  </div>
                </div>
                {metrics.rag?.stage_latency && typeof metrics.rag.stage_latency === "object" && "count" in metrics.rag.stage_latency && (
                  <div className="pt-3 border-t">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Avg Latency</span>
                      <span className="font-mono font-bold">
                        {Number((metrics.rag.stage_latency as Record<string, number>).count) > 0
                          ? `${(Number((metrics.rag.stage_latency as Record<string, number>).sum) / Number((metrics.rag.stage_latency as Record<string, number>).count)).toFixed(1)}s`
                          : "—"}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Total Time</span>
                      <span className="font-mono">{Number((metrics.rag.stage_latency as Record<string, number>).sum || 0).toFixed(1)}s</span>
                    </div>
                  </div>
                )}
                {Number(metrics.rag?.queries_total) === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-2">No queries yet</p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">Loading metrics...</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* DB Pool + Ingestion */}
      {metrics && (
        <div className="grid gap-4 md:grid-cols-2">
          {/* DB Pool */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Database className="h-5 w-5" />Database Connection Pool</CardTitle>
              <CardDescription>PostgreSQL connection utilization</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { label: "In Use", value: metrics.db_pool?.connections_in_use ?? metrics.db_pool?.in_use ?? 0, color: "bg-blue-500" },
                  { label: "Idle", value: metrics.db_pool?.connections_idle ?? metrics.db_pool?.idle ?? 0, color: "bg-green-500" },
                  { label: "Overflow", value: metrics.db_pool?.overflow ?? 0, color: "bg-amber-500" },
                ].map(({ label, value, color }) => {
                  const total = Number(metrics.db_pool?.connections_in_use ?? metrics.db_pool?.in_use ?? 0) +
                    Number(metrics.db_pool?.connections_idle ?? metrics.db_pool?.idle ?? 0) || 1;
                  const pct = (Number(value) / total) * 100;
                  return (
                    <div key={label}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-muted-foreground">{label}</span>
                        <span className="font-mono font-bold">{Number(value)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-muted overflow-hidden">
                        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${Math.min(pct, 100)}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Ingestion */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5" />Ingestion Stats</CardTitle>
              <CardDescription>Document processing totals</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Indexed Files</span>
                  <span className="font-mono font-bold">{registry?.total_files || 0}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Total Chunks</span>
                  <span className="font-mono font-bold">{registry?.total_chunks?.toLocaleString() || 0}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Documents Ingested</span>
                  <span className="font-mono font-bold">{Number(metrics.ingestion?.documents_total) || 0}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Files Processed</span>
                  <span className="font-mono font-bold">{Number(metrics.ingestion?.files_total) || 0}</span>
                </div>
                {metrics.ingestion?.duration && typeof metrics.ingestion.duration === "object" && "sum" in (metrics.ingestion.duration as Record<string, unknown>) && (
                  <div className="pt-3 border-t">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Total Processing Time</span>
                      <span className="font-mono">{Number((metrics.ingestion.duration as Record<string, number>).sum || 0).toFixed(1)}s</span>
                    </div>
                    {(metrics.ingestion.duration as Record<string, number>).count > 0 && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Avg per Document</span>
                        <span className="font-mono">
                          {(Number((metrics.ingestion.duration as Record<string, number>).sum) / Number((metrics.ingestion.duration as Record<string, number>).count)).toFixed(1)}s
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
