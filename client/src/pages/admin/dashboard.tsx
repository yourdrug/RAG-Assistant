"use client";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  FileText,
  FolderOpen,
  Globe,
  HardDrive,
  Server,
  TrendingUp,
  Users,
  XCircle,
  Zap,
} from "lucide-react";
import {
  useDocuments,
  useGroups,
  useHealth,
  useIngestRegistry,
  useMetrics,
  useUsers,
} from "@/shared/api/hooks";
import { Badge } from "@/shared/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";

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
    <span
      className={`inline-flex items-center gap-1.5 ${isOk ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}
    >
      {isOk ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
      <span className="text-sm font-medium">{isOk ? "Healthy" : "Error"}</span>
    </span>
  );
}

function ServiceCard({
  title,
  icon,
  status,
  latency,
  details,
  showDetails = false,
  isLoading,
}: {
  title: string;
  icon: React.ReactNode;
  status?: string;
  latency?: number | null;
  details?: string[];
  showDetails?: boolean;
  isLoading: boolean;
}) {
  const isOk = status === "ok";
  return (
    <Card
      className={`min-w-0 transition-colors ${status && !isOk ? "border-red-200 dark:border-red-800" : ""}`}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <div className="flex items-center gap-3">
          <div
            className={`p-2 rounded-lg ${!status ? "bg-muted" : isOk ? "bg-green-500/10" : "bg-red-500/10"}`}
          >
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            <div className="flex h-6 items-center">
              {status ? (
                <StatusIndicator status={status} />
              ) : isLoading ? (
                <Skeleton className="h-4 w-16" />
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </div>
          </div>
        </div>
        {latency != null && (
          <div className="text-right">
            <div className="text-lg font-bold font-mono">{latency}</div>
            <div className="text-[10px] text-muted-foreground uppercase">ms</div>
          </div>
        )}
      </CardHeader>
      {showDetails && (
        <CardContent className="pt-0">
          <div className="flex h-8 items-start gap-1.5 overflow-x-auto">
            {details?.map((d, i) => (
              <Badge
                key={i}
                variant="secondary"
                className="shrink-0 whitespace-nowrap text-xs font-mono"
              >
                {d}
              </Badge>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function StatCard({
  label,
  value,
  isLoading,
  icon: Icon,
  color,
}: {
  label: string;
  value?: string | number;
  isLoading: boolean;
  icon: typeof BarChart3;
  color: string;
}) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${color}`}>
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <div className="flex h-8 items-center text-2xl font-bold font-mono">
              {isLoading ? <Skeleton className="h-5 w-10" /> : (value ?? "—")}
            </div>
            <div className="text-xs text-muted-foreground">{label}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function AdminDashboardPage() {
  const { data: health, isLoading: hl } = useHealth();
  const { data: users, isLoading: usersLoading } = useUsers();
  const { data: groups, isLoading: groupsLoading } = useGroups();
  const { data: documents, isLoading: dl } = useDocuments();
  const { data: metrics, isLoading: ml } = useMetrics();
  const { data: registry } = useIngestRegistry();

  const apiStatus = health ? health.checks?.api?.status || "—" : undefined;
  const qdrantStatus = health ? health.checks?.qdrant?.status || "—" : undefined;
  const llmStatus = health ? health.checks?.llm?.status || "—" : undefined;
  const postgresStatus = health ? health.checks?.postgres?.status || "—" : undefined;

  const docsDone = documents?.filter((d) => d.status === "done").length || 0;
  const docsProcessing =
    documents?.filter((d) => d.status === "processing" || d.status === "indexing").length || 0;
  const docsFailed = documents?.filter((d) => d.status === "failed").length || 0;
  const totalChunks = documents?.reduce((sum, d) => sum + (d.chunks || 0), 0) || 0;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="grid gap-2 sm:grid-cols-[1fr_auto] sm:items-center">
        <div>
          <h1 className="text-2xl font-bold">Admin Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1">System overview and health</p>
        </div>
        <div className="flex min-h-8 items-center gap-2 whitespace-nowrap text-sm text-muted-foreground sm:gap-4">
          {health && (
            <>
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
            </>
          )}
        </div>
      </div>

      {/* Overall Status */}
      <Card
        className={`border-2 ${health ? (health.status === "healthy" ? "border-green-200 dark:border-green-800" : "border-amber-200 dark:border-amber-800") : ""}`}
      >
        <CardContent className="p-3 sm:px-6 sm:py-4">
          {health ? (
            <div className="grid min-h-32 content-center items-center justify-items-start gap-2 sm:flex sm:min-h-12 sm:justify-between">
              <div className="flex min-w-0 items-center gap-3">
                <div
                  className={`p-3 rounded-full ${health.status === "healthy" ? "bg-green-500/10" : "bg-amber-500/10"}`}
                >
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
              <Badge
                variant={health.status === "healthy" ? "success" : "destructive"}
                className="text-sm px-3 py-1"
              >
                {health.status.toUpperCase()}
              </Badge>
            </div>
          ) : (
            <div className="flex min-h-32 items-center gap-3 sm:min-h-12">
              {hl ? (
                <>
                  <Skeleton className="h-12 w-12 shrink-0 rounded-full" />
                  <div className="w-52 min-w-0 space-y-2">
                    <Skeleton className="h-5 w-5/6" />
                    <Skeleton className="h-4 w-full" />
                  </div>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">System status unavailable</p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Service Cards */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <ServiceCard
          title="API Server"
          icon={<Server className="h-5 w-5 text-blue-600 dark:text-blue-400" />}
          status={apiStatus}
          isLoading={hl}
        />
        <ServiceCard
          title="PostgreSQL"
          icon={<Database className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />}
          status={postgresStatus}
          isLoading={hl}
          latency={health?.checks?.postgres?.latency_ms}
        />
        <ServiceCard
          title="Qdrant"
          icon={<HardDrive className="h-5 w-5 text-cyan-600 dark:text-cyan-400" />}
          status={qdrantStatus}
          isLoading={hl}
          latency={health?.checks?.qdrant?.latency_ms}
        />
        <ServiceCard
          title="LLM"
          icon={<Cpu className="h-5 w-5 text-orange-600 dark:text-orange-400" />}
          status={llmStatus}
          isLoading={hl}
          latency={health?.checks?.llm?.latency_ms}
          details={health?.checks?.llm?.models || undefined}
          showDetails
        />
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-5">
        <StatCard
          label="Users"
          value={users ? (users.total ?? users.users?.length ?? 0) : undefined}
          isLoading={usersLoading}
          icon={Users}
          color="bg-violet-500/10 text-violet-600 dark:text-violet-400"
        />
        <StatCard
          label="Groups"
          value={groups?.length}
          isLoading={groupsLoading}
          icon={FolderOpen}
          color="bg-pink-500/10 text-pink-600 dark:text-pink-400"
        />
        <StatCard
          label="Documents"
          value={documents?.length}
          isLoading={dl}
          icon={FileText}
          color="bg-blue-500/10 text-blue-600 dark:text-blue-400"
        />
        <StatCard
          label="Vector Points"
          value={metrics ? Number(metrics.qdrant?.points || 0).toLocaleString() : undefined}
          isLoading={ml}
          icon={BarChart3}
          color="bg-cyan-500/10 text-cyan-600 dark:text-cyan-400"
        />
        <StatCard
          label="Total Chunks"
          value={documents ? totalChunks.toLocaleString() : undefined}
          isLoading={dl}
          icon={TrendingUp}
          color="bg-green-500/10 text-green-600 dark:text-green-400"
        />
      </div>

      {/* Documents + RAG Pipeline */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Documents */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Documents
            </CardTitle>
            <CardDescription>Indexed document status</CardDescription>
          </CardHeader>
          <CardContent className="min-h-92">
            {dl ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-19" />
                  ))}
                </div>
                <div className="space-y-2">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Skeleton key={i} className="h-11" />
                  ))}
                </div>
              </div>
            ) : documents && documents.length > 0 ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center p-3 rounded-lg bg-green-500/5">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {docsDone}
                    </div>
                    <div className="text-xs text-muted-foreground">Done</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-amber-500/5">
                    <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
                      {docsProcessing}
                    </div>
                    <div className="text-xs text-muted-foreground">Processing</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-red-500/5">
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                      {docsFailed}
                    </div>
                    <div className="text-xs text-muted-foreground">Failed</div>
                  </div>
                </div>
                <div className="space-y-2">
                  {documents.slice(0, 5).map((d) => (
                    <div
                      key={d.id}
                      className="flex items-center justify-between rounded-md border p-2.5 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                        <span className="text-sm font-medium truncate">{d.filename}</span>
                      </div>
                      <Badge
                        variant={
                          d.status === "done"
                            ? "success"
                            : d.status === "failed"
                              ? "destructive"
                              : d.status === "indexing"
                                ? "default"
                                : "secondary"
                        }
                      >
                        {d.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">
                {documents ? "No documents yet" : "Documents unavailable"}
              </p>
            )}
          </CardContent>
        </Card>

        {/* RAG Pipeline */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              RAG Pipeline
            </CardTitle>
            <CardDescription>Query statistics and performance</CardDescription>
          </CardHeader>
          <CardContent className="min-h-48">
            {metrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-3 rounded-lg bg-blue-500/5">
                    <div className="text-2xl font-bold">
                      {Number(metrics.rag?.queries_total) || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Total Queries</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-amber-500/5">
                    <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
                      {Number(metrics.rag?.not_found_total) || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Not Found</div>
                  </div>
                </div>
                {metrics.rag?.stage_latency &&
                  typeof metrics.rag.stage_latency === "object" &&
                  "count" in metrics.rag.stage_latency && (
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
                        <span className="font-mono">
                          {Number(
                            (metrics.rag.stage_latency as Record<string, number>).sum || 0,
                          ).toFixed(1)}
                          s
                        </span>
                      </div>
                    </div>
                  )}
                {Number(metrics.rag?.queries_total) === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-2">No queries yet</p>
                )}
              </div>
            ) : ml ? (
              <div className="grid grid-cols-2 gap-3">
                <Skeleton className="h-19" />
                <Skeleton className="h-19" />
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">Metrics unavailable</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* DB Pool + Ingestion */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* DB Pool */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Database Connection Pool
            </CardTitle>
            <CardDescription>PostgreSQL connection utilization</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="min-h-30 space-y-3">
              {metrics ? (
                (() => {
                  const inUse = Number(
                    metrics.db_pool?.connections_in_use ?? metrics.db_pool?.in_use ?? 0,
                  );
                  const idle = Number(
                    metrics.db_pool?.connections_idle ?? metrics.db_pool?.idle ?? 0,
                  );
                  const poolSize = 100;
                  const available = Math.max(0, poolSize - inUse - idle);
                  const total = inUse + idle + available || 1;
                  return [
                    {
                      label: "In Use",
                      value: inUse,
                      color: "bg-blue-500",
                    },
                    {
                      label: "Idle",
                      value: idle,
                      color: "bg-green-500",
                    },
                    {
                      label: "Available",
                      value: available,
                      color: "bg-emerald-500",
                    },
                  ].map(({ label, value, color }) => {
                    const pct = (value / total) * 100;
                    return (
                      <div key={label}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-muted-foreground">{label}</span>
                          <span className="font-mono font-bold">{value}</span>
                        </div>
                        <div className="h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className={`h-full rounded-full ${color} transition-all`}
                            style={{ width: `${Math.min(pct, 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  });
                })()
              ) : ml ? (
                [1, 2, 3].map((i) => <Skeleton key={i} className="h-8" />)
              ) : (
                <p className="text-sm text-muted-foreground">Metrics unavailable</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Ingestion */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Ingestion Stats
            </CardTitle>
            <CardDescription>Document processing totals</CardDescription>
          </CardHeader>
          <CardContent className="min-h-52">
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Indexed Files</span>
                <span className="font-mono font-bold">{registry?.total_files ?? "—"}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Total Chunks</span>
                <span className="font-mono font-bold">
                  {registry?.total_chunks?.toLocaleString() ?? "—"}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Documents Ingested</span>
                <span className="font-mono font-bold">
                  {metrics ? Number(metrics.ingestion?.documents_total) || 0 : "—"}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Files Processed</span>
                <span className="font-mono font-bold">
                  {metrics ? Number(metrics.ingestion?.files_total) || 0 : "—"}
                </span>
              </div>
              {metrics?.ingestion?.duration &&
                typeof metrics.ingestion.duration === "object" &&
                "sum" in (metrics.ingestion.duration as Record<string, unknown>) && (
                  <div className="pt-3 border-t">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Total Processing Time</span>
                      <span className="font-mono">
                        {Number(
                          (metrics.ingestion.duration as Record<string, number>).sum || 0,
                        ).toFixed(1)}
                        s
                      </span>
                    </div>
                    {(metrics.ingestion.duration as Record<string, number>).count > 0 && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Avg per Document</span>
                        <span className="font-mono">
                          {(
                            Number((metrics.ingestion.duration as Record<string, number>).sum) /
                            Number((metrics.ingestion.duration as Record<string, number>).count)
                          ).toFixed(1)}
                          s
                        </span>
                      </div>
                    )}
                  </div>
                )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* LLM Models + HTTP Requests + Ingestion by Status */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* LLM Models */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              LLM
            </CardTitle>
            <CardDescription>OpenAI-compatible endpoint</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <Badge variant="secondary" className="max-w-full text-xs wrap-anywhere">
                {health?.checks?.llm?.models?.[0] || "—"}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* HTTP Requests */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Globe className="h-5 w-5" />
              HTTP Requests
            </CardTitle>
            <CardDescription>
              Total: {metrics ? Number(metrics.http_requests?.total) || 0 : "—"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {(() => {
              const byEndpoint = (metrics?.http_requests?.by_endpoint || {}) as Record<
                string,
                number
              >;
              const topEndpoints = Object.entries(byEndpoint)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 6);
              return topEndpoints.length > 0 ? (
                <div className="space-y-2">
                  {topEndpoints.map(([endpoint, count]) => {
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
                  })}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {metrics
                    ? "No requests recorded"
                    : ml
                      ? "Loading metrics…"
                      : "Metrics unavailable"}
                </p>
              );
            })()}
          </CardContent>
        </Card>

        {/* Ingestion by Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Ingestion by Status
            </CardTitle>
            <CardDescription>Document processing breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            {(() => {
              // Ingestion runs in worker processes, so the API's Prometheus registry never
              // sees it; the persisted document statuses are the source of truth.
              const byStatus: Record<string, number> = {};
              for (const d of documents || []) {
                byStatus[d.status] = (byStatus[d.status] || 0) + 1;
              }
              const statusEntries = Object.entries(byStatus).sort(([, a], [, b]) => b - a);
              return statusEntries.length > 0 ? (
                <div className="space-y-2">
                  {statusEntries.map(([status, count]) => (
                    <div key={status} className="flex justify-between text-sm">
                      <span className="text-muted-foreground">{status}</span>
                      <span className="font-mono font-bold">{Number(count)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {metrics ? "No ingestion data" : ml ? "Loading metrics…" : "Metrics unavailable"}
                </p>
              );
            })()}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
