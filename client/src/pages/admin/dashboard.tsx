"use client";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  Cloud,
  Cpu,
  Database,
  FileText,
  FolderOpen,
  HardDrive,
  ListTodo,
  MemoryStick,
  Server,
  Users,
  XCircle,
} from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useCurrentUser, useGroups, useHealth, useMetrics, useUsers } from "@/shared/api/hooks";
import { useRollingChatLogStats } from "@/shared/api/hooks/use-chat-logs";
import { documentStatuses, useDocumentOverview } from "@/shared/api/hooks/use-documents";
import { useOperationalHealth } from "@/shared/api/hooks/use-health";
import type { DocumentOverview, HealthCheck, OperationalHealthResponse } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";
import { chatLogsUrl, DashboardRag, DashboardRagBreakdown } from "./dashboard-rag";

const periods = [
  { hours: 24, label: "24h" },
  { hours: 168, label: "7d" },
  { hours: 720, label: "30d" },
];

const serviceStatusLabels: Record<string, string> = {
  ok: "Healthy",
  idle: "Idle",
  queued: "Queued",
  degraded: "Degraded",
};

function formatUptime(seconds: number) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function UpdatedAgo({ timestamp }: { timestamp: number }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const seconds = Math.max(0, Math.floor((now - timestamp) / 1000));
  return <span>Health updated {seconds}s ago</span>;
}

function ServiceCard({
  title,
  icon,
  check,
  loading,
  children,
  className = "",
}: {
  title: string;
  icon: ReactNode;
  check?: HealthCheck;
  loading: boolean;
  children?: ReactNode;
  className?: string;
}) {
  const healthy = check?.status === "ok";
  const neutralStatus = check?.status === "idle" || check?.status === "queued";
  return (
    <Card
      className={`min-w-0 ${healthy ? "border-green-500/60 dark:border-green-400/50" : ""} ${className}`}
    >
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0 p-4">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-muted p-2 text-muted-foreground">{icon}</div>
          <div className="space-y-1">
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            {check ? (
              <span
                className={`flex items-center gap-1.5 text-sm ${neutralStatus ? "text-muted-foreground" : healthy ? "text-green-600 dark:text-green-400" : "text-destructive"}`}
              >
                {neutralStatus ? (
                  <Clock size={14} />
                ) : healthy ? (
                  <CheckCircle2 size={14} />
                ) : (
                  <XCircle size={14} />
                )}
                {serviceStatusLabels[check.status] ?? "Error"}
              </span>
            ) : loading ? (
              <div className="flex h-5 items-center">
                <Skeleton className="h-4 w-16" />
              </div>
            ) : (
              <span className="text-sm text-muted-foreground">Unavailable</span>
            )}
          </div>
        </div>
        {check?.latency_ms != null && (
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {check.latency_ms} ms
          </span>
        )}
      </CardHeader>
      {!!check?.models?.length && (
        <CardContent className="flex flex-wrap gap-1.5 px-4 pb-4">
          {check.models.map((model) => (
            <Badge key={model} variant="secondary" className="max-w-full wrap-anywhere">
              {model}
            </Badge>
          ))}
        </CardContent>
      )}
      {children}
    </Card>
  );
}

function IndexingWorkersCard({
  indexing,
  loading,
}: {
  indexing?: OperationalHealthResponse["indexing"];
  loading: boolean;
}) {
  const queue = indexing?.queue;
  const oldest = queue?.oldest_ready_seconds;
  let status = "idle";
  if (queue?.pending) status = "queued";
  if (queue?.running) status = "ok";
  if (indexing?.degraded) status = "degraded";
  const showMetrics = queue && status !== "idle";

  return (
    <ServiceCard
      title="Indexing workers"
      icon={<ListTodo size={20} />}
      check={queue ? { status } : undefined}
      loading={loading}
      className="sm:col-span-2"
    >
      <CardContent className="flex min-h-25 items-start px-4 pb-4 sm:min-h-13">
        {showMetrics ? (
          <dl className="grid w-full grid-cols-2 gap-3 text-xs text-muted-foreground sm:grid-cols-4">
            {[
              { label: "Pending", value: queue.pending },
              { label: "Running", value: queue.running },
              {
                label: "Oldest ready job",
                value: oldest == null ? "None" : oldest < 60 ? `${oldest}s` : formatUptime(oldest),
              },
              { label: "Expired leases", value: queue.expired_leases },
            ].map(({ label, value }) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd className="mt-1 font-semibold tabular-nums text-foreground">{value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="text-xs text-muted-foreground">
            {loading
              ? "Checking queue…"
              : queue
                ? "No jobs queued or running."
                : "Queue statistics unavailable."}
          </p>
        )}
      </CardContent>
    </ServiceCard>
  );
}

function StatCard({
  label,
  value,
  icon,
  to,
  loading,
  children,
}: {
  label: string;
  value?: number | string;
  icon: ReactNode;
  to?: string;
  loading?: boolean;
  children?: ReactNode;
}) {
  const stat = (
    <div className="flex items-center gap-3">
      <div className="rounded-lg bg-primary/10 p-2 text-primary">{icon}</div>
      <div className="min-w-0">
        <div className="text-xs font-medium text-muted-foreground">{label}</div>
        {loading ? (
          <Skeleton className="mt-1 h-6 w-14" />
        ) : (
          <div className="text-lg font-semibold tabular-nums">{value?.toLocaleString() ?? "—"}</div>
        )}
      </div>
    </div>
  );
  return (
    <Card className="min-w-0">
      {to ? (
        <Link to={to} className="block rounded-xl p-4 hover:bg-muted/50 focus-visible:outline-ring">
          {stat}
        </Link>
      ) : (
        <div className="p-4">{stat}</div>
      )}
      {children}
    </Card>
  );
}

function DocumentStatusCard({ data, loading }: { data?: DocumentOverview; loading: boolean }) {
  return (
    <Card className="min-w-0">
      <CardHeader className="p-4">
        <CardTitle>Documents by status</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 px-4 pb-4">
        {loading ? (
          <Skeleton className="h-28 w-full" />
        ) : data ? (
          <>
            <div className="flex flex-wrap gap-2">
              {documentStatuses.map((status) => (
                <Link
                  key={status}
                  to={`/admin/documents?status=${status}`}
                  className="rounded-md border px-3 py-2 text-sm capitalize hover:bg-muted focus-visible:outline-ring"
                >
                  <span
                    className={
                      status === "failed" && data.by_status.failed ? "text-destructive" : ""
                    }
                  >
                    {status}
                  </span>{" "}
                  <span className="font-semibold tabular-nums">
                    {data.by_status[status].toLocaleString()}
                  </span>
                </Link>
              ))}
            </div>
            <div className="space-y-2 border-t pt-3">
              <h3 className="text-sm font-semibold">Latest documents with errors</h3>
              {data.failed.length ? (
                <ul className="divide-y">
                  {data.failed.map((doc) => (
                    <li key={doc.id} className="space-y-1 py-2">
                      <p className="wrap-anywhere text-sm font-medium">{doc.filename}</p>
                      <p className="wrap-anywhere whitespace-pre-wrap text-xs text-muted-foreground">
                        {doc.error_message || "No error message recorded"}
                      </p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">No failed documents</p>
              )}
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Document statistics unavailable</p>
        )}
      </CardContent>
    </Card>
  );
}

export function AdminDashboardPage() {
  const [hours, setHours] = useState(168);
  const { data: user, isLoading: userLoading } = useCurrentUser();
  const isAdmin = user?.role === "admin";
  const showAdminSlots = isAdmin || userLoading;
  const {
    data: health,
    isLoading: healthLoading,
    dataUpdatedAt,
    isError: healthError,
  } = useHealth();
  const {
    data: operations,
    isLoading: operationsLoading,
    isError: operationsError,
  } = useOperationalHealth(isAdmin);
  const { data: users, isLoading: usersLoading } = useUsers(isAdmin);
  const { data: groups, isLoading: groupsLoading } = useGroups(isAdmin);
  const { data: overview, isLoading: overviewLoading } = useDocumentOverview();
  const { data: metrics, isLoading: metricsLoading, isError: metricsError } = useMetrics(isAdmin);
  const { data: rag, isLoading: ragLoading, isError: ragError } = useRollingChatLogStats(hours);

  const sampledAt = metrics?.qdrant?.collected_at ?? 0;
  const points =
    !metricsError && health?.checks?.qdrant?.status === "ok" && Date.now() / 1000 - sampledAt < 90
      ? metrics?.qdrant?.points
      : undefined;
  const chunks = overview?.chunks;
  const mismatch = isAdmin && points != null && chunks != null && points !== chunks;
  const failedCount = overview?.by_status.failed ?? 0;
  const currentOperations = isAdmin && !operationsError ? operations : undefined;
  const operational =
    health?.status === "healthy" &&
    currentOperations?.storage.status !== "error" &&
    !currentOperations?.indexing?.degraded;
  const serviceCards = [
    { title: "API Server", icon: <Server size={20} />, check: health?.checks?.api },
    { title: "PostgreSQL", icon: <Database size={20} />, check: health?.checks?.postgres },
    { title: "Qdrant", icon: <HardDrive size={20} />, check: health?.checks?.qdrant },
    { title: "LLM", icon: <Cpu size={20} />, check: health?.checks?.llm },
    { title: "Redis", icon: <MemoryStick size={20} />, check: health?.checks?.redis },
  ];

  return (
    <div className="mx-auto max-w-screen-2xl space-y-6 p-4 sm:p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">Admin Dashboard</h1>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
          {health && (
            <>
              <span>v{health.version}</span>
              <span className="flex items-center gap-1.5">
                <Clock size={14} /> {formatUptime(health.uptime_seconds)} uptime
              </span>
              <Badge variant={operational ? "success" : "destructive"}>
                {operational ? "Operational" : "Degraded"}
              </Badge>
              {isAdmin && (
                <Link to="/admin/jobs" className="text-primary hover:underline">
                  {health.background_jobs.running ?? 0} running jobs
                </Link>
              )}
            </>
          )}
          {healthError && <span className="text-destructive">Health check unavailable</span>}
          {dataUpdatedAt > 0 && <UpdatedAgo timestamp={dataUpdatedAt} />}
        </div>
      </header>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {serviceCards.map((service) => (
          <ServiceCard key={service.title} {...service} loading={healthLoading} />
        ))}
        {showAdminSlots && (
          <>
            <ServiceCard
              title="S3 Storage"
              icon={<Cloud size={20} />}
              check={currentOperations?.storage}
              loading={userLoading || operationsLoading}
            />
            <IndexingWorkersCard
              indexing={currentOperations?.indexing}
              loading={userLoading || operationsLoading}
            />
          </>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {showAdminSlots && (
          <>
            <StatCard
              label="Users"
              value={users?.total ?? users?.users?.length}
              icon={<Users size={18} />}
              to={isAdmin ? "/admin/users" : undefined}
              loading={userLoading || usersLoading}
            />
            <StatCard
              label="Groups"
              value={groups?.length}
              icon={<FolderOpen size={18} />}
              to={isAdmin ? "/admin/groups" : undefined}
              loading={userLoading || groupsLoading}
            />
          </>
        )}
        <StatCard
          label="Documents"
          value={overview?.total}
          icon={<FileText size={18} />}
          to="/admin/documents"
          loading={overviewLoading}
        >
          {failedCount > 0 && (
            <Link
              to="/admin/documents?status=failed"
              className="block border-t px-4 py-2 text-xs font-medium text-destructive hover:underline"
            >
              {failedCount.toLocaleString()} failed
            </Link>
          )}
        </StatCard>
        <StatCard
          label={isAdmin ? "DB chunks / Qdrant points" : "DB chunks"}
          value={
            isAdmin && (chunks != null || points != null)
              ? `${chunks?.toLocaleString() ?? "—"} / ${points?.toLocaleString() ?? "—"}`
              : chunks
          }
          icon={<HardDrive size={18} />}
          loading={overviewLoading || (isAdmin && metricsLoading)}
        >
          {mismatch && (
            <p
              role="status"
              className="flex items-start gap-1.5 border-t px-4 py-2 text-xs text-amber-700 dark:text-amber-400"
            >
              <AlertCircle className="shrink-0" size={14} />
              Chunk and point counts differ
            </p>
          )}
        </StatCard>
        <StatCard
          label={`Requests · ${periods.find((period) => period.hours === hours)?.label}`}
          value={rag?.stats.total}
          icon={<Activity size={18} />}
          to={rag && chatLogsUrl(rag.dates)}
          loading={ragLoading}
        />
      </div>

      <section aria-labelledby="dashboard-rag-title" className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="dashboard-rag-title" className="text-lg font-semibold">
            RAG
          </h2>
          <div role="group" aria-label="RAG period" className="flex gap-1.5">
            {periods.map((period) => (
              <Button
                key={period.hours}
                size="sm"
                variant={hours === period.hours ? "default" : "outline"}
                aria-pressed={hours === period.hours}
                onClick={() => setHours(period.hours)}
              >
                {period.label}
              </Button>
            ))}
          </div>
        </div>
        {ragError && (
          <p role="status" className="text-sm text-destructive">
            {rag ? "RAG statistics could not refresh" : "RAG statistics unavailable"}
          </p>
        )}
        {rag && <DashboardRag stats={rag.stats} dates={rag.dates} />}
        {ragLoading && <Skeleton className="h-48 w-full" />}
      </section>

      <div className="grid items-start gap-3 lg:grid-cols-2">
        <DocumentStatusCard data={overview} loading={overviewLoading} />
        {rag && <DashboardRagBreakdown stats={rag.stats} dates={rag.dates} />}
      </div>
    </div>
  );
}
