"use client";
import { useHealth, useUsers, useGroups, useDocuments, useIngestRegistry } from "@/shared/api/hooks";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Skeleton } from "@/shared/ui/skeleton";
import { Users, FileText, FolderOpen, Activity, Database } from "lucide-react";

export function AdminDashboardPage() {
  const { data: health, isLoading: hl } = useHealth();
  const { data: users, isLoading: ul } = useUsers();
  const { data: groups, isLoading: gl } = useGroups();
  const { data: documents, isLoading: dl } = useDocuments();
  const { data: registry, isLoading: rl } = useIngestRegistry();

  const stats = [
    { title: "Users", value: users?.length ?? 0, icon: Users, loading: ul },
    { title: "Groups", value: groups?.length ?? 0, icon: FolderOpen, loading: gl },
    { title: "Documents", value: documents?.length ?? 0, icon: FileText, loading: dl },
    { title: "Indexed Files", value: registry?.total_files ?? 0, icon: Database, loading: rl },
  ];

  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">Admin Dashboard</h1><p className="text-muted-foreground">System overview</p></div>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" />System Health</CardTitle></CardHeader>
        <CardContent>
          {hl ? <div className="flex gap-4">{[1, 2, 3].map((i) => <Skeleton key={i} className="h-8 w-24" />)}</div> : (
            <div className="flex flex-wrap gap-4">
              <div className="flex items-center gap-2"><span className="text-sm text-muted-foreground">API:</span><Badge variant={health?.api === "ok" ? "success" : "destructive"}>{health?.api || "unknown"}</Badge></div>
              <div className="flex items-center gap-2"><span className="text-sm text-muted-foreground">Qdrant:</span><Badge variant={health?.qdrant === "ok" ? "success" : "destructive"}>{health?.qdrant || "unknown"}</Badge></div>
              <div className="flex items-center gap-2"><span className="text-sm text-muted-foreground">Ollama:</span><Badge variant={health?.ollama === "ok" ? "success" : "destructive"}>{health?.ollama || "unknown"}</Badge></div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((s) => {
          const Icon = s.icon;
          return (
            <Card key={s.title}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{s.title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>{s.loading ? <Skeleton className="h-7 w-16" /> : <div className="text-2xl font-bold">{s.value}</div>}</CardContent>
            </Card>
          );
        })}
      </div>

      {registry && (
        <Card>
          <CardHeader><CardTitle>Index Statistics</CardTitle><CardDescription>Total indexed content</CardDescription></CardHeader>
          <CardContent>
            <div className="flex gap-8">
              <div><p className="text-sm text-muted-foreground">Total Files</p><p className="text-2xl font-bold">{registry.total_files}</p></div>
              <div><p className="text-sm text-muted-foreground">Total Chunks</p><p className="text-2xl font-bold">{registry.total_chunks.toLocaleString()}</p></div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
