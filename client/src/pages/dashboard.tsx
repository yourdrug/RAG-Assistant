"use client";
import { useHealth, useDocuments } from "@/shared/api/hooks";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Skeleton } from "@/shared/ui/skeleton";
import { Activity, FileText, Server } from "lucide-react";

export function DashboardPage() {
  const { data: health, isLoading: hl } = useHealth();
  const { data: documents, isLoading: dl } = useDocuments();

  const stats = [
    { title: "API Status", value: health?.api || "—", icon: Server, color: health?.api === "ok" ? "success" : "destructive" },
    { title: "Qdrant", value: health?.qdrant || "—", icon: Activity, color: health?.qdrant === "ok" ? "success" : "destructive" },
    { title: "Ollama", value: health?.ollama || "—", icon: Activity, color: health?.ollama === "ok" ? "success" : "destructive" },
    { title: "Documents", value: documents?.length?.toString() || "0", icon: FileText, color: "default" },
  ];

  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">Dashboard</h1><p className="text-muted-foreground">System overview</p></div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((s) => {
          const Icon = s.icon;
          return (
            <Card key={s.title}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{s.title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>{hl || dl ? <Skeleton className="h-7 w-20" /> : <Badge variant={s.color as "success" | "destructive" | "default"}>{s.value}</Badge>}</CardContent>
            </Card>
          );
        })}
      </div>
      {health?.ollama_models && health.ollama_models.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Available Models</CardTitle><CardDescription>Models loaded in Ollama</CardDescription></CardHeader>
          <CardContent><div className="flex flex-wrap gap-2">{health.ollama_models.map((m) => <Badge key={m} variant="secondary">{m}</Badge>)}</div></CardContent>
        </Card>
      )}
      <Card>
        <CardHeader><CardTitle>Recent Documents</CardTitle><CardDescription>Latest uploaded documents</CardDescription></CardHeader>
        <CardContent>
          {dl ? <div className="space-y-2">{[1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full" />)}</div>
            : documents && documents.length > 0 ? <div className="space-y-2">{documents.slice(0, 5).map((d) => (
              <div key={d.id} className="flex items-center justify-between rounded-md border p-3">
                <div className="flex items-center gap-3"><FileText className="h-4 w-4 text-muted-foreground" /><span className="text-sm font-medium">{d.filename}</span></div>
                <Badge variant={d.status === "done" ? "success" : d.status === "failed" ? "destructive" : "secondary"}>{d.status}</Badge>
              </div>
            ))}</div>
            : <p className="text-sm text-muted-foreground">No documents yet</p>}
        </CardContent>
      </Card>
    </div>
  );
}
