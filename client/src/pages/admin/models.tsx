"use client";
import { useQuery } from "@tanstack/react-query";
import { Brain, Cloud, Cpu, FileText, ScanText } from "lucide-react";
import { apiClient } from "@/shared/api/client";
import { Badge } from "@/shared/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";

interface ModelsInfo {
  llm_base_url: string;
  llm_model: string;
  llm_auxiliary_model: string;
  llm_api_key_configured: boolean;
  embedding_base_url: string;
  embed_model: string;
  embedding_api_key_configured: boolean;
  rerank_base_url: string;
  rerank_model: string;
  rerank_api_format: string;
  rerank_api_key_configured: boolean;
  ocr_engine: string;
  ocr_enabled: boolean;
}

export function AdminModelsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["admin", "models"],
    queryFn: async () => (await apiClient.get<ModelsInfo>("/admin/models/info")).data,
  });

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div>
          <Skeleton className="h-8 w-48 mb-2" />
          <Skeleton className="h-4 w-72" />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-32" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-6 w-48" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const models = [
    {
      title: "LLM Endpoint",
      value: data?.llm_base_url,
      icon: Cloud,
      description: "OpenAI-compatible API",
    },
    {
      title: "LLM Model",
      value: data?.llm_model,
      icon: Brain,
      description: `Auxiliary: ${data?.llm_auxiliary_model ?? "—"}`,
    },
    {
      title: "Embedding Model",
      value: data?.embed_model,
      icon: FileText,
      description: data?.embedding_base_url ?? "Vector embeddings",
    },
    {
      title: "Reranker Model",
      value: data?.rerank_model,
      icon: Cpu,
      description: `${data?.rerank_api_format ?? "—"} · ${data?.rerank_base_url ?? ""}`,
    },
    {
      title: "OCR Engine",
      value: data?.ocr_engine,
      icon: ScanText,
      description: data?.ocr_enabled ? "Enabled" : "Disabled",
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Models</h1>
        <p className="text-muted-foreground">Current ML model configuration (read-only)</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {models.map((m) => {
          const Icon = m.icon;
          return (
            <Card key={m.title}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{m.title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="break-all text-lg font-semibold">{m.value}</div>
                <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>LLM Endpoint</CardTitle>
          <CardDescription>
            Change the URL and models in Settings → Model connections
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between gap-4 border-b pb-2">
              <span className="text-sm text-muted-foreground">API Endpoint</span>
              <span className="break-all text-right text-sm font-medium">{data?.llm_base_url}</span>
            </div>
            <div className="flex justify-between gap-4 border-b pb-2">
              <span className="text-sm text-muted-foreground">Answer Model</span>
              <span className="break-all text-right text-sm font-medium">{data?.llm_model}</span>
            </div>
            <div className="flex justify-between gap-4 border-b pb-2">
              <span className="text-sm text-muted-foreground">Auxiliary Model</span>
              <span className="break-all text-right text-sm font-medium">
                {data?.llm_auxiliary_model}
              </span>
            </div>
            <div className="flex justify-between gap-4 border-b pb-2">
              <span className="text-sm text-muted-foreground">API Key (LLM_API_KEY)</span>
              <Badge variant={data?.llm_api_key_configured ? "success" : "secondary"}>
                {data?.llm_api_key_configured ? "••••••• set" : "Not set"}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between gap-4 border-b pb-2">
              <span className="text-sm text-muted-foreground">
                Embedding Key (EMBEDDING_API_KEY)
              </span>
              <Badge variant={data?.embedding_api_key_configured ? "success" : "secondary"}>
                {data?.embedding_api_key_configured ? "••••••• set" : "Not set"}
              </Badge>
            </div>
            <div className="flex justify-between gap-4 border-b pb-2">
              <span className="text-sm text-muted-foreground">Rerank Key (RERANK_API_KEY)</span>
              <Badge variant={data?.rerank_api_key_configured ? "success" : "secondary"}>
                {data?.rerank_api_key_configured ? "••••••• set" : "Not set"}
              </Badge>
            </div>
            <div className="flex justify-between border-b pb-2">
              <span className="text-sm text-muted-foreground">OCR Enabled</span>
              <Badge variant={data?.ocr_enabled ? "success" : "destructive"}>
                {data?.ocr_enabled ? "Yes" : "No"}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
