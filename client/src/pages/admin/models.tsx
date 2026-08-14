"use client";
import { useQuery } from "@tanstack/react-query";
import { Brain, Cloud, Cpu, FileText, ScanText } from "lucide-react";
import { apiClient } from "@/shared/api/client";
import { Badge } from "@/shared/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";

interface ModelsInfo {
  llm_provider: string;
  llm_model: string;
  embed_model: string;
  rerank_model: string;
  device: string;
  ocr_engine: string;
  ocr_enabled: boolean;
  ollama_models: string[] | null;
  openrouter_model: string | null;
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
      title: "LLM Provider",
      value: data?.llm_provider === "openrouter" ? "OpenRouter (Cloud)" : "Ollama (Local)",
      icon: data?.llm_provider === "openrouter" ? Cloud : Brain,
      description: "Active LLM provider",
    },
    {
      title: "LLM Model",
      value: data?.llm_provider === "openrouter" ? data?.openrouter_model : data?.llm_model,
      icon: Brain,
      description: data?.llm_provider === "openrouter" ? "Cloud model via OpenRouter" : "Local model via Ollama",
    },
    {
      title: "Embedding Model",
      value: data?.embed_model,
      icon: FileText,
      description: "Vector embeddings",
    },
    {
      title: "Reranker Model",
      value: data?.rerank_model,
      icon: Cpu,
      description: "Cross-encoder for reranking",
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
                <div className="text-lg font-semibold">{m.value}</div>
                <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {data?.llm_provider === "ollama" && data?.ollama_models && data.ollama_models.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Available Ollama Models</CardTitle>
            <CardDescription>Models loaded in Ollama</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {data.ollama_models.map((model) => (
                <Badge key={model} variant={model === data?.llm_model ? "default" : "secondary"}>
                  {model}
                  {model === data?.llm_model && " (active)"}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {data?.llm_provider === "openrouter" && (
        <Card>
          <CardHeader>
            <CardTitle>OpenRouter Configuration</CardTitle>
            <CardDescription>Cloud LLM API settings</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between border-b pb-2">
                <span className="text-sm text-muted-foreground">Active Model</span>
                <span className="text-sm font-medium">{data?.openrouter_model}</span>
              </div>
              <div className="flex justify-between border-b pb-2">
                <span className="text-sm text-muted-foreground">API Endpoint</span>
                <span className="text-sm font-medium">https://openrouter.ai/api/v1</span>
              </div>
              <div className="flex justify-between border-b pb-2">
                <span className="text-sm text-muted-foreground">Status</span>
                <Badge variant="success">Connected</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Model Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between border-b pb-2">
              <span className="text-sm text-muted-foreground">Device</span>
              <span className="text-sm font-medium">{data?.device}</span>
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
