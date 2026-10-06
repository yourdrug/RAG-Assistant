import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { apiClient } from "@/shared/api/client";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Skeleton } from "@/shared/ui/skeleton";

interface ModelsInfo {
  llm_provider: string;
  llm_model: string;
  openrouter_model: string;
  ollama_models: string[] | null;
}

interface OpenRouterModel {
  id: string;
  name: string;
  context_length: number;
  pricing: { prompt: number; completion: number };
}

export function BenchmarkJudgeModel({
  value,
  onChange,
  disabled,
}: {
  value: string | null;
  onChange: (model: string) => void;
  disabled: boolean;
}) {
  const { data: info, isLoading } = useQuery({
    queryKey: ["admin", "models"],
    queryFn: async () => (await apiClient.get<ModelsInfo>("/admin/models/info")).data,
  });
  const { data: catalog, isLoading: catalogLoading } = useQuery({
    queryKey: ["admin", "openrouter-models"],
    queryFn: async () =>
      (await apiClient.get<{ models: OpenRouterModel[] }>("/admin/models/openrouter")).data,
    enabled: info?.llm_provider === "openrouter",
  });

  useEffect(() => {
    if (value === null && info) {
      onChange(info.llm_provider === "openrouter" ? info.openrouter_model : info.llm_model);
    }
  }, [value, info, onChange]);

  const models =
    info?.llm_provider === "openrouter"
      ? (catalog?.models ?? []).map((model) => ({
          id: model.id,
          label: `${model.name} (${model.context_length.toLocaleString()} ctx, $${model.pricing.prompt.toFixed(2)}/M)`,
        }))
      : (info?.ollama_models ?? []).map((model) => ({ id: model, label: model }));
  const selected = value ?? "";
  const loading = isLoading || catalogLoading;

  return (
    <div className="space-y-2">
      <Label htmlFor="benchmark-judge-model" className="text-xs">
        Judge model
      </Label>
      {loading ? (
        <Skeleton className="h-9 w-full" />
      ) : models.length > 0 ? (
        <select
          id="benchmark-judge-model"
          value={selected}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50"
        >
          {!selected && <option value="">Select a judge model</option>}
          {selected && !models.some((model) => model.id === selected) && (
            <option value={selected}>{selected}</option>
          )}
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
      ) : (
        <Input
          id="benchmark-judge-model"
          value={selected}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          maxLength={255}
          placeholder="Enter a judge model ID"
        />
      )}
      <p className="text-xs text-muted-foreground">
        {disabled
          ? "Not used when Top-N for LLM is 0."
          : `Uses ${info?.llm_provider ?? "the configured LLM provider"}. This selection applies only to this sweep.`}
      </p>
    </div>
  );
}
