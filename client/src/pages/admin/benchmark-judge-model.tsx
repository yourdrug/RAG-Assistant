import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { apiClient } from "@/shared/api/client";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { llmModelLabel, useLLMModels } from "./llm-models";

interface ModelsInfo {
  llm_base_url: string;
  llm_model: string;
  llm_auxiliary_model: string;
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
  const [showCatalog, setShowCatalog] = useState(false);
  const { data: info } = useQuery({
    queryKey: ["admin", "models"],
    queryFn: async () => (await apiClient.get<ModelsInfo>("/admin/models/info")).data,
  });
  const catalog = useLLMModels(showCatalog && !disabled);

  useEffect(() => {
    if (value === null && info) {
      onChange(info.llm_auxiliary_model);
    }
  }, [value, info, onChange]);

  const models = (catalog.data?.models ?? []).map((model) => ({
    id: model.id,
    label: llmModelLabel(model),
  }));
  const selected = value ?? "";

  return (
    <div className="space-y-2">
      <Label htmlFor="benchmark-judge-model" className="text-xs">
        Judge model
      </Label>
      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="min-w-40 flex-1"
          id="benchmark-judge-model"
          value={selected}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          maxLength={255}
          placeholder="Enter a judge model ID"
        />
        <Button
          variant="outline"
          type="button"
          disabled={disabled}
          aria-expanded={showCatalog && !disabled}
          onClick={() => setShowCatalog(!showCatalog)}
        >
          {showCatalog ? "Hide model catalog" : "Browse models"}
        </Button>
      </div>
      {showCatalog && !disabled && (
        <>
          {catalog.isPending && (
            <p role="status" className="text-sm text-muted-foreground">
              Loading provider models…
            </p>
          )}
          {catalog.isError && (
            <QueryError query={catalog} fallback="Could not load provider models." />
          )}
          {catalog.data && models.length === 0 && (
            <p className="text-sm text-muted-foreground">The provider returned no models.</p>
          )}
          {models.length > 0 && (
            <select
              aria-label="Provider models"
              value={models.some((model) => model.id === selected) ? selected : ""}
              onChange={(event) => onChange(event.target.value)}
              className="h-10 w-full rounded-md border bg-background px-2.5 text-sm"
            >
              <option value="" disabled>
                Select a provider model
              </option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          )}
        </>
      )}
      <p className="text-xs text-muted-foreground">
        {disabled
          ? "Not used when Top-N for LLM is 0."
          : `Uses ${info?.llm_base_url ?? "the configured LLM endpoint"}. This selection applies only to this sweep.`}
      </p>
    </div>
  );
}
