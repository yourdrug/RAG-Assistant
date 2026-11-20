import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/shared/api/client";

/** One entry of the configured endpoint's OpenAI-compatible `GET /models`. */
export interface LLMModel {
  id: string;
  name: string;
  context_length: number | null;
  /** USD per 1M tokens; only endpoints that report pricing (e.g. OpenRouter) fill it. */
  pricing: { prompt: number; completion: number } | null;
}

export const LLM_MODELS_QUERY_KEY = ["admin", "llm-models"] as const;

export function useLLMModels(enabled = true) {
  return useQuery({
    queryKey: LLM_MODELS_QUERY_KEY,
    queryFn: async ({ signal }) =>
      (await apiClient.get<{ models: LLMModel[] }>("/admin/models/llm", { signal })).data,
    enabled,
    staleTime: 60_000,
  });
}

export function llmModelLabel(model: LLMModel): string {
  const details = [
    model.context_length ? `${model.context_length.toLocaleString()} ctx` : null,
    model.pricing ? `$${model.pricing.prompt.toFixed(2)}/M` : null,
  ].filter(Boolean);
  return details.length > 0 ? `${model.name} (${details.join(", ")})` : model.name;
}
