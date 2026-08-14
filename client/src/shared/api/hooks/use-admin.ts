import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";

// ─── Config params ─────────────────────────────────────────────────────────

export interface ConfigParam {
  key: string;
  value: string;
  value_type: string;
  category: string;
  description?: string | null;
  min_value?: number | null;
  max_value?: number | null;
  allowed_values?: Record<string, unknown> | null;
}

export function useAdminConfig() {
  return useQuery({
    queryKey: queryKeys.admin.config(),
    queryFn: async () => (await apiClient.get<ConfigParam[]>("/admin/config")).data,
  });
}

export function useUpdateAdminConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ key, value }: { key: string; value: string }) =>
      (await apiClient.put(`/admin/config/${key}`, { value })).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.admin.config() }),
  });
}

// ─── Models info ───────────────────────────────────────────────────────────

export interface ModelInfo {
  model: string;
  parameter_size: string;
  quantization: string;
  family: string;
}

export interface ModelsInfo {
  ollama_models: ModelInfo[];
  embedding_model: string;
  rerank_model: string;
}

export function useAdminModels() {
  return useQuery({
    queryKey: queryKeys.admin.models(),
    queryFn: async () => (await apiClient.get<ModelsInfo>("/admin/models/info")).data,
  });
}

// ─── Vector DB info ────────────────────────────────────────────────────────

export interface VectorDBInfo {
  collection_name: string;
  vector_size: number;
  points_count: number;
  status: string;
}

export function useAdminVectorDB() {
  return useQuery({
    queryKey: queryKeys.admin.vectordb(),
    queryFn: async () => (await apiClient.get<VectorDBInfo>("/admin/vectordb/info")).data,
    refetchInterval: 15000,
  });
}
