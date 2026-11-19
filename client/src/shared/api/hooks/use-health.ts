import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { HealthResponse, OperationalHealthResponse } from "../types";

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health.all,
    queryFn: async () => (await apiClient.get<HealthResponse>("/health")).data,
    refetchInterval: 30000,
  });
}

export function useOperationalHealth(enabled: boolean) {
  return useQuery({
    enabled,
    queryKey: [...queryKeys.health.all, "operations"],
    queryFn: async () =>
      (await apiClient.get<OperationalHealthResponse>("/admin/health", { timeout: 6000 })).data,
    retry: false,
    refetchInterval: 30000,
  });
}
