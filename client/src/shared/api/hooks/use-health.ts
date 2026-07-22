import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { HealthResponse } from "../types";

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health.all,
    queryFn: async () => (await apiClient.get<HealthResponse>("/health")).data,
    refetchInterval: 30000,
  });
}
