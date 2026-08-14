import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { MetricsResponse } from "../types";

export function useMetrics() {
  return useQuery({
    queryKey: queryKeys.metrics.all,
    queryFn: async () => (await apiClient.get<MetricsResponse>("/admin/metrics")).data,
    refetchInterval: 15000,
  });
}
