import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { LogsResponse } from "../types";

export const LOGS_REFRESH_INTERVAL_MS = 5000;

export function useLogs(
  params?: { limit?: number; level?: string; search?: string },
  options?: { autoRefresh?: boolean },
) {
  return useQuery({
    queryKey: queryKeys.logs.list(params),
    queryFn: async () => (await apiClient.get<LogsResponse>("/admin/logs", { params })).data,
    refetchInterval: options?.autoRefresh === false ? false : LOGS_REFRESH_INTERVAL_MS,
    placeholderData: keepPreviousData,
  });
}
