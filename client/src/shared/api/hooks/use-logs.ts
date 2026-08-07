import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { LogsResponse } from "../types";

export function useLogs(params?: {
  limit?: number;
  level?: string;
  search?: string;
}) {
  return useQuery({
    queryKey: queryKeys.logs.list(params),
    queryFn: async () =>
      (await apiClient.get<LogsResponse>("/admin/logs", { params })).data,
    refetchInterval: 5000,
  });
}
