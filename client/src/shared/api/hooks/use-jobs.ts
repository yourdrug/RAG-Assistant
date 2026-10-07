import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { JobResponse, JobsListResponse, JobsStatsResponse } from "../types";

export function useJobs(params?: { limit?: number; offset?: number }) {
  return useQuery({
    queryKey: queryKeys.jobs.list(params),
    queryFn: async () => (await apiClient.get<JobsListResponse>("/admin/jobs", { params })).data,
    refetchInterval: 10000,
  });
}

export function useJobStats() {
  return useQuery({
    queryKey: queryKeys.jobs.stats(),
    queryFn: async () => (await apiClient.get<JobsStatsResponse>("/admin/jobs/stats")).data,
    refetchInterval: 10000,
  });
}

export function useJob(jobId: number | null) {
  return useQuery({
    queryKey: ["jobs", "detail", jobId],
    queryFn: async () => (await apiClient.get<JobResponse>(`/admin/jobs/${jobId}`)).data,
    enabled: jobId !== null,
    refetchInterval: (query) =>
      query.state.data?.status === "failed" || query.state.data?.status === "done" ? false : 3000,
  });
}
