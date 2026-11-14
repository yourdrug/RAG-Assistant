import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { JobResponse, JobsListResponse, JobsStatsResponse } from "../types";

export function useJobs(params?: { limit?: number; offset?: number; status?: string }) {
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
      ["failed", "done", "cancelled"].includes(query.state.data?.status ?? "") ? false : 3000,
  });
}

export function useJobAction(action: "retry" | "cancel") {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: number) => {
      const path = action === "retry" ? `/admin/jobs/${jobId}/retry` : `/jobs/${jobId}/cancel`;
      await apiClient.post(path);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.jobs.all }),
    onError: () => toast.error("Could not update the task. Refresh and try again."),
  });
}
