import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import type {
  BenchmarkHistoryResponse,
  BenchmarkQuestion,
  BenchmarkQuestionCreate,
  BenchmarkQuestionsListResponse,
  BenchmarkQuestionUpdate,
  BenchmarkRequest,
  BenchmarkResponse,
  BenchmarkResultDetail,
  BenchmarkResultsListResponse,
  BenchmarkRun,
  BenchmarkRunsListResponse,
  RunCompareResponse,
  SweepCreateRequest,
  SweepResponse,
  SweepsListResponse,
} from "../types";

// ─── Legacy benchmark (file-based) ────────────────────────────────────────

export function useBenchmark() {
  return useMutation({
    mutationFn: async (data: BenchmarkRequest) =>
      (await apiClient.post<BenchmarkResponse>("/benchmark", data)).data,
  });
}

export function useBenchmarkResults() {
  return useQuery({
    queryKey: ["benchmark", "results"],
    queryFn: async () =>
      (await apiClient.get<BenchmarkResultsListResponse>("/benchmark/results")).data,
    refetchInterval: 30000,
  });
}

export function useBenchmarkResult(
  runId: number | null,
  options?: { refetchInterval?: number | false },
) {
  return useQuery({
    queryKey: ["benchmark", "result", runId],
    queryFn: async () =>
      (await apiClient.get<BenchmarkResultDetail>(`/benchmark/results/${runId}`)).data,
    enabled: runId != null,
    refetchInterval: options?.refetchInterval ?? 3000,
  });
}

// ─── Benchmark Lab: Questions ─────────────────────────────────────────────

export function useBenchmarkDatasets() {
  return useQuery({
    queryKey: ["benchmark", "datasets"],
    queryFn: async () => (await apiClient.get<string[]>("/admin/benchmark/datasets")).data,
  });
}

export function useBenchmarkQuestions(params?: {
  dataset?: string;
  tag?: string;
  search?: string;
  is_active?: boolean;
  limit?: number;
  offset?: number;
}) {
  const query = new URLSearchParams();
  if (params?.dataset) query.set("dataset", params.dataset);
  if (params?.tag) query.set("tag", params.tag);
  if (params?.search) query.set("search", params.search);
  if (params?.is_active !== undefined) query.set("is_active", String(params.is_active));
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));

  return useQuery({
    queryKey: ["benchmark", "questions", params],
    queryFn: async () =>
      (
        await apiClient.get<BenchmarkQuestionsListResponse>(
          `/admin/benchmark/questions?${query.toString()}`,
        )
      ).data,
  });
}

export function useCreateBenchmarkQuestion() {
  const queryClient = useQueryClient();
  return useMutation({
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["benchmark", "datasets"] }),
    mutationFn: async (data: BenchmarkQuestionCreate) =>
      (await apiClient.post<BenchmarkQuestion>("/admin/benchmark/questions", data)).data,
  });
}

export function useUpdateBenchmarkQuestion() {
  const queryClient = useQueryClient();
  return useMutation({
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["benchmark", "datasets"] }),
    mutationFn: async ({ id, data }: { id: number; data: BenchmarkQuestionUpdate }) =>
      (await apiClient.put<BenchmarkQuestion>(`/admin/benchmark/questions/${id}`, data)).data,
  });
}

export function useDeleteBenchmarkQuestion() {
  const queryClient = useQueryClient();
  return useMutation({
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["benchmark", "datasets"] }),
    mutationFn: async (id: number) =>
      (await apiClient.delete(`/admin/benchmark/questions/${id}`)).data,
  });
}

export function useImportBenchmarkQuestions() {
  const queryClient = useQueryClient();
  return useMutation({
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["benchmark", "datasets"] }),
    mutationFn: async (data: { questions: BenchmarkQuestionCreate[] }) =>
      (await apiClient.post<{ imported: number }>("/admin/benchmark/questions/import", data)).data,
  });
}

export function useExportBenchmarkQuestions(dataset?: string) {
  return useQuery({
    queryKey: ["benchmark", "questions", "export", dataset],
    queryFn: async () => {
      const params = dataset ? `?dataset=${dataset}` : "";
      const response = await apiClient.get<BenchmarkQuestion[]>(
        `/admin/benchmark/questions/export${params}`,
      );
      return response.data;
    },
  });
}

export function useSourceFiles(search?: string) {
  return useQuery({
    queryKey: ["benchmark", "source-files", search],
    queryFn: async () => {
      const params = search ? `?search=${encodeURIComponent(search)}` : "";
      const response = await apiClient.get<{ files: string[] }>(
        `/admin/benchmark/source-files${params}`,
      );
      return response.data.files;
    },
    staleTime: 60000,
  });
}

// ─── Benchmark Lab: Sweeps ────────────────────────────────────────────────

function cacheSweep(client: ReturnType<typeof useQueryClient>, sweep: SweepResponse) {
  client.setQueryData(["benchmark", "sweep", sweep.id], sweep);
  client.setQueryData<SweepsListResponse>(["benchmark", "sweeps"], (current) => {
    if (!current) return { sweeps: [sweep], total: 1 };
    const exists = current.sweeps.some((item) => item.id === sweep.id);
    return {
      ...current,
      total: current.total + (exists ? 0 : 1),
      sweeps: exists
        ? current.sweeps.map((item) => (item.id === sweep.id ? sweep : item))
        : [sweep, ...current.sweeps],
    };
  });
}

export function useCreateSweep() {
  const client = useQueryClient();
  return useMutation({
    onSuccess: async (sweep) => {
      await client.cancelQueries({ queryKey: ["benchmark", "sweeps"] });
      cacheSweep(client, sweep);
      void client.invalidateQueries({ queryKey: ["benchmark", "sweeps"] });
    },
    mutationFn: async (data: SweepCreateRequest) =>
      (await apiClient.post<SweepResponse>("/admin/benchmark/sweep", data)).data,
  });
}

export function useSweep(sweepId: number | null) {
  return useQuery({
    queryKey: ["benchmark", "sweep", sweepId],
    queryFn: async () =>
      (await apiClient.get<SweepResponse>(`/admin/benchmark/sweep/${sweepId}`)).data,
    enabled: !!sweepId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "pending" || status === "running") return 3000;
      return false;
    },
  });
}

export function useSweeps() {
  return useQuery({
    queryKey: ["benchmark", "sweeps"],
    queryFn: async () => (await apiClient.get<SweepsListResponse>("/admin/benchmark/sweeps")).data,
    refetchInterval: (query) =>
      query.state.data?.sweeps.some((s) => s.status === "pending" || s.status === "running")
        ? 3000
        : 10000,
  });
}

export function useResumeSweep() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) =>
      (await apiClient.post<SweepResponse>(`/admin/benchmark/sweep/${id}/resume`)).data,
    onSuccess: async (sweep) => {
      await client.cancelQueries({ queryKey: ["benchmark", "sweeps"] });
      cacheSweep(client, sweep);
      void client.invalidateQueries({ queryKey: ["benchmark"] });
    },
  });
}

export function useCancelSweep() {
  const client = useQueryClient();
  return useMutation({
    onSuccess: async (_result, id) => {
      await Promise.all([
        client.cancelQueries({ queryKey: ["benchmark", "sweeps"] }),
        client.cancelQueries({ queryKey: ["benchmark", "sweep", id] }),
      ]);
      client.setQueryData<SweepResponse>(["benchmark", "sweep", id], (sweep) =>
        sweep ? { ...sweep, status: "cancelled" } : sweep,
      );
      client.setQueryData<SweepsListResponse>(["benchmark", "sweeps"], (current) =>
        current
          ? {
              ...current,
              sweeps: current.sweeps.map((sweep) =>
                sweep.id === id ? { ...sweep, status: "cancelled" } : sweep,
              ),
            }
          : current,
      );
      void client.invalidateQueries({ queryKey: ["benchmark"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
    mutationFn: async (sweepId: number) =>
      (await apiClient.post(`/admin/benchmark/sweep/${sweepId}/cancel`)).data,
  });
}

// ─── Benchmark Lab: Runs ──────────────────────────────────────────────────

export function useBenchmarkRuns(params?: {
  sweep_id?: number;
  dataset?: string;
  sort_by?: string;
  sort_order?: string;
  limit?: number;
  offset?: number;
}) {
  const query = new URLSearchParams();
  if (params?.sweep_id) query.set("sweep_id", String(params.sweep_id));
  if (params?.dataset) query.set("dataset", params.dataset);
  if (params?.sort_by) query.set("sort_by", params.sort_by);
  if (params?.sort_order) query.set("sort_order", params.sort_order);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));

  return useQuery({
    queryKey: ["benchmark", "runs", params],
    queryFn: async () =>
      (await apiClient.get<BenchmarkRunsListResponse>(`/admin/benchmark/runs?${query.toString()}`))
        .data,
    refetchInterval: params?.sweep_id ? 3000 : 10000,
  });
}

export function useBenchmarkRun(runId: number | null) {
  return useQuery({
    queryKey: ["benchmark", "run", runId],
    queryFn: async () => (await apiClient.get<BenchmarkRun>(`/admin/benchmark/runs/${runId}`)).data,
    enabled: !!runId,
  });
}

export function useApplyRunConfig() {
  return useMutation({
    mutationFn: async (runId: number) =>
      (
        await apiClient.post<{
          applied: number;
          keys: string[];
          failed: Array<{ key: string; error: string }>;
        }>(`/admin/benchmark/runs/${runId}/apply`)
      ).data,
  });
}

export function useCompareRuns(ids: number[]) {
  return useQuery({
    queryKey: ["benchmark", "compare", ids],
    queryFn: async () =>
      (
        await apiClient.get<RunCompareResponse>(
          `/admin/benchmark/runs/compare?ids=${ids.join(",")}`,
        )
      ).data,
    enabled: ids.length >= 2,
  });
}

// ─── Benchmark Lab: History ───────────────────────────────────────────────

export function useBenchmarkHistory(params?: { metric?: string; dataset?: string; days?: number }) {
  return useQuery({
    queryKey: ["benchmark", "history", params],
    queryFn: async () => {
      const query = new URLSearchParams();
      if (params?.metric) query.set("metric", params.metric);
      if (params?.dataset) query.set("dataset", params.dataset);
      if (params?.days) query.set("days", String(params.days));
      return (
        await apiClient.get<BenchmarkHistoryResponse>(
          `/admin/benchmark/history?${query.toString()}`,
        )
      ).data;
    },
  });
}

export interface RegressionCheckResult {
  metric: string;
  baseline: number | null;
  current: number | null;
  delta: number | null;
  threshold: number;
  failed: boolean;
  note?: string | null;
}

export interface RegressionCheckResponse {
  passed: boolean;
  results: RegressionCheckResult[];
}

export function useRegressionCheck(runId: number | null) {
  return useQuery({
    queryKey: ["benchmark", "regression-check", runId],
    queryFn: async () =>
      (
        await apiClient.get<RegressionCheckResponse>(
          `/admin/benchmark/regression-check${runId != null ? `?run_id=${runId}` : ""}`,
        )
      ).data,
    enabled: runId != null,
  });
}
