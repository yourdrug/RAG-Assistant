import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import type {
  BenchmarkRequest,
  BenchmarkResponse,
  BenchmarkResultDetail,
  BenchmarkResultsListResponse,
} from "../types";

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

export function useBenchmarkResult(filename: string | null) {
  return useQuery({
    queryKey: ["benchmark", "result", filename],
    queryFn: async () =>
      (await apiClient.get<BenchmarkResultDetail>(`/benchmark/results/${filename}`)).data,
    enabled: !!filename,
  });
}
