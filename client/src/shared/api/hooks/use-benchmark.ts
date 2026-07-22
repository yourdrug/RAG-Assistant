import { useMutation } from "@tanstack/react-query";
import { apiClient } from "../client";
import type { BenchmarkRequest, BenchmarkResponse } from "../types";

export function useBenchmark() {
  return useMutation({
    mutationFn: async (data: BenchmarkRequest) =>
      (await apiClient.post<BenchmarkResponse>("/benchmark", data)).data,
  });
}
