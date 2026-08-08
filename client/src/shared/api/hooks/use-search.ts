import { useMutation } from "@tanstack/react-query";
import { apiClient } from "../client";
import type { ExactSearchResponse } from "../types";

export type SearchMode = "exact" | "icontains";

export function useExactSearch() {
  return useMutation({
    mutationFn: async ({ query, mode = "exact", limit = 20 }: { query: string; mode?: SearchMode; limit?: number }) =>
      (await apiClient.post<ExactSearchResponse>("/search/exact", { query, mode, limit })).data,
  });
}
