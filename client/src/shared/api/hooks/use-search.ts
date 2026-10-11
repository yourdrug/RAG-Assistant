import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import type { ExactSearchResponse } from "../types";

export type SearchMode = "exact" | "icontains";

export function useExactSearch(
  {
    query,
    mode = "exact",
    limit = 20,
    offset = 0,
    documentId,
  }: {
    query: string;
    mode?: SearchMode;
    limit?: number;
    offset?: number;
    documentId?: number;
  },
  enabled = true,
) {
  return useQuery({
    queryKey: ["search", "exact", { query, mode, limit, offset, documentId }],
    queryFn: async ({ signal }) =>
      (
        await apiClient.post<ExactSearchResponse>(
          "/search/exact",
          { query, mode, limit, offset, document_id: documentId },
          { signal },
        )
      ).data,
    enabled: enabled && query.trim().length >= 3,
  });
}
