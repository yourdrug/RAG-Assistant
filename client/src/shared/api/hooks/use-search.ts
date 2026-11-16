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
    spaceId,
  }: {
    query: string;
    mode?: SearchMode;
    limit?: number;
    offset?: number;
    documentId?: number;
    /** Knowledge space to search; the document's or the only space when omitted. */
    spaceId?: number | null;
  },
  enabled = true,
) {
  return useQuery({
    queryKey: ["search", "exact", { query, mode, limit, offset, documentId, spaceId }],
    queryFn: async ({ signal }) =>
      (
        await apiClient.post<ExactSearchResponse>(
          "/search/exact",
          { query, mode, limit, offset, document_id: documentId, space_id: spaceId ?? undefined },
          { signal },
        )
      ).data,
    enabled: enabled && query.trim().length >= 3,
  });
}
