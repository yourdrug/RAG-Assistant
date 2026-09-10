import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { ActVersionListResponse, ActVersionUpdateRequest } from "../types";

export function usePendingActVersions() {
  return useQuery({
    queryKey: queryKeys.actVersions.pending(),
    queryFn: async () =>
      (
        await apiClient.get<ActVersionListResponse>("/admin/act-versions", {
          params: { review: "pending" },
        })
      ).data,
  });
}

export function useUpdateActVersion() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ versionId, data }: { versionId: number; data: ActVersionUpdateRequest }) =>
      (await apiClient.patch(`/admin/act-versions/${versionId}`, data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.actVersions.all }),
  });
}
