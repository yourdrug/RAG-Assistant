import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { DocumentResponse, UploadStatusResponse } from "../types";

export function useDocuments() {
  return useQuery({
    queryKey: queryKeys.documents.list(),
    queryFn: async () => (await apiClient.get<DocumentResponse[]>("/documents")).data,
  });
}

export function useDocument(id: number) {
  return useQuery({
    queryKey: queryKeys.documents.detail(id),
    queryFn: async () => (await apiClient.get<DocumentResponse>(`/documents/${id}`)).data,
    enabled: !!id,
  });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ file, visibility, groupId, renameOnConflict }: { file: File; visibility: string; groupId?: number | null; renameOnConflict?: boolean }) => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("visibility", visibility);
      if (groupId != null) fd.append("group_id", String(groupId));
      if (renameOnConflict) fd.append("rename_on_conflict", "true");
      return (await apiClient.post<UploadStatusResponse>("/documents", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      })).data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.documents.all }),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) => (await apiClient.delete(`/documents/${id}`)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.documents.all }),
  });
}
