import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  ChunkCreateRequest,
  ChunkEditRequest,
  ChunkListResponse,
  ChunkResponse,
  DocumentResponse,
  ManualDocumentRequest,
} from "../types";

export function useChunks(documentId: number, limit?: number, offset?: number) {
  return useQuery({
    queryKey: queryKeys.chunks.list(documentId, limit, offset),
    queryFn: async () => {
      const params = new URLSearchParams();
      if (limit) params.set("limit", String(limit));
      if (offset) params.set("offset", String(offset));
      const qs = params.toString();
      return (
        await apiClient.get<ChunkListResponse>(
          `/documents/${documentId}/chunks${qs ? `?${qs}` : ""}`,
        )
      ).data;
    },
    enabled: !!documentId,
  });
}

export function useAddChunk() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ documentId, data }: { documentId: number; data: ChunkCreateRequest }) =>
      (await apiClient.post<ChunkResponse>(`/documents/${documentId}/chunks`, data)).data,
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: queryKeys.chunks.all(variables.documentId) });
      qc.invalidateQueries({ queryKey: queryKeys.documents.all });
    },
  });
}

export function useUpdateChunk() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({
      documentId,
      chunkId,
      data,
    }: {
      documentId: number;
      chunkId: number;
      data: ChunkEditRequest;
    }) =>
      (await apiClient.put<ChunkResponse>(`/documents/${documentId}/chunks/${chunkId}`, data)).data,
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: queryKeys.chunks.all(variables.documentId) });
    },
  });
}

export function useDeleteChunk() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ documentId, chunkId }: { documentId: number; chunkId: number }) =>
      (await apiClient.delete(`/documents/${documentId}/chunks/${chunkId}`)).data,
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: queryKeys.chunks.all(variables.documentId) });
      qc.invalidateQueries({ queryKey: queryKeys.documents.all });
    },
  });
}

export function useCreateManualDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: ManualDocumentRequest) =>
      (await apiClient.post<DocumentResponse>("/documents/manual", data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.documents.all }),
  });
}
