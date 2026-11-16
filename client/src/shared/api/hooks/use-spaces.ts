import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  CreateKnowledgeSpaceRequest,
  KnowledgeSpace,
  UpdateKnowledgeSpaceRequest,
} from "../types";

export function useSpaces() {
  return useQuery({
    queryKey: queryKeys.spaces.list(),
    queryFn: async () => (await apiClient.get<KnowledgeSpace[]>("/spaces")).data,
  });
}

export function useDomainPacks(enabled = true) {
  return useQuery({
    queryKey: queryKeys.spaces.packs(),
    queryFn: async () =>
      (await apiClient.get<{ pack_keys: string[] }>("/spaces/packs")).data.pack_keys,
    enabled,
    staleTime: Infinity,
  });
}

export function useCreateSpace() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: CreateKnowledgeSpaceRequest) =>
      (await apiClient.post<KnowledgeSpace>("/spaces", data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spaces.all }),
  });
}

export function useUpdateSpace() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, ...data }: UpdateKnowledgeSpaceRequest & { id: number }) =>
      (await apiClient.patch<KnowledgeSpace>(`/spaces/${id}`, data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spaces.all }),
  });
}

export function useDeleteSpace() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) => {
      await apiClient.delete(`/spaces/${id}`);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spaces.all }),
  });
}

/** The explicit choice, or the only space when there is exactly one. */
export function defaultSpaceId(spaces: KnowledgeSpace[] | undefined): number | null {
  return spaces?.length === 1 ? spaces[0].id : null;
}
