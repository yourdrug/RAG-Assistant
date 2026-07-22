import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { ClientAssignmentResponse } from "../types";

export function useClientAssignments(clientUserId: number) {
  return useQuery({
    queryKey: queryKeys.clients.assignments(clientUserId),
    queryFn: async () => (await apiClient.get<ClientAssignmentResponse[]>(`/clients/${clientUserId}/assignments`)).data,
    enabled: !!clientUserId,
  });
}

export function useAssignClient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ clientUserId, internalUserId }: { clientUserId: number; internalUserId: number }) =>
      (await apiClient.post(`/clients/${clientUserId}/assignments`, { internal_user_id: internalUserId })).data,
    onSuccess: (_, v) => qc.invalidateQueries({ queryKey: queryKeys.clients.assignments(v.clientUserId) }),
  });
}

export function useUnassignClient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ clientUserId, internalUserId }: { clientUserId: number; internalUserId: number }) =>
      (await apiClient.delete(`/clients/${clientUserId}/assignments/${internalUserId}`)).data,
    onSuccess: (_, v) => qc.invalidateQueries({ queryKey: queryKeys.clients.assignments(v.clientUserId) }),
  });
}
