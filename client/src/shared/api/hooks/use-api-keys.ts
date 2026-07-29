import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { ApiKeyResponse, ApiKeyCreateRequest, ApiKeyCreatedResponse } from "../types";

export function useClientApiKeys(clientUserId: number) {
  return useQuery({
    queryKey: queryKeys.apiKeys.clientKeys(clientUserId),
    queryFn: async () => (await apiClient.get<ApiKeyResponse[]>(`/clients/${clientUserId}/api-keys`)).data,
    enabled: !!clientUserId,
  });
}

export function useCreateClientApiKey(clientUserId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: ApiKeyCreateRequest) =>
      (await apiClient.post<ApiKeyCreatedResponse>(`/clients/${clientUserId}/api-keys`, data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.apiKeys.clientKeys(clientUserId) }),
  });
}

export function useRevokeClientApiKey(clientUserId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ apiKeyId }: { apiKeyId: number }) =>
      (await apiClient.delete(`/clients/${clientUserId}/api-keys/${apiKeyId}`)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.apiKeys.clientKeys(clientUserId) }),
  });
}
