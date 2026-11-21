import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  CreateSyncChannelRequest,
  SyncAdoptionReport,
  SyncChannel,
  SyncChannelSettings,
  SyncDeletionMode,
  SyncRun,
} from "../types";

/** Administrators see every channel; other users the channels they are responsible for. */
export function useSyncChannels() {
  return useQuery({
    queryKey: queryKeys.syncChannels.list(),
    queryFn: async () => (await apiClient.get<SyncChannel[]>("/sync/channels")).data,
    refetchInterval: 15_000,
  });
}

export function useSyncRuns(channelId: number | null) {
  return useQuery({
    queryKey: queryKeys.syncChannels.runs(channelId ?? 0),
    queryFn: async () => (await apiClient.get<SyncRun[]>(`/sync/channels/${channelId}/runs`)).data,
    enabled: channelId != null,
  });
}

export function useCreateSyncChannel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: CreateSyncChannelRequest) =>
      (await apiClient.post<SyncChannel>("/sync/channels", data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.syncChannels.all }),
  });
}

export function useUpdateSyncChannel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, ...data }: SyncChannelSettings & { id: number }) =>
      (await apiClient.put<SyncChannel>(`/sync/channels/${id}`, data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.syncChannels.all }),
  });
}

export function useDeleteSyncChannel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, mode }: { id: number; mode: SyncDeletionMode }) =>
      (await apiClient.delete(`/sync/channels/${id}`, { params: { mode } })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.syncChannels.all }),
  });
}

/** Move documents ingested from an S3 channel's prefix into the channel, keeping their ids. */
export function useAdoptIngested() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (channelId: number) =>
      (await apiClient.post<SyncAdoptionReport>(`/sync/channels/${channelId}/adopt`)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.syncChannels.all }),
  });
}

export function useStartSyncRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (channelId: number) =>
      (await apiClient.post<SyncRun>(`/sync/channels/${channelId}/runs`)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.syncChannels.all }),
  });
}
