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
  SyncRunStatus,
  SyncStorageKinds,
} from "../types";

const UNFINISHED: SyncRunStatus[] = ["queued", "running"];

/** Administrators see every channel; other users the channels they are responsible for. */
export function useSyncChannels() {
  return useQuery({
    queryKey: queryKeys.syncChannels.list(),
    queryFn: async () => (await apiClient.get<SyncChannel[]>("/sync/channels")).data,
    refetchInterval: 15_000,
  });
}

/** Whether the user is responsible for any channel (the navigation shows the page then). */
export function useHasSyncChannels(enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.syncChannels.list(),
    queryFn: async () => (await apiClient.get<SyncChannel[]>("/sync/channels")).data,
    enabled,
    staleTime: 5 * 60_000,
    select: (channels) => channels.length > 0,
  });
}

/** Storage kinds this deployment can synchronize. */
export function useSyncStorageKinds() {
  return useQuery({
    queryKey: queryKeys.syncChannels.storageKinds(),
    queryFn: async () => (await apiClient.get<SyncStorageKinds>("/sync/storage-kinds")).data.kinds,
    staleTime: 5 * 60_000,
  });
}

/** Runs of an open history: refreshed until every run reached a final status. */
export function useSyncRuns(channelId: number | null) {
  return useQuery({
    queryKey: queryKeys.syncChannels.runs(channelId ?? 0),
    queryFn: async () => (await apiClient.get<SyncRun[]>(`/sync/channels/${channelId}/runs`)).data,
    enabled: channelId != null,
    refetchInterval: (query) =>
      (query.state.data ?? []).some((run) => UNFINISHED.includes(run.status)) ? 5_000 : false,
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
