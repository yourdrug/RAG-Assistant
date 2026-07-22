import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { GroupResponse, CreateGroupRequest, GroupMemberResponse } from "../types";

export function useGroups() {
  return useQuery({
    queryKey: queryKeys.groups.list(),
    queryFn: async () => (await apiClient.get<GroupResponse[]>("/groups")).data,
  });
}

export function useCreateGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: CreateGroupRequest) => (await apiClient.post<GroupResponse>("/groups", data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.groups.all }),
  });
}

export function useGroupMembers(groupId: number) {
  return useQuery({
    queryKey: queryKeys.groups.members(groupId),
    queryFn: async () => (await apiClient.get<GroupMemberResponse[]>(`/groups/${groupId}/members`)).data,
    enabled: !!groupId,
  });
}

export function useAddGroupMember() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ groupId, userId }: { groupId: number; userId: number }) =>
      (await apiClient.post(`/groups/${groupId}/members`, { user_id: userId })).data,
    onSuccess: (_, v) => qc.invalidateQueries({ queryKey: queryKeys.groups.members(v.groupId) }),
  });
}

export function useRemoveGroupMember() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ groupId, userId }: { groupId: number; userId: number }) =>
      (await apiClient.delete(`/groups/${groupId}/members/${userId}`)).data,
    onSuccess: (_, v) => qc.invalidateQueries({ queryKey: queryKeys.groups.members(v.groupId) }),
  });
}
