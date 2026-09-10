import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/stores/auth-store";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  CreateUserRequest,
  CuratorScopeResponse,
  LoginRequest,
  TokenResponse,
  UserListResponse,
  UserResponse,
} from "../types";

export function useLogin() {
  const setAuth = useAuthStore((s) => s.setAuth);
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: LoginRequest) => {
      const { data: token } = await apiClient.post<TokenResponse>("/auth/login", data);
      const { data: user } = await apiClient.get<UserResponse>("/auth/me", {
        headers: { Authorization: `Bearer ${token.access_token}` },
      });
      setAuth(token.access_token, user);
      qc.setQueryData(queryKeys.auth.me(), user);
      return { token, user };
    },
  });
}

export function useCurrentUser() {
  const token = useAuthStore((s) => s.token);
  return useQuery({
    queryKey: queryKeys.auth.me(),
    queryFn: async () => (await apiClient.get<UserResponse>("/auth/me")).data,
    enabled: !!token,
    retry: false,
  });
}

export function useUsers() {
  return useQuery({
    queryKey: queryKeys.auth.users(),
    queryFn: async () => (await apiClient.get<UserListResponse>("/auth/users")).data,
  });
}

export function useCreateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (data: CreateUserRequest) =>
      (await apiClient.post<UserResponse>("/auth/users", data)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.auth.users() }),
  });
}

export function useToggleUserActive() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ userId, isActive }: { userId: number; isActive: boolean }) =>
      (await apiClient.patch(`/auth/users/${userId}`, null, { params: { is_active: isActive } }))
        .data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.auth.users() }),
  });
}

export function useChangeUserRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ userId, role }: { userId: number; role: string }) =>
      (await apiClient.patch<UserResponse>(`/auth/users/${userId}/role`, { role })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.auth.users() }),
  });
}

export function useCuratorScope(curatorId: number | null) {
  return useQuery({
    queryKey: ["curator-scope", curatorId],
    queryFn: async () =>
      (await apiClient.get<CuratorScopeResponse>(`/admin/curators/${curatorId}/scope`)).data,
    enabled: curatorId !== null,
  });
}

export function useAssignUserToCurator() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ curatorId, targetUserId }: { curatorId: number; targetUserId: number }) =>
      (await apiClient.post(`/admin/curators/${curatorId}/users/${targetUserId}`)).data,
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({ queryKey: ["curator-scope", vars.curatorId] });
    },
  });
}

export function useUnassignUserFromCurator() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ curatorId, targetUserId }: { curatorId: number; targetUserId: number }) =>
      (await apiClient.delete(`/admin/curators/${curatorId}/users/${targetUserId}`)).data,
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({ queryKey: ["curator-scope", vars.curatorId] });
    },
  });
}

export function useAssignGroupToCurator() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ curatorId, groupId }: { curatorId: number; groupId: number }) =>
      (await apiClient.post(`/admin/curators/${curatorId}/groups/${groupId}`)).data,
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({ queryKey: ["curator-scope", vars.curatorId] });
    },
  });
}

export function useUnassignGroupFromCurator() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ curatorId, groupId }: { curatorId: number; groupId: number }) =>
      (await apiClient.delete(`/admin/curators/${curatorId}/groups/${groupId}`)).data,
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({ queryKey: ["curator-scope", vars.curatorId] });
    },
  });
}
