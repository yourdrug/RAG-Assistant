import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import { useAuthStore } from "@/stores/auth-store";
import type { LoginRequest, TokenResponse, UserResponse, CreateUserRequest } from "../types";

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
    queryFn: async () => (await apiClient.get<UserResponse[]>("/auth/users")).data,
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
      (await apiClient.patch(`/auth/users/${userId}`, null, { params: { is_active: isActive } })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.auth.users() }),
  });
}
