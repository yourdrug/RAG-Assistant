import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { ChatResponse, NewConversationResponse, ConversationHistoryResponse } from "../types";

export function useSyncChat() {
  return useMutation({
    mutationFn: async (data: { question: string; conversation_id?: number | null }) =>
      (await apiClient.post<ChatResponse>("/chat/sync", data)).data,
  });
}

export function useCreateConversation() {
  return useMutation({
    mutationFn: async () => (await apiClient.post<NewConversationResponse>("/conversations")).data,
  });
}

export function useConversationHistory(id: number) {
  return useQuery({
    queryKey: queryKeys.conversations.history(id),
    queryFn: async () => (await apiClient.get<ConversationHistoryResponse>(`/conversations/${id}`)).data,
    enabled: !!id,
  });
}
