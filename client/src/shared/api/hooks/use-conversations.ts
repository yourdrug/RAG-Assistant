import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  ChatResponse,
  ConversationHistoryResponse,
  ConversationListResponse,
  NewConversationResponse,
} from "../types";

export function useSyncChat() {
  return useMutation({
    mutationFn: async (data: {
      question: string;
      conversation_id?: number | null;
      /** Explicit "as of" date (ISO YYYY-MM-DD) for temporal retrieval. */
      as_of_date?: string | null;
    }) => (await apiClient.post<ChatResponse>("/chat/sync", data)).data,
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
    queryFn: async () =>
      (await apiClient.get<ConversationHistoryResponse>(`/conversations/${id}`)).data,
    enabled: !!id,
  });
}

export function useConversations() {
  return useQuery({
    queryKey: queryKeys.conversations.list(),
    queryFn: async () => (await apiClient.get<ConversationListResponse>("/conversations")).data,
  });
}
