import { useMutation } from "@tanstack/react-query";
import { apiClient } from "../client";
import type { FeedbackRequest, MessageFeedback } from "../types";

/** Rate your own assistant answer; `null` clears the reaction. */
export function useMessageFeedback(messageId: number) {
  return useMutation({
    mutationFn: async (body: FeedbackRequest | null): Promise<MessageFeedback | null> => {
      if (body === null) {
        await apiClient.delete(`/chat/messages/${messageId}/feedback`);
        return null;
      }
      return (await apiClient.put<MessageFeedback>(`/chat/messages/${messageId}/feedback`, body))
        .data;
    },
  });
}
