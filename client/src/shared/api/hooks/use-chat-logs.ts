import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { ChatLogsResponse } from "../types";

interface ChatLogsParams {
  user_id?: number;
  domain?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export function useChatLogs(params: ChatLogsParams = {}) {
  return useQuery({
    queryKey: queryKeys.chatLogs.list(params as Record<string, unknown>),
    queryFn: async () => {
      const queryParams: Record<string, string | number> = {};
      if (params.user_id !== undefined) queryParams.user_id = params.user_id;
      if (params.domain) queryParams.domain = params.domain;
      if (params.date_from) queryParams.date_from = params.date_from;
      if (params.date_to) queryParams.date_to = params.date_to;
      if (params.search) queryParams.search = params.search;
      if (params.limit !== undefined) queryParams.limit = params.limit;
      if (params.offset !== undefined) queryParams.offset = params.offset;
      return (await apiClient.get<ChatLogsResponse>("/admin/chat-logs", { params: queryParams }))
        .data;
    },
  });
}
