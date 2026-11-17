import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { ChatLogDetail, ChatLogsResponse } from "../types";

interface ChatLogsParams {
  space_id?: number;
  outcome?: string;
  model?: string;
  breadth?: string;
  config_hash?: string;
  sort_by?: "date" | "latency" | "confidence";
  sort_order?: "asc" | "desc";
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
      for (const key of [
        "space_id",
        "outcome",
        "model",
        "breadth",
        "config_hash",
        "sort_by",
        "sort_order",
      ] as const) {
        const value = params[key];
        if (value !== undefined && value !== "") queryParams[key] = value;
      }
      if (params.limit !== undefined) queryParams.limit = params.limit;
      if (params.offset !== undefined) queryParams.offset = params.offset;
      return (await apiClient.get<ChatLogsResponse>("/admin/chat-logs", { params: queryParams }))
        .data;
    },
  });
}

export function useChatLog(id: number, enabled: boolean) {
  return useQuery({
    queryKey: ["chatLogs", "detail", id],
    queryFn: async () => (await apiClient.get<ChatLogDetail>(`/admin/chat-logs/${id}`)).data,
    enabled,
  });
}
