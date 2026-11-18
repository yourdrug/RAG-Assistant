import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createElement } from "react";
import toast from "react-hot-toast";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  BenchmarkDraft,
  ChatLogDetail,
  ChatLogReview,
  ChatLogStats,
  ChatLogsResponse,
  ReviewStatus,
} from "../types";

export interface ChatLogsParams {
  space_id?: number;
  outcome?: string;
  model?: string;
  breadth?: string;
  config_hash?: string;
  rating?: string;
  reason?: string;
  has_comment?: string;
  review_status?: string;
  needs_review?: boolean;
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

/** Query string of the journal filters; the list, stats and export share it. */
export function chatLogQueryParams(params: ChatLogsParams): Record<string, string | number> {
  const queryParams: Record<string, string | number> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "" || value === false) continue;
    queryParams[key] = typeof value === "boolean" ? "true" : value;
  }
  return queryParams;
}

export function useChatLogs(params: ChatLogsParams = {}) {
  return useQuery({
    queryKey: queryKeys.chatLogs.list(params as Record<string, unknown>),
    queryFn: async () =>
      (
        await apiClient.get<ChatLogsResponse>("/admin/chat-logs", {
          params: chatLogQueryParams(params),
        })
      ).data,
  });
}

export function useChatLogStats(params: ChatLogsParams) {
  return useQuery({
    queryKey: [...queryKeys.chatLogs.all, "stats", params],
    queryFn: async () =>
      (
        await apiClient.get<ChatLogStats>("/admin/chat-logs/stats", {
          params: chatLogQueryParams(params),
        })
      ).data,
  });
}

export function useChatLog(id: number, enabled: boolean) {
  return useQuery({
    queryKey: [...queryKeys.chatLogs.all, "detail", id],
    queryFn: async () => (await apiClient.get<ChatLogDetail>(`/admin/chat-logs/${id}`)).data,
    enabled,
  });
}

export function useReviewChatLog(id: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: { status: ReviewStatus; note?: string | null }) =>
      (await apiClient.put<ChatLogReview>(`/admin/chat-logs/${id}/review`, body)).data,
    onSuccess: (review) => {
      toast.success(
        review.status === "benchmark_candidate"
          ? "Saved as a benchmark candidate. An administrator can now create the question."
          : "Review saved.",
      );
      return queryClient.invalidateQueries({ queryKey: queryKeys.chatLogs.all });
    },
  });
}

export function useBenchmarkDraft(id: number, enabled: boolean) {
  return useQuery({
    queryKey: [...queryKeys.chatLogs.all, "benchmark-draft", id],
    queryFn: async () =>
      (await apiClient.get<BenchmarkDraft>(`/admin/chat-logs/${id}/benchmark-question`)).data,
    enabled,
  });
}

export function useTransferToBenchmark(id: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      dataset: string;
      question: string;
      history: BenchmarkDraft["history"];
    }) =>
      (
        await apiClient.post<{ question_id: number; dataset: string }>(
          `/admin/chat-logs/${id}/benchmark-question`,
          body,
        )
      ).data,
    onSuccess: (question) => {
      toast.success(
        createElement(
          "span",
          null,
          `Benchmark question #${question.question_id} created in ${question.dataset} (inactive). `,
          createElement(
            "a",
            {
              href: "/admin/benchmark?tab=questions",
              className: "font-medium underline underline-offset-2",
            },
            "Open benchmark questions",
          ),
        ),
        { duration: 10000 },
      );
      return Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.chatLogs.all }),
        queryClient.invalidateQueries({ queryKey: ["benchmark", "questions"] }),
        queryClient.invalidateQueries({ queryKey: ["benchmark", "datasets"] }),
      ]);
    },
  });
}

export async function downloadChatLogs(params: ChatLogsParams, format: "csv" | "json") {
  const { limit: _limit, offset: _offset, ...filters } = params;
  const response = await apiClient.get<Blob>("/admin/chat-logs/export", {
    params: { ...chatLogQueryParams(filters), format },
    responseType: "blob",
  });
  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = url;
  link.download = `chat-logs.${format}`;
  link.click();
  URL.revokeObjectURL(url);
}
