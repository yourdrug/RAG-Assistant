import type { Source } from "@/shared/api/types";
import { describeHttpError } from "./errors.ts";

export interface SSEDone {
  message_id?: number | null;
  conversation_id: number;
  sources?: Source[] | null;
  confidence?: number | null;
  as_of_date?: string | null;
  /** Authoritative final answer; replaces the streamed fragments (guardrails may rewrite them). */
  answer?: string;
}

export type PipelineStage = "searching" | "reranking" | "generating";

interface StreamChatParams {
  question: string;
  conversationId?: number | null;
  token: string;
  depth?: "narrow" | "broad" | null;
  /** Explicit "as of" date (ISO YYYY-MM-DD) for temporal retrieval. null = current state. */
  asOfDate?: string | null;
  /** Knowledge space of a new conversation; an existing conversation keeps its own. */
  spaceId?: number | null;
  onChunk: (text: string) => void;
  onDone: (data: SSEDone) => void;
  onError: (error: string, status?: number) => void;
  onStatus?: (stage: PipelineStage) => void;
  signal?: AbortSignal;
}

const API_BASE_URL = import.meta.env?.VITE_API_URL || "/api";

export async function streamChat({
  question,
  conversationId,
  token,
  depth,
  asOfDate,
  spaceId,
  onChunk,
  onDone,
  onError,
  onStatus,
  signal,
}: StreamChatParams): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        question,
        conversation_id: conversationId ?? null,
        depth: depth ?? null,
        as_of_date: asOfDate ?? null,
        space_id: spaceId ?? null,
      }),
      signal,
    });
  } catch {
    if (!signal?.aborted) {
      onError("Could not reach the server. Check your connection and try again.");
    }
    return;
  }

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => undefined);
    if (!signal?.aborted) {
      onError(
        describeHttpError(response.status, payload, response.headers.get("Retry-After")),
        response.status,
      );
    }
    return;
  }

  const interrupted = "The answer was interrupted. Please try again.";
  const reader = response.body?.getReader();
  if (!reader) {
    if (!signal?.aborted) onError(interrupted);
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  try {
    while (!signal?.aborted) {
      const { done, value } = await reader.read();
      if (signal?.aborted) return;
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const rawLine of lines) {
        const line = rawLine.replace(/\r$/, "");
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          const event = currentEvent;
          currentEvent = "";
          const parsed = JSON.parse(line.slice(6));
          if (event === "done") {
            if (
              Number.isSafeInteger(parsed.conversation_id) &&
              (parsed.message_id == null || Number.isSafeInteger(parsed.message_id)) &&
              (parsed.sources == null || Array.isArray(parsed.sources)) &&
              (parsed.answer == null || typeof parsed.answer === "string") &&
              (parsed.as_of_date == null || typeof parsed.as_of_date === "string")
            ) {
              onDone(parsed as SSEDone);
            } else {
              onError(interrupted);
            }
            return;
          }
          if (event === "error") {
            const message = typeof parsed.error === "string" && parsed.error.trim();
            switch (parsed.code) {
              case "unauthorized":
                onError(describeHttpError(401, null), 401);
                break;
              case "server_busy":
                onError(describeHttpError(429, { errors: { code: "semaphore_timeout" } }));
                break;
              case "rate_limit_exceeded":
                onError(describeHttpError(429, null, parsed.retry_after));
                break;
              case "llm_unavailable":
                onError(describeHttpError(503, null, parsed.retry_after));
                break;
              case "internal_error":
                onError(describeHttpError(500, null));
                break;
              default:
                onError(message || interrupted);
            }
            return;
          }
          if (event === "status") {
            const { stage } = parsed;
            if (stage === "searching" || stage === "reranking" || stage === "generating") {
              onStatus?.(stage);
            }
          } else if (typeof parsed.text === "string") onChunk(parsed.text);
        }
      }
    }
  } catch {
    // A failed read or malformed event leaves the answer incomplete.
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  if (!signal?.aborted) onError(interrupted);
}
