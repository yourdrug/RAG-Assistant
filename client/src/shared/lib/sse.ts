import type { Source } from "@/shared/api/types";

export interface SSEDone {
  conversation_id: number;
  sources: Source[];
}

interface StreamChatParams {
  question: string;
  conversationId?: number | null;
  token: string;
  onChunk: (text: string) => void;
  onDone: (data: SSEDone) => void;
  onError: (error: string) => void;
  signal?: AbortSignal;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

export async function streamChat({
  question,
  conversationId,
  token,
  onChunk,
  onDone,
  onError,
  signal,
}: StreamChatParams): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ question, conversation_id: conversationId ?? null }),
    signal,
  });

  if (!response.ok) {
    onError(`HTTP ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) { onError("No response body"); return; }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      let currentEvent = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(line.slice(6));
            if (currentEvent === "done") onDone(parsed as SSEDone);
            else if (currentEvent === "error") onError(parsed.error);
            else onChunk(parsed.text || "");
          } catch { /* non-JSON */ }
          currentEvent = "";
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err.message : "Stream error");
  }
}
