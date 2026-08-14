"use client";
import { MessageSquare } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiClient } from "@/shared/api/client";
import type { ConversationHistoryResponse, Source } from "@/shared/api/types";
import { streamChat } from "@/shared/lib/sse";
import { useAuthStore } from "@/stores/auth-store";
import { ChatInput, type DepthOption } from "@/widgets/chat/chat-input";
import { MessageBubble } from "@/widgets/chat/message-bubble";
import { SourcePanel } from "@/widgets/chat/source-panel";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingMsg, setStreamingMsg] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<number | null>(() => {
    const urlId = searchParams.get("id");
    return urlId ? Number(urlId) : null;
  });
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [depth, setDepth] = useState<DepthOption>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const token = useAuthStore((s) => s.token);
  const streamingContentRef = useRef("");
  const hadChunksRef = useRef(false);

  const scroll = useCallback(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), []);
  useEffect(() => {
    scroll();
  }, [messages, streamingMsg, scroll]);

  // Sync conversationId with URL params
  useEffect(() => {
    const urlId = searchParams.get("id");
    const newId = urlId ? Number(urlId) : null;
    if (newId !== conversationId) {
      setConversationId(newId);
    }
  }, [searchParams]);

  // Load conversation history when conversationId changes
  useEffect(() => {
    if (!conversationId || !token) {
      if (!conversationId) {
        setMessages([]);
        setSelectedSources([]);
      }
      return;
    }

    const controller = new AbortController();
    setIsLoadingHistory(true);

    apiClient
      .get<ConversationHistoryResponse>(`/conversations/${conversationId}`, {
        signal: controller.signal,
      })
      .then((res) => {
        const loaded: Message[] = res.data.messages.map((m) => ({
          role: m.role,
          content: m.content,
          sources: m.sources,
        }));
        setMessages(loaded);
      })
      .catch((err) => {
        if (err?.code === "ERR_CANCELED") return;
        setSearchParams({});
        setConversationId(null);
      })
      .finally(() => setIsLoadingHistory(false));

    return () => controller.abort();
  }, [conversationId, token]);

  const handleSend = async (question: string) => {
    if (!question.trim() || isStreaming || !token) return;
    setMessages((p) => [...p, { role: "user", content: question }]);
    setIsStreaming(true);
    setStreamingMsg("");
    streamingContentRef.current = "";
    abortRef.current = new AbortController();
    hadChunksRef.current = false;

    await streamChat({
      question,
      conversationId,
      token,
      depth,
      onChunk: (text) => {
        hadChunksRef.current = true;
        streamingContentRef.current += text;
        setStreamingMsg(streamingContentRef.current);
      },
      onDone: (data) => {
        setMessages((p) => [
          ...p,
          {
            role: "assistant",
            content: streamingContentRef.current,
            sources: data.sources,
          },
        ]);
        setConversationId(data.conversation_id);
        setSearchParams({ id: String(data.conversation_id) });
        setStreamingMsg(null);
        setIsStreaming(false);
      },
      onError: (err) => {
        setMessages((p) => [...p, { role: "assistant", content: `Error: ${err}` }]);
        setStreamingMsg(null);
        setIsStreaming(false);
      },
      signal: abortRef.current.signal,
    });
  };

  const handleStop = () => {
    abortRef.current?.abort();
    if (hadChunksRef.current && streamingContentRef.current) {
      setMessages((p) => [...p, { role: "assistant", content: streamingContentRef.current }]);
    } else if (!hadChunksRef.current) {
      setMessages((p) => p.slice(0, -1));
    }
    setStreamingMsg(null);
    setIsStreaming(false);
  };

  const handleNew = () => {
    abortRef.current?.abort();
    setMessages([]);
    setStreamingMsg(null);
    setIsStreaming(false);
    setConversationId(null);
    setSelectedSources([]);
    setSearchParams({});
  };

  return (
    <div className="flex h-full">
      <div className="flex flex-1 flex-col">
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            <h1 className="font-semibold">Chat</h1>
          </div>
          <button
            onClick={handleNew}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-4">
          {messages.length === 0 && !streamingMsg && !isLoadingHistory && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center space-y-2">
                <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground/50" />
                <p className="text-lg font-medium text-muted-foreground">Start a conversation</p>
                <p className="text-sm text-muted-foreground/70">
                  Ask questions about your documents
                </p>
              </div>
            </div>
          )}
          {isLoadingHistory && (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-muted-foreground">Loading history...</p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              role={m.role}
              content={m.content}
              sources={m.sources}
              onSourcesClick={setSelectedSources}
            />
          ))}
          {streamingMsg !== null && (
            <MessageBubble role="assistant" content={streamingMsg} streaming />
          )}
          <div ref={endRef} />
        </div>
        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          disabled={isStreaming}
          depth={depth}
          onDepthChange={setDepth}
        />
      </div>
      {selectedSources.length > 0 && (
        <SourcePanel sources={selectedSources} onClose={() => setSelectedSources([])} />
      )}
    </div>
  );
}
