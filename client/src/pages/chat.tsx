"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { streamChat } from "@/shared/lib/sse";
import { MessageBubble } from "@/widgets/chat/message-bubble";
import { SourcePanel } from "@/widgets/chat/source-panel";
import { ChatInput } from "@/widgets/chat/chat-input";
import type { Source } from "@/shared/api/types";
import { MessageSquare } from "lucide-react";

interface Message { role: "user" | "assistant"; content: string; sources?: Source[]; }

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingMsg, setStreamingMsg] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const token = useAuthStore((s) => s.token);
  const streamingContentRef = useRef("");

  const scroll = useCallback(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), []);
  useEffect(() => { scroll(); }, [messages, streamingMsg, scroll]);

  const handleSend = async (question: string) => {
    if (!question.trim() || isStreaming || !token) return;
    setMessages((p) => [...p, { role: "user", content: question }]);
    setIsStreaming(true);
    setStreamingMsg("");
    streamingContentRef.current = "";
    abortRef.current = new AbortController();

    await streamChat({
      question, conversationId, token,
      onChunk: (text) => { streamingContentRef.current += text; setStreamingMsg(streamingContentRef.current); },
      onDone: (data) => {
        setMessages((p) => [...p, { role: "assistant", content: streamingContentRef.current, sources: data.sources }]);
        setConversationId(data.conversation_id);
        setStreamingMsg(null); setIsStreaming(false);
      },
      onError: (err) => {
        setMessages((p) => [...p, { role: "assistant", content: `Error: ${err}` }]);
        setStreamingMsg(null); setIsStreaming(false);
      },
      signal: abortRef.current.signal,
    });
  };

  const handleStop = () => {
    abortRef.current?.abort();
    if (streamingContentRef.current) setMessages((p) => [...p, { role: "assistant", content: streamingContentRef.current }]);
    setStreamingMsg(null); setIsStreaming(false);
  };

  const handleNew = () => { setMessages([]); setStreamingMsg(null); setConversationId(null); setSelectedSources([]); };

  return (
    <div className="flex h-full">
      <div className="flex flex-1 flex-col">
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2"><MessageSquare className="h-5 w-5" /><h1 className="font-semibold">Chat</h1></div>
          <button onClick={handleNew} className="text-sm text-muted-foreground hover:text-foreground transition-colors">New Chat</button>
        </div>
        <div className="flex-1 overflow-auto p-4 space-y-4">
          {messages.length === 0 && !streamingMsg && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center space-y-2"><MessageSquare className="h-12 w-12 mx-auto text-muted-foreground/50" /><p className="text-lg font-medium text-muted-foreground">Start a conversation</p><p className="text-sm text-muted-foreground/70">Ask questions about your documents</p></div>
            </div>
          )}
          {messages.map((m, i) => <MessageBubble key={i} role={m.role} content={m.content} sources={m.sources} onSourcesClick={setSelectedSources} />)}
          {streamingMsg !== null && <MessageBubble role="assistant" content={streamingMsg} streaming />}
          <div ref={endRef} />
        </div>
        <ChatInput onSend={handleSend} onStop={handleStop} disabled={isStreaming} />
      </div>
      {selectedSources.length > 0 && <SourcePanel sources={selectedSources} onClose={() => setSelectedSources([])} />}
    </div>
  );
}
