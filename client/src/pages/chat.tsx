"use client";
import { useQueryClient } from "@tanstack/react-query";
import { CalendarDays, MessageSquare } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { apiClient, expireSession } from "@/shared/api/client";
import { defaultSpaceId, useConversations, useSpaces } from "@/shared/api/hooks";
import { queryKeys } from "@/shared/api/query-keys";
import type { ConversationHistoryResponse, MessageFeedback, Source } from "@/shared/api/types";
import { formatDate } from "@/shared/lib/date-time";
import { canRetryRequest, getErrorMessage } from "@/shared/lib/errors";
import type { PipelineStage } from "@/shared/lib/sse";
import { streamChat } from "@/shared/lib/sse";
import { readTemporalDate, validIsoDate, writeTemporalDate } from "@/shared/lib/temporal-date";
import { QueryError } from "@/shared/ui/query-error";
import { RequestError } from "@/shared/ui/request-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { useAuthStore } from "@/stores/auth-store";
import { ChatInput, type DepthOption } from "@/widgets/chat/chat-input";
import { MessageBubble } from "@/widgets/chat/message-bubble";
import { SourcePanel } from "@/widgets/chat/source-panel";

interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  feedback?: MessageFeedback | null;
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
  const [historyCursor, setHistoryCursor] = useState<number | null>(null);
  const [historyAttempt, setHistoryAttempt] = useState(0);
  const [draft, setDraft] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<{ message: string; retryable: boolean } | null>(
    null,
  );
  const historyRequestRef = useRef<AbortController | null>(null);
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [depth, setDepth] = useState<DepthOption>(null);
  const [temporalDate, setTemporalDate] = useState(() => readTemporalDate(searchParams));
  const asOfDate = temporalDate.date;
  const calendarDate = temporalDate.source === "calendar" ? asOfDate : null;
  const [pipelineStage, setPipelineStage] = useState<PipelineStage | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const suppressScrollRef = useRef(false);
  const token = useAuthStore((s) => s.token);
  const isClient = useAuthStore((s) => s.user?.kind === "client");
  const streamingContentRef = useRef("");
  const hadChunksRef = useRef(false);
  const queryClient = useQueryClient();
  const spacesQuery = useSpaces();
  const spaces = spacesQuery.data;
  const { data: conversations } = useConversations();
  const [spaceChoice, setSpaceChoice] = useState<number | null>(null);
  // A conversation asks every question in the space of its first question.
  const conversationSpaceId =
    conversationId != null
      ? (conversations?.conversations.find((c) => c.id === conversationId)?.space_id ?? null)
      : null;
  const supportUnassigned = isClient && spaces?.length === 0;
  // An admin revokes a client's space: the old conversation stays readable, the next
  // question starts a new conversation in one of the spaces still assigned.
  const conversationOutdated =
    isClient &&
    conversationSpaceId != null &&
    spaces != null &&
    spaces.length > 0 &&
    !spaces.some((s) => s.id === conversationSpaceId);
  const startsConversation = conversationId == null || conversationOutdated;
  const spaceId =
    (conversationOutdated ? null : conversationSpaceId) ?? spaceChoice ?? defaultSpaceId(spaces);
  const needsSpace = startsConversation && spaceId == null && (spaces?.length ?? 0) > 1;

  const scroll = useCallback(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), []);
  useEffect(() => {
    if (suppressScrollRef.current) {
      suppressScrollRef.current = false;
      return;
    }
    scroll();
  }, [messages, streamingMsg, scroll]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Sync conversationId with URL params
  useEffect(() => {
    const urlId = searchParams.get("id");
    const newId = urlId ? Number(urlId) : null;
    if (newId !== conversationId) {
      abortRef.current?.abort();
      setStreamingMsg(null);
      setIsStreaming(false);
      setPipelineStage(null);
      setSendError(null);
      setMessages([]);
      setSelectedSources([]);
      setConversationId(newId);
    }
  }, [searchParams]);

  useEffect(() => {
    setTemporalDate(readTemporalDate(searchParams));
  }, [searchParams]);

  const handleAsOfDateChange = (value: string | null) => {
    const next = validIsoDate(value);
    const selection = { date: next, source: "calendar" as const };
    setTemporalDate(selection);
    setSearchParams(writeTemporalDate(searchParams, selection));
  };

  // Load conversation history when conversationId changes
  useEffect(() => {
    if (!conversationId || !token) {
      if (!conversationId) {
        setHistoryCursor(null);
        setHistoryError(null);
        setIsLoadingHistory(false);
        setMessages([]);
        setSelectedSources([]);
      }
      return;
    }

    const controller = new AbortController();
    historyRequestRef.current?.abort();
    historyRequestRef.current = controller;
    setHistoryCursor(null);
    setHistoryError(null);
    setIsLoadingHistory(true);

    apiClient
      .get<ConversationHistoryResponse>(`/conversations/${conversationId}`, {
        signal: controller.signal,
      })
      .then((res) => {
        if (controller.signal.aborted) return;
        const loaded: Message[] = res.data.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.sources,
          feedback: m.feedback ?? null,
        }));
        setMessages(loaded);
        setHistoryCursor(res.data.next_cursor);
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          setHistoryError({
            message: getErrorMessage(err, "Could not load the conversation."),
            retryable: canRetryRequest(err),
          });
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoadingHistory(false);
      });

    return () => {
      controller.abort();
      historyRequestRef.current?.abort();
    };
  }, [conversationId, token, historyAttempt]);

  const loadOlderMessages = async () => {
    if (!conversationId || historyCursor === null || isLoadingHistory) return;
    const controller = new AbortController();
    historyRequestRef.current = controller;
    setIsLoadingHistory(true);
    setHistoryError(null);
    try {
      const res = await apiClient.get<ConversationHistoryResponse>(
        `/conversations/${conversationId}`,
        {
          params: { before_id: historyCursor },
          signal: controller.signal,
        },
      );
      if (controller.signal.aborted) return;
      suppressScrollRef.current = true;
      setMessages((current) => [...res.data.messages, ...current]);
      setHistoryCursor(res.data.next_cursor);
    } catch (err) {
      if (!controller.signal.aborted) {
        setHistoryError({
          message: getErrorMessage(err, "Could not load earlier messages."),
          retryable: canRetryRequest(err),
        });
      }
    } finally {
      if (!controller.signal.aborted) setIsLoadingHistory(false);
    }
  };

  const handleSend = async (question: string) => {
    if (!question.trim() || isStreaming || !token || needsSpace) return;
    setDraft("");
    setSendError(null);
    const askedConversationId = conversationOutdated ? null : conversationId;
    if (conversationOutdated) {
      // conversationId is replaced in onDone; resetting it here would clear the messages.
      setHistoryCursor(null);
      setSelectedSources([]);
      setMessages([{ role: "user", content: question }]);
    } else {
      setMessages((p) => [...p, { role: "user", content: question }]);
    }
    setIsStreaming(true);
    setStreamingMsg("");
    streamingContentRef.current = "";
    abortRef.current = new AbortController();
    hadChunksRef.current = false;

    await streamChat({
      question,
      conversationId: askedConversationId,
      token,
      depth,
      asOfDate: calendarDate,
      spaceId: askedConversationId == null ? spaceId : null,
      onChunk: (text) => {
        hadChunksRef.current = true;
        streamingContentRef.current += text;
        setStreamingMsg(streamingContentRef.current);
        setPipelineStage(null);
      },
      onStatus: (stage) => {
        setPipelineStage(stage);
      },
      onDone: (data) => {
        setMessages((p) => [
          ...p,
          {
            role: "assistant",
            id: data.message_id ?? undefined,
            content: data.answer ?? streamingContentRef.current,
            sources: data.sources ?? undefined,
          },
        ]);
        setConversationId(data.conversation_id);
        const resolvedAsOfDate = validIsoDate(data.as_of_date ?? calendarDate);
        const selection = {
          date: resolvedAsOfDate,
          source: calendarDate ? ("calendar" as const) : ("question" as const),
        };
        setTemporalDate(selection);
        setSearchParams(
          writeTemporalDate(new URLSearchParams({ id: String(data.conversation_id) }), selection),
        );
        setStreamingMsg(null);
        setIsStreaming(false);
        setPipelineStage(null);
        queryClient.invalidateQueries({ queryKey: queryKeys.conversations.list() });
      },
      onError: (err, status) => {
        if (status === 401) expireSession(token);
        if (conversationOutdated) {
          setMessages(messages);
          setHistoryCursor(historyCursor);
          setSelectedSources(selectedSources);
        } else {
          setMessages((current) => current.slice(0, -1));
        }
        setDraft(question);
        setSendError(err);
        setStreamingMsg(null);
        setIsStreaming(false);
        setPipelineStage(null);
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
    setPipelineStage(null);
  };

  const handleNew = () => {
    abortRef.current?.abort();
    setMessages([]);
    setStreamingMsg(null);
    setIsStreaming(false);
    setConversationId(null);
    setSelectedSources([]);
    setSendError(null);
    setPipelineStage(null);
    setDraft("");
    setTemporalDate({ date: null, source: "calendar" });
    setSearchParams({});
  };

  return (
    <div className="relative flex h-full min-h-0 min-w-0 overflow-hidden">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3">
          <div className="flex min-w-0 flex-wrap items-center gap-3">
            <MessageSquare className="h-5 w-5" />
            <h1 className="font-semibold">Chat</h1>
            {(spaces?.length ?? 0) > 1 && (
              <Select
                value={spaceId != null ? String(spaceId) : ""}
                onValueChange={(v) => setSpaceChoice(Number(v))}
                disabled={!startsConversation || isStreaming}
              >
                <SelectTrigger
                  className="h-auto min-h-10 w-60 max-w-full gap-3 whitespace-normal text-left [&>span]:line-clamp-none [&>span]:min-w-0 [&>span]:whitespace-normal [&>span]:[overflow-wrap:anywhere] [&>svg]:shrink-0"
                  title="Questions are answered from the documents of this knowledge space"
                >
                  <SelectValue placeholder="Choose a space" />
                </SelectTrigger>
                <SelectContent className="w-[var(--radix-select-trigger-width)]">
                  {spaces?.map((s) => (
                    <SelectItem
                      key={s.id}
                      value={String(s.id)}
                      className="whitespace-normal [overflow-wrap:anywhere]"
                    >
                      {s.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {asOfDate && (
              <span
                className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-600 dark:text-amber-400"
                title="Ответы строятся по редакциям документов, действовавшим на эту дату"
              >
                <CalendarDays className="h-3 w-3" />
                по состоянию на {formatDate(asOfDate)}
                {temporalDate.source === "question" ? " (из вопроса)" : ""}
              </span>
            )}
          </div>
          <button
            onClick={handleNew}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            New Chat
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-4 space-y-4">
          {historyCursor !== null && (
            <button
              type="button"
              disabled={isLoadingHistory || isStreaming}
              onClick={loadOlderMessages}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Загрузить предыдущие сообщения
            </button>
          )}
          {historyError && (
            <RequestError
              message={historyError.message}
              retrying={isLoadingHistory || isStreaming}
              onRetry={
                historyError.retryable
                  ? () => {
                      if (historyCursor !== null) void loadOlderMessages();
                      else setHistoryAttempt((attempt) => attempt + 1);
                    }
                  : undefined
              }
            />
          )}
          {messages.length === 0 && !streamingMsg && !isLoadingHistory && !historyError && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center space-y-2">
                <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground/50" />
                <p className="text-lg font-medium text-muted-foreground">Start a conversation</p>
                <p className="text-sm text-muted-foreground/70">
                  {needsSpace
                    ? "Choose a knowledge space to ask in"
                    : "Ask questions about your documents"}
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
              messageId={m.role === "assistant" ? (m.id ?? undefined) : undefined}
              feedback={m.feedback}
              onFeedbackChange={(feedback) =>
                setMessages((current) =>
                  current.map((item) => (item === m ? { ...item, feedback } : item)),
                )
              }
            />
          ))}
          {streamingMsg !== null && (
            <MessageBubble
              role="assistant"
              content={streamingMsg}
              streaming
              stage={pipelineStage}
            />
          )}

          <div ref={endRef} />
        </div>
        {conversationOutdated && !isStreaming && (
          <p className="px-4 text-sm text-muted-foreground">
            This chat's space is no longer available to you. Your next question will start a new
            chat{needsSpace ? " — choose a space above first" : ""}.
          </p>
        )}
        {supportUnassigned && (
          <p className="px-4 text-sm text-muted-foreground">
            No support space assigned. Contact your administrator.
          </p>
        )}
        {spacesQuery.isError && !spaces && (
          <QueryError query={spacesQuery} fallback="Could not load knowledge spaces." />
        )}
        {sendError && <RequestError message={sendError} />}
        <ChatInput
          value={draft}
          onChange={setDraft}
          onSend={handleSend}
          onStop={handleStop}
          disabled={
            needsSpace ||
            supportUnassigned ||
            !spaces ||
            (historyCursor === null && (isLoadingHistory || Boolean(historyError)))
          }
          isStreaming={isStreaming}
          depth={depth}
          onDepthChange={setDepth}
          asOfDate={calendarDate}
          onAsOfDateChange={handleAsOfDateChange}
        />
      </div>
      {selectedSources.length > 0 && (
        <SourcePanel sources={selectedSources} onClose={() => setSelectedSources([])} />
      )}
    </div>
  );
}
