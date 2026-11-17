"use client";
import { Check, Copy, ExternalLink } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";
import remarkGfm from "remark-gfm";
import type { MessageFeedback, Source } from "@/shared/api/types";
import { remarkDocumentCitations } from "@/shared/lib/document-citations";
import type { PipelineStage } from "@/shared/lib/sse";
import { cn } from "@/shared/lib/utils";
import { Avatar, AvatarFallback } from "@/shared/ui/avatar";
import { Button } from "@/shared/ui/button";
import { MessageFeedbackControls } from "./message-feedback";

interface Props {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
  stage?: PipelineStage | null;
  onSourcesClick?: (s: Source[]) => void;
  /** Persisted assistant message: enables rating. */
  messageId?: number;
  feedback?: MessageFeedback | null;
  onFeedbackChange?: (feedback: MessageFeedback | null) => void;
}

const STAGE_LABELS: Record<PipelineStage, string> = {
  searching: "Searching documents",
  reranking: "Reranking results",
  generating: "Generating answer",
};

function ThinkingDots({ stage }: { stage?: PipelineStage | null }) {
  return (
    <div
      className="flex items-center justify-center gap-2.5 py-0.5"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-1">
        <span className="h-1.5 w-1.5 animate-dot-pulse rounded-full bg-foreground/50 [animation-delay:0ms]" />
        <span className="h-1.5 w-1.5 animate-dot-pulse rounded-full bg-foreground/50 [animation-delay:150ms]" />
        <span className="h-1.5 w-1.5 animate-dot-pulse rounded-full bg-foreground/50 [animation-delay:300ms]" />
      </div>
      <span className="animate-shimmer bg-size-[200%_100%] bg-linear-to-r from-foreground/40 via-foreground to-foreground/40 bg-clip-text text-xs leading-none">
        {stage ? `${STAGE_LABELS[stage]}...` : "Thinking..."}
      </span>
    </div>
  );
}

export function MessageBubble({
  role,
  content,
  sources,
  streaming,
  stage,
  onSourcesClick,
  messageId,
  feedback,
  onFeedbackChange,
}: Props) {
  const [copied, setCopied] = useState(false);
  const isUser = role === "user";
  const isPending = streaming && content.length === 0;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={cn("group flex gap-3", isUser && "flex-row-reverse")}>
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback
          className={cn("text-xs", isUser ? "bg-primary text-primary-foreground" : "bg-muted")}
        >
          {isUser ? "U" : "AI"}
        </AvatarFallback>
      </Avatar>
      <div className={cn("max-w-[80%] space-y-2", isUser && "text-right")}>
        <div
          className={cn(
            "rounded-lg px-4 py-3 text-sm",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted",
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{content}</p>
          ) : isPending ? (
            <ThinkingDots stage={stage} />
          ) : (
            <div
              className={cn(
                "prose prose-sm dark:prose-invert max-w-none",
                streaming &&
                  content.length > 0 &&
                  "[&>*:last-child]:after:ml-1.5 [&>*:last-child]:after:inline-block [&>*:last-child]:after:h-4 [&>*:last-child]:after:w-2 [&>*:last-child]:after:translate-y-0.5 [&>*:last-child]:after:animate-pulse [&>*:last-child]:after:bg-foreground/70 [&>*:last-child]:after:content-['']",
              )}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkDocumentCitations(sources ?? [])]}
                components={{
                  a: ({ href, children, title }) =>
                    href?.match(/^\/documents\/\d+$/) ? (
                      <Link
                        to={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        title={title ? `Открыть документ: ${title}` : "Открыть документ"}
                        aria-label={title ? `Открыть документ: ${title}` : "Открыть документ"}
                        className="rounded px-0.5 text-primary font-medium no-underline hover:underline focus-visible:outline-2"
                      >
                        {children}
                      </Link>
                    ) : (
                      <a href={href} title={title} target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    ),
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {sources && sources.length > 0 && (
          <button
            onClick={() => onSourcesClick?.(sources)}
            className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-1 text-xs text-muted-foreground hover:bg-muted transition-colors"
          >
            <ExternalLink className="h-3 w-3" />
            Источники · {sources.length}
          </button>
        )}
        {!isUser && !streaming && content && (
          <div
            className={cn(
              "flex items-center gap-1 transition-opacity",
              feedback
                ? "opacity-100"
                : "opacity-0 group-hover:opacity-100 focus-within:opacity-100",
            )}
          >
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={handleCopy}
              aria-label={copied ? "Copied" : "Copy message"}
            >
              {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
            </Button>
            {messageId !== undefined && (
              <MessageFeedbackControls
                messageId={messageId}
                feedback={feedback ?? null}
                onChange={(next) => onFeedbackChange?.(next)}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
