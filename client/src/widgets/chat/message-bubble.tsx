"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, Check, ThumbsUp, ThumbsDown, ExternalLink } from "lucide-react";
import { useState } from "react";
import { Button } from "@/shared/ui/button";
import { Avatar, AvatarFallback } from "@/shared/ui/avatar";
import type { Source } from "@/shared/api/types";
import { cn } from "@/shared/lib/utils";

interface Props { role: "user" | "assistant"; content: string; sources?: Source[]; streaming?: boolean; onSourcesClick?: (s: Source[]) => void; }

export function MessageBubble({ role, content, sources, streaming, onSourcesClick }: Props) {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<"like" | "dislike" | null>(null);
  const isUser = role === "user";

  const handleCopy = async () => { await navigator.clipboard.writeText(content); setCopied(true); setTimeout(() => setCopied(false), 2000); };

  return (
    <div className={cn("group flex gap-3", isUser && "flex-row-reverse")}>
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback className={cn("text-xs", isUser ? "bg-primary text-primary-foreground" : "bg-muted")}>{isUser ? "U" : "AI"}</AvatarFallback>
      </Avatar>
      <div className={cn("max-w-[80%] space-y-2", isUser && "text-right")}>
        <div className={cn("rounded-lg px-4 py-3 text-sm", isUser ? "bg-primary text-primary-foreground" : "bg-muted")}>
          {isUser ? <p className="whitespace-pre-wrap">{content}</p> : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              {streaming && content.length > 0 && <span className="inline-block h-4 w-2 animate-pulse bg-foreground/70" />}
            </div>
          )}
        </div>
        {sources && sources.length > 0 && (
          <button onClick={() => onSourcesClick?.(sources)} className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-1 text-xs text-muted-foreground hover:bg-muted transition-colors">
            <ExternalLink className="h-3 w-3" />{sources.length} source{sources.length > 1 ? "s" : ""}
          </button>
        )}
        {!isUser && !streaming && content && (
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleCopy}>{copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}</Button>
            <Button variant="ghost" size="icon" className={cn("h-6 w-6", liked === "like" && "text-emerald-500")} onClick={() => setLiked(liked === "like" ? null : "like")}><ThumbsUp className="h-3 w-3" /></Button>
            <Button variant="ghost" size="icon" className={cn("h-6 w-6", liked === "dislike" && "text-red-500")} onClick={() => setLiked(liked === "dislike" ? null : "dislike")}><ThumbsDown className="h-3 w-3" /></Button>
          </div>
        )}
      </div>
    </div>
  );
}
