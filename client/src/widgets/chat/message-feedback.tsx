"use client";
import * as Popover from "@radix-ui/react-popover";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { useMessageFeedback } from "@/shared/api/hooks";
import type { FeedbackReason, MessageFeedback } from "@/shared/api/types";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/button";
import { Textarea } from "@/shared/ui/textarea";

export const FEEDBACK_REASON_LABELS: Record<FeedbackReason, string> = {
  no_answer: "Нет ответа",
  wrong_fact: "Неверный факт",
  irrelevant_sources: "Не те источники",
  incomplete: "Неполный ответ",
  outdated: "Устаревшие сведения",
  too_long: "Слишком длинно",
  other: "Другое",
};

const MAX_COMMENT = 1000;

interface Props {
  messageId: number;
  feedback: MessageFeedback | null;
  onChange: (feedback: MessageFeedback | null) => void;
}

/** 👍/👎 under an answer; clicking the active reaction again clears it. */
export function MessageFeedbackControls({ messageId, feedback, onChange }: Props) {
  const mutation = useMessageFeedback(messageId);
  const [open, setOpen] = useState(false);
  const [reasons, setReasons] = useState<FeedbackReason[]>(feedback?.reasons ?? []);
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);

  const save = (body: Parameters<typeof mutation.mutate>[0]) => {
    setError(null);
    mutation.mutate(body, {
      onSuccess: (saved) => {
        onChange(saved);
        setOpen(false);
      },
      onError: () => setError("Не удалось сохранить оценку"),
    });
  };

  const toggleReason = (reason: FeedbackReason) =>
    setReasons((current) =>
      current.includes(reason) ? current.filter((item) => item !== reason) : [...current, reason],
    );

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className={cn("h-6 w-6", feedback?.rating === "up" && "text-emerald-500")}
        disabled={mutation.isPending}
        onClick={() => save(feedback?.rating === "up" ? null : { rating: "up" })}
        aria-label="Хороший ответ"
        aria-pressed={feedback?.rating === "up"}
      >
        <ThumbsUp className="h-3 w-3" />
      </Button>
      <Popover.Root
        open={open}
        onOpenChange={(next) => {
          if (next && feedback?.rating === "down") {
            save(null);
            return;
          }
          setOpen(next);
        }}
      >
        <Popover.Trigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn("h-6 w-6", feedback?.rating === "down" && "text-red-500")}
            disabled={mutation.isPending}
            aria-label="Плохой ответ"
            aria-pressed={feedback?.rating === "down"}
          >
            <ThumbsDown className="h-3 w-3" />
          </Button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            side="bottom"
            align="start"
            sideOffset={4}
            className="z-50 w-72 space-y-3 rounded-md border bg-popover p-3 text-sm text-popover-foreground shadow-md"
          >
            <p className="font-medium">Что не так с ответом?</p>
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(FEEDBACK_REASON_LABELS) as FeedbackReason[]).map((reason) => (
                <button
                  type="button"
                  key={reason}
                  aria-pressed={reasons.includes(reason)}
                  onClick={() => toggleReason(reason)}
                  className={cn(
                    "rounded-full border px-2 py-0.5 text-xs transition-colors",
                    reasons.includes(reason)
                      ? "border-primary bg-primary text-primary-foreground"
                      : "hover:bg-muted",
                  )}
                >
                  {FEEDBACK_REASON_LABELS[reason]}
                </button>
              ))}
            </div>
            <Textarea
              value={comment}
              maxLength={MAX_COMMENT}
              placeholder="Комментарий (необязательно)"
              aria-label="Комментарий к оценке"
              onChange={(event) => setComment(event.target.value)}
            />
            {error && (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            )}
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                disabled={mutation.isPending}
                onClick={() => save({ rating: "down" })}
              >
                Без пояснений
              </Button>
              <Button
                size="sm"
                disabled={mutation.isPending}
                onClick={() => save({ rating: "down", reasons, comment: comment.trim() || null })}
              >
                Отправить
              </Button>
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
      {error && !open && (
        <span role="alert" className="text-xs text-destructive">
          {error}
        </span>
      )}
    </>
  );
}
