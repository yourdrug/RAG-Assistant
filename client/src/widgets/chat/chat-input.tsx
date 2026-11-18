"use client";
import { CalendarDays, Send, Square, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/shared/ui/button";
import { DateInput } from "@/shared/ui/date-input";
import { Textarea } from "@/shared/ui/textarea";

export type DepthOption = "narrow" | "broad" | null;

interface Props {
  onSend: (msg: string) => void;
  onStop: () => void;
  disabled?: boolean;
  isStreaming: boolean;
  depth: DepthOption;
  onDepthChange: (d: DepthOption) => void;
  /** ISO date (YYYY-MM-DD) or null for current state */
  asOfDate: string | null;
  onAsOfDateChange: (d: string | null) => void;
}

export function ChatInput({
  onSend,
  onStop,
  disabled,
  isStreaming,
  depth,
  onDepthChange,
  asOfDate,
  onAsOfDateChange,
}: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = `${Math.min(ref.current.scrollHeight, 200)}px`;
    }
  }, [value]);

  const submit = () => {
    if (!value.trim() || disabled || isStreaming) return;
    onSend(value.trim());
    setValue("");
  };
  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const depthButtons: { value: DepthOption; label: string }[] = [
    { value: null, label: "Авто" },
    { value: "narrow", label: "Кратко" },
    { value: "broad", label: "Подробно" },
  ];

  return (
    <div className="shrink-0 border-t bg-background p-3 sm:p-4">
      <div className="max-w-3xl mx-auto space-y-2">
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-xs text-muted-foreground mr-1">Ответ:</span>
          {depthButtons.map((opt) => (
            <button
              key={opt.label}
              onClick={() => onDepthChange(opt.value)}
              disabled={disabled || isStreaming}
              className={`text-xs px-2 py-1 rounded-md transition-colors ${
                depth === opt.value
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              }`}
            >
              {opt.label}
            </button>
          ))}
          <span className="text-xs text-muted-foreground mr-1 ml-3 flex items-center gap-1">
            <CalendarDays className="h-3 w-3" />
            Редакция:
          </span>
          <DateInput
            value={asOfDate ?? ""}
            max={new Date().toISOString().slice(0, 10)}
            onValueChange={(value) => onAsOfDateChange(value || null)}
            disabled={disabled || isStreaming}
            title="Ответ по состоянию на дату (пусто = текущая редакция)"
            className="text-xs px-2 py-1 rounded-md bg-muted text-muted-foreground hover:bg-muted/80 transition-colors outline-none"
          />
          {asOfDate && (
            <button
              onClick={() => onAsOfDateChange(null)}
              disabled={disabled || isStreaming}
              title="Вернуть текущую редакцию"
              className="text-xs px-1.5 py-1 rounded-md bg-muted text-muted-foreground hover:bg-muted/80 transition-colors flex items-center gap-0.5"
            >
              <X className="h-3 w-3" />
              сброс
            </button>
          )}
        </div>
        <div className="flex items-end gap-2">
          <Textarea
            ref={ref}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about your documents..."
            disabled={disabled || isStreaming}
            className="min-h-[44px] min-w-0 max-h-[200px] resize-none"
            rows={1}
          />
          {isStreaming ? (
            <Button
              size="icon"
              variant="destructive"
              onClick={onStop}
              aria-label="Остановить ответ"
              className="shrink-0"
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={submit}
              disabled={disabled || !value.trim()}
              aria-label="Отправить сообщение"
              className="shrink-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <p className="text-center text-xs text-muted-foreground">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
