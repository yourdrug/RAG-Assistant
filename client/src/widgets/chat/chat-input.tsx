"use client";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/shared/ui/button";
import { Send, Square } from "lucide-react";
import { Textarea } from "@/shared/ui/textarea";

export type DepthOption = "short" | "detailed" | null;

interface Props {
  onSend: (msg: string) => void;
  onStop: () => void;
  disabled?: boolean;
  depth: DepthOption;
  onDepthChange: (d: DepthOption) => void;
}

export function ChatInput({ onSend, onStop, disabled, depth, onDepthChange }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (ref.current) { ref.current.style.height = "auto"; ref.current.style.height = `${Math.min(ref.current.scrollHeight, 200)}px`; } }, [value]);

  const submit = () => { if (!value.trim() || disabled) return; onSend(value.trim()); setValue(""); };
  const handleKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } };

  const depthButtons: { value: DepthOption; label: string }[] = [
    { value: null, label: "Авто" },
    { value: "short", label: "Кратко" },
    { value: "detailed", label: "Подробно" },
  ];

  return (
    <div className="border-t p-4">
      <div className="max-w-3xl mx-auto space-y-2">
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground mr-1">Ответ:</span>
          {depthButtons.map((opt) => (
            <button
              key={opt.label}
              onClick={() => onDepthChange(opt.value)}
              disabled={disabled}
              className={`text-xs px-2 py-1 rounded-md transition-colors ${
                depth === opt.value
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="flex items-end gap-2">
          <Textarea ref={ref} value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={handleKey} placeholder="Ask a question about your documents..." disabled={disabled} className="min-h-[44px] max-h-[200px] resize-none" rows={1} />
          {disabled ? <Button size="icon" variant="destructive" onClick={onStop} className="shrink-0"><Square className="h-4 w-4" /></Button>
            : <Button size="icon" onClick={submit} disabled={!value.trim()} className="shrink-0"><Send className="h-4 w-4" /></Button>}
        </div>
        <p className="text-center text-xs text-muted-foreground">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}
