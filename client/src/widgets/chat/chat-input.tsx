"use client";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/shared/ui/button";
import { Send, Square } from "lucide-react";
import { Textarea } from "@/shared/ui/textarea";

interface Props { onSend: (msg: string) => void; onStop: () => void; disabled?: boolean; }

export function ChatInput({ onSend, onStop, disabled }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (ref.current) { ref.current.style.height = "auto"; ref.current.style.height = `${Math.min(ref.current.scrollHeight, 200)}px`; } }, [value]);

  const submit = () => { if (!value.trim() || disabled) return; onSend(value.trim()); setValue(""); };
  const handleKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } };

  return (
    <div className="border-t p-4">
      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <Textarea ref={ref} value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={handleKey} placeholder="Ask a question about your documents..." disabled={disabled} className="min-h-[44px] max-h-[200px] resize-none" rows={1} />
        {disabled ? <Button size="icon" variant="destructive" onClick={onStop} className="shrink-0"><Square className="h-4 w-4" /></Button>
          : <Button size="icon" onClick={submit} disabled={!value.trim()} className="shrink-0"><Send className="h-4 w-4" /></Button>}
      </div>
      <p className="text-center text-xs text-muted-foreground mt-2">Enter to send · Shift+Enter for new line</p>
    </div>
  );
}
