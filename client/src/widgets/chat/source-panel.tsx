"use client";
import type { Source } from "@/shared/api/types";
import { Button } from "@/shared/ui/button";
import { X, FileText, Hash } from "lucide-react";
import { ScrollArea } from "@/shared/ui/scroll-area";

interface Props { sources: Source[]; onClose: () => void; }

export function SourcePanel({ sources, onClose }: Props) {
  return (
    <div className="w-80 border-l bg-background flex flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="font-semibold text-sm">Sources</h2>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}><X className="h-4 w-4" /></Button>
      </div>
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {sources.map((s, i) => (
            <div key={i} className="rounded-md border p-3 space-y-2">
              <div className="flex items-center gap-2"><FileText className="h-4 w-4 text-muted-foreground shrink-0" /><span className="text-sm font-medium truncate">{s.source}</span></div>
              {s.pages && s.pages.length > 0 && <div className="flex items-center gap-1 text-xs text-muted-foreground"><Hash className="h-3 w-3" /><span>Pages: {s.pages.join(", ")}</span></div>}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
