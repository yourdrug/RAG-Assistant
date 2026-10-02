"use client";
import { CalendarClock, FileText, Hash, Pencil, X } from "lucide-react";
import { Link } from "react-router-dom";
import type { Source } from "@/shared/api/types";
import { documentUrl } from "@/shared/lib/document-citations";
import { Button } from "@/shared/ui/button";
import { ScrollArea } from "@/shared/ui/scroll-area";

function formatRuDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const [y, m, d] = iso.split("-");
  return d && m && y ? `${d}.${m}.${y}` : iso;
}

interface Props {
  sources: Source[];
  onClose: () => void;
}

export function SourcePanel({ sources, onClose }: Props) {
  return (
    <aside
      className="absolute inset-y-0 right-0 z-20 w-80 max-w-full border-l bg-background flex flex-col shadow-lg lg:relative lg:shadow-none"
      aria-label="Источники ответа"
    >
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="font-semibold text-sm">Источники</h2>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={onClose}
          aria-label="Закрыть источники"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {sources.map((s, i) => (
            <div
              key={`${s.document_id ?? s.source}-${i}`}
              className="rounded-md border p-3 space-y-2 overflow-hidden"
            >
              <div className="flex items-start gap-2 min-w-0">
                <FileText className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                {documentUrl(s) ? (
                  <Link
                    to={documentUrl(s)!}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium break-words min-w-0 text-primary hover:underline"
                  >
                    {s.citation_id != null ? `[${s.citation_id}] ` : ""}
                    {s.source}
                  </Link>
                ) : (
                  <span className="text-sm font-medium break-words min-w-0">
                    {s.citation_id != null ? `[${s.citation_id}] ` : ""}
                    {s.source}
                  </span>
                )}
              </div>
              {s.pages && s.pages.length > 0 && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Hash className="h-3 w-3" />
                  <span>Страницы: {s.pages.join(", ")}</span>
                </div>
              )}
              {s.articles && s.articles.length > 0 && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Hash className="h-3 w-3" />
                  <span>Статьи: {s.articles.join(", ")}</span>
                </div>
              )}
              {(s.act_number || s.effective_from || s.effective_to) && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <CalendarClock className="h-3 w-3" />
                  <span>
                    {s.act_number ? `Акт №${s.act_number}` : null}
                    {s.effective_from ? ` · с ${formatRuDate(s.effective_from)}` : ""}
                    {s.effective_to ? ` по ${formatRuDate(s.effective_to)}` : ""}
                  </span>
                </div>
              )}
              {s.edited_at && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Pencil className="h-3 w-3" />
                  <span>Изменён: {new Date(s.edited_at).toLocaleDateString("ru-RU")}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>
    </aside>
  );
}
