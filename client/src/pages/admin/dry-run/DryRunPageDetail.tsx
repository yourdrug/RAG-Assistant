import { Loader2, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { usePageImage } from "@/shared/api/hooks";
import type { DryRunPageResult } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";

interface DryRunPageDetailProps {
  page: DryRunPageResult | null;
  previewId: string | null;
}

export function DryRunPageDetail({ page, previewId }: DryRunPageDetailProps) {
  const pageImage = usePageImage();
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  useEffect(() => {
    setImageSrc(null);
    if (!page || !previewId || !page.image_available) return;

    pageImage.mutate(
      { previewId, page: page.page },
      {
        onSuccess: (data) => {
          setImageSrc(`data:image/png;base64,${data.image_base64}`);
        },
        onError: () => {
          setImageSrc(null);
        },
      },
    );
  }, [page?.page, previewId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!page) {
    return (
      <div className="flex items-center justify-center min-h-100 p-6 text-muted-foreground text-sm">
        Select a unit to view details
      </div>
    );
  }

  const displayLabel = page.label || `Unit ${page.page}`;

  return (
    <div className="space-y-3 p-3">
      {/* Unit header */}
      <div className="flex items-center gap-2 flex-wrap ">
        <span className="font-mono text-sm font-medium">{displayLabel}</span>
        <Badge
          variant={
            page.type === "text"
              ? "success"
              : page.type === "scan" || page.type === "garbled"
                ? "destructive"
                : page.type === "table"
                  ? "default"
                  : "secondary"
          }
          className="text-xs"
        >
          {page.type}
        </Badge>
        <span className="text-xs text-muted-foreground">{page.chars.toLocaleString()} chars</span>
        {page.previous_type && (
          <span className="inline-flex items-center gap-1 text-xs">
            <RefreshCw className="h-3 w-3 text-blue-500" />
            <Badge variant="secondary" className="text-[10px]">
              {page.previous_type}
            </Badge>
            <span className="text-muted-foreground">&rarr;</span>
            <Badge variant="success" className="text-[10px]">
              {page.type}
            </Badge>
            <span
              className={
                page.chars > 0 ? "text-green-600 dark:text-green-400" : "text-muted-foreground"
              }
            >
              (+{page.chars} chars)
            </span>
          </span>
        )}
      </div>

      {/* Page image — only for PDF pages */}
      {page.unit_kind === "page" && page.image_available && (
        <div className="rounded-md border bg-muted/30">
          {pageImage.isPending ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : imageSrc ? (
            <img src={imageSrc} alt={displayLabel} className="w-2/3 h-auto mx-auto" />
          ) : (
            <div className="flex items-center justify-center py-8 text-xs text-muted-foreground">
              Image not available
            </div>
          )}
        </div>
      )}

      {/* Extracted text */}
      <div>
        <span className="text-xs font-medium text-muted-foreground mb-1 block">Extracted text</span>
        <pre className="text-xs whitespace-pre-wrap font-mono p-3 bg-muted rounded-md max-h-96 overflow-auto scrollbar-thin">
          {page.full_text || page.preview || "No text extracted."}
        </pre>
      </div>
    </div>
  );
}
