import { Loader2, RefreshCw } from "lucide-react";
import { useEffect } from "react";
import { usePageImage } from "@/shared/api/hooks";
import type { DryRunPageResult } from "@/shared/api/types";
import { canRetryRequest, getErrorMessage } from "@/shared/lib/errors";
import { Badge } from "@/shared/ui/badge";
import { RequestError } from "@/shared/ui/request-error";

interface DryRunPageDetailProps {
  page: DryRunPageResult | null;
  previewId: string | null;
}

export function DryRunPageDetail({ page, previewId }: DryRunPageDetailProps) {
  const pageImage = usePageImage();
  const { mutate } = pageImage;
  const pageNumber = page?.page;
  const imageAvailable = page?.image_available;

  useEffect(() => {
    if (pageNumber == null || !previewId || !imageAvailable) return;
    mutate({ previewId, page: pageNumber });
  }, [imageAvailable, mutate, pageNumber, previewId]);

  if (!page) {
    return (
      <div className="flex items-center justify-center min-h-100 p-6 text-muted-foreground text-sm">
        Select a unit to view details
      </div>
    );
  }

  const currentImage =
    pageImage.variables?.previewId === previewId && pageImage.variables?.page === pageNumber;
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
          {previewId && (!currentImage || pageImage.isPending) ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : previewId && currentImage && pageImage.isError ? (
            <RequestError
              message={getErrorMessage(pageImage.error, "Could not load the page image.")}
              onRetry={
                canRetryRequest(pageImage.error)
                  ? () => mutate({ previewId, page: page.page })
                  : undefined
              }
            />
          ) : currentImage && pageImage.data ? (
            <img
              src={`data:image/png;base64,${pageImage.data.image_base64}`}
              alt={displayLabel}
              className="w-2/3 h-auto mx-auto"
            />
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
