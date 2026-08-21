import { Loader2, FileText, Upload } from "lucide-react";
import { Button } from "@/shared/ui/button";
import { useDryRunOcr, useIndexFromPreview } from "@/shared/api/hooks";
import type { DryRunResponse } from "@/shared/api/types";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/shared/ui/dialog";

interface DryRunActionsProps {
  result: DryRunResponse;
  onResultUpdate: (r: DryRunResponse) => void;
  onReset: () => void;
}

export function DryRunActions({
  result,
  onResultUpdate,
  onReset,
}: DryRunActionsProps) {
  const dryRunOcr = useDryRunOcr();
  const indexFromPreview = useIndexFromPreview();
  const [showIndexDialog, setShowIndexDialog] = useState(false);
  const [indexVisibility, setIndexVisibility] = useState("internal_public");
  const [indexDocDomain, setIndexDocDomain] = useState<string>("");

  const ocrTargetPages = result.pages
    .filter((p) => p.type === "scan" || p.type === "empty")
    .map((p) => p.page);

  const allPages = result.pages.map((p) => p.page);

  const handleRunOcr = async (pages: number[]) => {
    if (pages.length === 0) return;
    try {
      const updated = await dryRunOcr.mutateAsync({
        previewId: result.preview_id,
        pages,
      });
      onResultUpdate(updated);
      toast.success(`OCR completed on ${pages.length} pages`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "OCR failed";
      toast.error(msg);
    }
  };

  const handleIndex = async () => {
    if (!result.preview_id) {
      toast.error("Preview expired — upload the file again");
      return;
    }
    try {
      const res = await indexFromPreview.mutateAsync({
        previewId: result.preview_id,
        visibility: indexVisibility,
        docDomain: indexDocDomain || null,
      });
      toast.success(
        <span>
          Document indexed:{" "}
          <a
            href={`/admin/documents`}
            className="underline font-medium"
            target="_blank"
            rel="noreferrer"
          >
            #{res.document_id}
          </a>
        </span>,
        { duration: 8000 },
      );
      setShowIndexDialog(false);
      onReset();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Indexing failed";
      toast.error(msg);
    }
  };

  return (
    <>
      <div className="flex items-center gap-2 pt-2 border-t flex-wrap">
        {ocrTargetPages.length > 0 && (
          <Button
            onClick={() => handleRunOcr(ocrTargetPages)}
            disabled={dryRunOcr.isPending}
            size="sm"
          >
            {dryRunOcr.isPending && (
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
            )}
            Run OCR on {ocrTargetPages.length} page
            {ocrTargetPages.length > 1 ? "s" : ""}
          </Button>
        )}
        <Button
          onClick={() => handleRunOcr(allPages)}
          variant="outline"
          size="sm"
        >
          Run OCR on all pages
        </Button>
        <Button
          onClick={() => setShowIndexDialog(true)}
          variant="default"
          size="sm"
        >
          <FileText className="h-4 w-4 mr-1" />
          Index this file
        </Button>
        <Button onClick={onReset} variant="outline" size="sm">
          <Upload className="h-4 w-4 mr-1" />
          Upload Another
        </Button>
      </div>

      {/* Index confirmation dialog */}
      <Dialog open={showIndexDialog} onOpenChange={setShowIndexDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Index Document</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              This will upload and index <strong>{result.filename}</strong> through the standard
              ingestion pipeline.
            </p>
            <div className="space-y-2">
              <label className="text-sm font-medium">Visibility</label>
              <select
                value={indexVisibility}
                onChange={(e) => setIndexVisibility(e.target.value)}
                className="w-full border rounded-md px-3 py-1.5 text-sm bg-background"
              >
                <option value="internal_public">Internal Public</option>
                <option value="internal_group">Internal Group</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Document Domain</label>
              <select
                value={indexDocDomain}
                onChange={(e) => setIndexDocDomain(e.target.value)}
                className="w-full border rounded-md px-3 py-1.5 text-sm bg-background"
              >
                <option value="">Auto-detect</option>
                <option value="general">General</option>
                <option value="legal">Legal</option>
              </select>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowIndexDialog(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleIndex}
              disabled={indexFromPreview.isPending}
            >
              {indexFromPreview.isPending && (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              )}
              Index
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
