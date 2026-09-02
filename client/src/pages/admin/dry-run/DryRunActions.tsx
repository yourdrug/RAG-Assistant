import { FileText, Loader2, Upload } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  useDryRunOcr,
  useGroups,
  useIndexFromPreview,
  useUploadableClients,
} from "@/shared/api/hooks";
import type { DocumentDomain, DocumentVisibility, DryRunResponse } from "@/shared/api/types";
import { Button } from "@/shared/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { useAuthStore } from "@/stores/auth-store";

interface DryRunActionsProps {
  result: DryRunResponse;
  onResultUpdate: (r: DryRunResponse) => void;
  onReset: () => void;
}

export function DryRunActions({ result, onResultUpdate, onReset }: DryRunActionsProps) {
  const dryRunOcr = useDryRunOcr();
  const indexFromPreview = useIndexFromPreview();
  const { data: groups } = useGroups();
  const { data: uploadableClients } = useUploadableClients();
  const user = useAuthStore((s) => s.user);
  const isClient = user?.kind === "client";
  const isAdmin = user?.role === "admin";

  const [showIndexDialog, setShowIndexDialog] = useState(false);
  const [indexVisibility, setIndexVisibility] = useState<DocumentVisibility>("internal_public");
  const [indexGroupId, setIndexGroupId] = useState<number | null>(null);
  const [indexClientId, setIndexClientId] = useState<number | null>(null);
  const [indexDocDomain, setIndexDocDomain] = useState<DocumentDomain | "auto">("auto");

  const isRtf = result.pages.length === 1 && result.pages[0].unit_kind === "document";

  const ocrTargetPages = result.pages
    .filter((p) => p.type === "scan" || p.type === "empty" || p.type === "image_only")
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
      toast.success(`OCR completed on ${pages.length} unit${pages.length > 1 ? "s" : ""}`);
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
        groupId: indexVisibility === "internal_group" ? indexGroupId : undefined,
        clientId: indexVisibility === "client_private" ? indexClientId : undefined,
        docDomain: indexDocDomain === "auto" ? undefined : indexDocDomain,
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
        {!isRtf && ocrTargetPages.length > 0 && (
          <Button
            onClick={() => handleRunOcr(ocrTargetPages)}
            disabled={dryRunOcr.isPending}
            size="sm"
          >
            {dryRunOcr.isPending && <Loader2 className="h-4 w-4 mr-1 animate-spin" />}
            Run OCR on {ocrTargetPages.length} image
            {ocrTargetPages.length > 1 ? "s" : ""}
          </Button>
        )}
        {!isRtf && (
          <Button onClick={() => handleRunOcr(allPages)} variant="outline" size="sm">
            Run OCR on all
          </Button>
        )}
        <Button onClick={() => setShowIndexDialog(true)} variant="default" size="sm">
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
              <Select
                value={indexVisibility}
                onValueChange={(v) => {
                  setIndexVisibility(v as DocumentVisibility);
                  if (v !== "internal_group") setIndexGroupId(null);
                  if (v !== "client_private") setIndexClientId(null);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {isClient ? (
                    <SelectItem value="client_private">Client Private</SelectItem>
                  ) : isAdmin ? (
                    <>
                      <SelectItem value="internal_private">Private</SelectItem>
                      <SelectItem value="internal_public">Public</SelectItem>
                      <SelectItem value="internal_group">Group</SelectItem>
                      <SelectItem value="client_private">Client Private</SelectItem>
                    </>
                  ) : (
                    <>
                      <SelectItem value="internal_private">Private</SelectItem>
                      <SelectItem value="internal_group">Group</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
            {indexVisibility === "internal_group" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Group</label>
                <Select
                  value={indexGroupId != null ? String(indexGroupId) : ""}
                  onValueChange={(v) => setIndexGroupId(Number(v))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a group" />
                  </SelectTrigger>
                  <SelectContent>
                    {groups?.map((g) => (
                      <SelectItem key={g.id} value={String(g.id)}>
                        {g.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            {indexVisibility === "client_private" && !isClient && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Client</label>
                <Select
                  value={indexClientId != null ? String(indexClientId) : ""}
                  onValueChange={(v) => setIndexClientId(Number(v))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a client" />
                  </SelectTrigger>
                  <SelectContent>
                    {uploadableClients?.map((c) => (
                      <SelectItem key={c.id} value={String(c.id)}>
                        {c.email}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {(!uploadableClients || uploadableClients.length === 0) && (
                  <p className="text-xs text-muted-foreground">No clients assigned to you</p>
                )}
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium">Document Domain</label>
              <Select
                value={indexDocDomain}
                onValueChange={(v) => setIndexDocDomain(v as DocumentDomain | "auto")}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  <SelectItem value="general">General</SelectItem>
                  <SelectItem value="legal">Legal</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowIndexDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleIndex} disabled={indexFromPreview.isPending}>
              {indexFromPreview.isPending && <Loader2 className="h-4 w-4 mr-1 animate-spin" />}
              Index
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
