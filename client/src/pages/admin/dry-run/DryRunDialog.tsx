"use client";
import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2, Upload } from "lucide-react";
import toast from "react-hot-toast";
import { useDryRun } from "@/shared/api/hooks";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/shared/ui/dialog";

export function DryRunDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dryRun = useDryRun();
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);

  const processFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        toast.error("Only PDF files are supported");
        return;
      }
      try {
        const res = await dryRun.mutateAsync(file);
        onClose();
        navigate("/admin/quality/dry-run", { state: res });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Dry-run failed";
        toast.error(msg);
      }
    },
    [dryRun, navigate, onClose],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile],
  );

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-lg w-[90vw]">
        <DialogHeader>
          <DialogTitle>Dry-Run Preview</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Upload a PDF to preview extracted text and quality assessment without indexing.
          </p>
          <label
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/25 hover:border-muted-foreground/50"
            }`}
          >
            <div className="flex flex-col items-center gap-2 text-center">
              {dryRun.isPending ? (
                <Loader2 className="h-10 w-10 text-muted-foreground animate-spin" />
              ) : (
                <Upload className="h-10 w-10 text-muted-foreground" />
              )}
              <span className="text-sm font-medium">
                {dryRun.isPending
                  ? "Analyzing..."
                  : isDragging
                    ? "Drop PDF here"
                    : "Drop PDF or click to browse"}
              </span>
              <span className="text-xs text-muted-foreground">PDF files only, max 50 MB</span>
            </div>
            <input type="file" accept=".pdf" onChange={handleFileInput} className="hidden" />
          </label>
        </div>
      </DialogContent>
    </Dialog>
  );
}
