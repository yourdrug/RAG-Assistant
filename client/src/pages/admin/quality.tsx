"use client";
import { type ColumnDef } from "@tanstack/react-table";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Info,
  Loader2,
  Upload,
  XCircle,
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import toast from "react-hot-toast";
import {
  useDocumentDiagnosis,
  useDocuments,
  useDryRun,
  useDryRunOcr,
} from "@/shared/api/hooks";
import type {
  DocumentQualityItem,
  DryRunResponse,
  PageDiagnostic,
} from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { ScrollArea } from "@/shared/ui/scroll-area";

export function AdminQualityPage() {
  const { data: documents } = useDocuments();
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [showDryRun, setShowDryRun] = useState(false);

  const qualityDocs = (documents || []).filter(
    (d) =>
      d.warning_message ||
      (d.quality_score != null && d.quality_score > 0.3),
  );

  const columns: ColumnDef<DocumentQualityItem>[] = [
    {
      accessorKey: "id",
      header: "ID",
      cell: ({ row }) => <span className="text-muted-foreground">#{row.original.id}</span>,
    },
    {
      accessorKey: "filename",
      header: "Filename",
      cell: ({ row }) => (
        <div className="flex items-center gap-2 max-w-xs">
          <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="font-medium truncate">{row.original.filename}</span>
        </div>
      ),
    },
    {
      accessorKey: "quality_score",
      header: "Quality",
      cell: ({ row }) => {
        const score = row.original.quality_score;
        if (score == null) return "—";
        const pct = Math.round(score * 100);
        const variant =
          pct <= 10 ? "success" : pct <= 30 ? "warning" : "destructive";
        return (
          <Badge variant={variant}>
            {pct}% bad
          </Badge>
        );
      },
    },
    {
      accessorKey: "warning_message",
      header: "Warning",
      cell: ({ row }) => (
        <p className="text-xs text-muted-foreground max-w-sm truncate">
          {row.original.warning_message || "—"}
        </p>
      ),
    },
    {
      accessorKey: "chunks",
      header: "Chunks",
      cell: ({ row }) => row.original.chunks ?? "—",
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => (
        <div className="flex gap-1">
          <Button
            variant={selectedDocId === row.original.id ? "default" : "outline"}
            size="sm"
            onClick={() =>
              setSelectedDocId(
                selectedDocId === row.original.id ? null : row.original.id,
              )
            }
          >
            {selectedDocId === row.original.id ? "Hide" : "Diagnose"}
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Indexing Quality</h1>
          <p className="text-muted-foreground">
            Documents with extraction issues and diagnostic tools
          </p>
        </div>
        <Button onClick={() => setShowDryRun(true)}>
          <Upload className="h-4 w-4 mr-2" />
          Dry-Run Preview
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Documents with Warnings ({qualityDocs.length})
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <DataTable
            columns={columns}
            data={qualityDocs as DocumentQualityItem[]}
            searchKey="filename"
            searchPlaceholder="Search documents..."
          />
        </CardContent>
      </Card>

      {selectedDocId && (
        <DiagnosisPanel documentId={selectedDocId} />
      )}

      {showDryRun && (
        <DryRunDialog open={showDryRun} onClose={() => setShowDryRun(false)} />
      )}
    </div>
  );
}

function DiagnosisPanel({ documentId }: { documentId: number }) {
  const { data, isLoading } = useDocumentDiagnosis(documentId);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin" />
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Page Diagnosis: {data.filename} ({data.total_pages} pages)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-5 gap-4 mb-4">
          <SummaryCard label="Text" value={data.summary.text} icon={<CheckCircle2 className="h-4 w-4 text-green-500" />} />
          <SummaryCard label="Scan" value={data.summary.scan} icon={<AlertTriangle className="h-4 w-4 text-yellow-500" />} />
          <SummaryCard label="Garbled" value={data.summary.garbled} icon={<XCircle className="h-4 w-4 text-red-500" />} />
          <SummaryCard label="Table" value={data.summary.table} icon={<Info className="h-4 w-4 text-blue-500" />} />
          <SummaryCard label="Empty" value={data.summary.empty} icon={<Info className="h-4 w-4 text-gray-400" />} />
        </div>

        <ScrollArea className="h-80">
          <DataTable
            columns={pageColumns}
            data={data.pages}
            searchKey="description"
            searchPlaceholder="Filter pages..."
          />
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function SummaryCard({ label, value, icon }: { label: string; value: number; icon: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 p-3 rounded-lg border">
      {icon}
      <div>
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

const pageColumns: ColumnDef<PageDiagnostic>[] = [
  {
    accessorKey: "page",
    header: "Page",
    cell: ({ row }) => (
      <span className="font-mono text-muted-foreground">{row.original.page}</span>
    ),
  },
  {
    accessorKey: "type",
    header: "Type",
    cell: ({ row }) => {
      const variant =
        row.original.type === "text"
          ? "success"
          : row.original.type === "scan" || row.original.type === "garbled"
            ? "destructive"
            : row.original.type === "table"
              ? "default"
              : "secondary";
      return <Badge variant={variant}>{row.original.type}</Badge>;
    },
  },
  {
    accessorKey: "chars",
    header: "Chars",
    cell: ({ row }) => row.original.chars.toLocaleString(),
  },
  {
    accessorKey: "description",
    header: "Description",
    cell: ({ row }) => (
      <span className="text-sm text-muted-foreground">{row.original.description}</span>
    ),
  },
];

function DryRunDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dryRun = useDryRun();
  const [result, setResult] = useState<DryRunResponse | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [fileRef, setFileRef] = useState<File | null>(null);

  const processFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        toast.error("Only PDF files are supported");
        return;
      }
      try {
        setFileRef(file);
        const res = await dryRun.mutateAsync(file);
        setResult(res);
        setShowDetails(false);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Dry-run failed";
        toast.error(msg);
      }
    },
    [dryRun],
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
      <DialogContent className="max-w-3xl max-h-[85vh]">
        <DialogHeader>
          <DialogTitle>Dry-Run Preview</DialogTitle>
        </DialogHeader>

        {!result ? (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Drag and drop a PDF here, or click to browse. Preview extracted text
              and quality assessment without indexing.
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
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileInput}
                className="hidden"
              />
            </label>
          </div>
        ) : (
          <DryRunResult
            result={result}
            fileRef={fileRef}
            setResult={setResult}
            showDetails={showDetails}
            setShowDetails={setShowDetails}
            onReset={() => { setResult(null); setFileRef(null); }}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Dry-run result view — two-phase
// ---------------------------------------------------------------------------

const TYPE_COLORS: Record<string, string> = {
  text: "bg-green-500",
  table: "bg-blue-500",
  scan: "bg-yellow-500",
  garbled: "bg-red-500",
  empty: "bg-gray-300 dark:bg-gray-600",
};

const TYPE_LABELS: Record<string, string> = {
  text: "Text",
  scan: "Scan (no OCR)",
  garbled: "Garbled",
  empty: "Empty",
  table: "Table",
};

function DryRunResult({
  result,
  fileRef,
  setResult,
  showDetails,
  setShowDetails,
  onReset,
}: {
  result: DryRunResponse;
  fileRef: File | null;
  setResult: (r: DryRunResponse) => void;
  showDetails: boolean;
  setShowDetails: (v: boolean) => void;
  onReset: () => void;
}) {
  const dryRunOcr = useDryRunOcr();
  const { summary, pages, total_pages, total_chars } = result;
  const okCount = summary.text + summary.table;
  const okPct = total_pages > 0 ? Math.round((okCount / total_pages) * 100) : 0;

  // Problem pages for OCR: scan + empty
  const ocrTargetPages = useMemo(
    () => pages.filter((p) => p.type === "scan" || p.type === "empty").map((p) => p.page),
    [pages],
  );

  // Detect anomalies: skip leading short pages (cover/titlepage)
  const anomalies = useMemo(() => {
    const leadingShortEnd = (() => {
      for (let i = 0; i < pages.length; i++) {
        if (pages[i].chars >= 50 || pages[i].type === "table") return i;
      }
      return pages.length;
    })();

    return pages.filter((p, idx) => {
      if (p.type === "text" && p.chars >= 50) return false;
      if (p.type === "table") return false;
      if (idx < leadingShortEnd && p.chars < 50) return false;
      return true;
    });
  }, [pages]);

  const problemCount = anomalies.length;

  const handleRunOcr = useCallback(async () => {
    if (!fileRef || ocrTargetPages.length === 0) return;
    try {
      const updated = await dryRunOcr.mutateAsync({
        file: fileRef,
        pages: ocrTargetPages,
      });
      setResult(updated);
      toast.success(`OCR completed on ${ocrTargetPages.length} pages`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "OCR failed";
      toast.error(msg);
    }
  }, [fileRef, ocrTargetPages, dryRunOcr, setResult]);

  return (
    <ScrollArea className="h-[65vh]">
      <div className="space-y-5">
        {/* ── 1. Summary ────────────────────────────────────────── */}
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold">
            {okPct}% OK
          </span>
          <span className="text-muted-foreground">·</span>
          <span className="text-sm">
            {total_pages} pages · {total_chars.toLocaleString()} chars
          </span>
          {problemCount > 0 && (
            <>
              <span className="text-muted-foreground">·</span>
              <Badge variant={okPct >= 90 ? "warning" : "destructive"}>
                {problemCount} need attention
              </Badge>
            </>
          )}
        </div>

        {/* ── 2. Breakdown chips ───────────────────────────────── */}
        <div className="flex flex-wrap gap-2 text-xs">
          {summary.text > 0 && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              {summary.text} text
            </span>
          )}
          {summary.table > 0 && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              {summary.table} table{summary.table > 1 ? "s" : ""}
            </span>
          )}
          {summary.scan > 0 && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-yellow-50 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300">
              <span className="w-2 h-2 rounded-full bg-yellow-500" />
              {summary.scan} scan
            </span>
          )}
          {summary.garbled > 0 && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              {summary.garbled} garbled
            </span>
          )}
          {summary.empty > 0 && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
              <span className="w-2 h-2 rounded-full bg-gray-400" />
              {summary.empty} empty
            </span>
          )}
        </div>

        {/* ── 3. Heatmap strip ─────────────────────────────────── */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-muted-foreground">Page heatmap</span>
            <span className="text-xs text-muted-foreground">{total_pages} pages</span>
          </div>
          <div className="flex gap-px rounded-md overflow-hidden" style={{ height: 24 }}>
            {pages.map((p) => (
              <HeatmapSegment key={p.page} page={p} />
            ))}
          </div>
        </div>

        {/* ── 4. Anomaly list ──────────────────────────────────── */}
        {anomalies.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium">
                Pages needing attention ({anomalies.length})
              </h4>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={() => setShowDetails(!showDetails)}
              >
                {showDetails ? "Hide details" : "Show all pages"}
              </Button>
            </div>
            <div className="space-y-1">
              {anomalies.map((p) => (
                <div
                  key={p.page}
                  className="flex items-center gap-3 px-3 py-1.5 rounded-md bg-muted/50 text-sm"
                >
                  <span className="font-mono text-muted-foreground w-12">p.{p.page}</span>
                  <Badge
                    variant={
                      p.type === "scan" || p.type === "garbled"
                        ? "destructive"
                        : p.type === "table"
                          ? "default"
                          : "secondary"
                    }
                    className="text-xs"
                  >
                    {p.type}
                  </Badge>
                  <span className="text-muted-foreground text-xs">
                    {p.chars === 0
                      ? "no text"
                      : p.chars < 50
                        ? `${p.chars} chars — likely cover/short`
                        : `${p.chars.toLocaleString()} chars`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {anomalies.length === 0 && (
          <div className="flex items-center gap-2 p-3 rounded-md bg-green-50 text-green-700 text-sm dark:bg-green-950 dark:text-green-300">
            <CheckCircle2 className="h-4 w-4" />
            All pages look good — no anomalies detected.
          </div>
        )}

        {/* ── 5. Details toggle (full page list) ──────────────── */}
        {showDetails && (
          <div>
            <h4 className="text-sm font-medium mb-2">All pages</h4>
            <ScrollArea className="h-48">
              <div className="space-y-0.5">
                {pages.map((p) => (
                  <div
                    key={p.page}
                    className="flex items-center gap-3 px-3 py-1 text-xs"
                  >
                    <span className="font-mono text-muted-foreground w-12">p.{p.page}</span>
                    <Badge
                      variant={
                        p.type === "text"
                          ? "success"
                          : p.type === "scan" || p.type === "garbled"
                            ? "destructive"
                            : p.type === "table"
                              ? "default"
                              : "secondary"
                      }
                      className="text-xs"
                    >
                      {p.type}
                    </Badge>
                    <span className="text-muted-foreground">
                      {p.chars.toLocaleString()} chars
                    </span>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

        {/* ── 6. Actions ──────────────────────────────────────── */}
        <div className="flex items-center gap-2 pt-2 border-t">
          {ocrTargetPages.length > 0 && (
            <Button
              onClick={handleRunOcr}
              disabled={dryRunOcr.isPending}
              size="sm"
            >
              {dryRunOcr.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : null}
              Run OCR on {ocrTargetPages.length} problem pages
            </Button>
          )}
          <Button onClick={onReset} variant="outline" size="sm">
            Upload Another
          </Button>
        </div>
      </div>
    </ScrollArea>
  );
}

// ---------------------------------------------------------------------------
// Heatmap segment with tooltip
// ---------------------------------------------------------------------------

function HeatmapSegment({ page }: { page: { page: number; type: string; chars: number } }) {
  const color = TYPE_COLORS[page.type] || "bg-gray-300";
  const label = TYPE_LABELS[page.type] || page.type;

  return (
    <div className="group relative flex-1 min-w-[2px]">
      <div className={`${color} w-full h-full`} />
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 rounded bg-popover text-popover-foreground text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 shadow-md">
        p.{page.page} · {label} · {page.chars} chars
      </div>
    </div>
  );
}
