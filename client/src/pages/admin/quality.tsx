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
import { useState } from "react";
import { useDocumentDiagnosis, useDocuments } from "@/shared/api/hooks";
import type {
  DocumentQualityItem,
  PageDiagnostic,
} from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { DryRunDialog } from "./dry-run/DryRunDialog";

export function AdminQualityPage() {
  const { data: documents } = useDocuments();
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [showDryRun, setShowDryRun] = useState(false);

  const qualityDocs = (documents || []).filter(
    (d) => d.warning_message || (d.quality_score != null && d.quality_score > 0.3),
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
        const variant = pct <= 10 ? "success" : pct <= 30 ? "warning" : "destructive";
        return <Badge variant={variant}>{pct}% bad</Badge>;
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
              setSelectedDocId(selectedDocId === row.original.id ? null : row.original.id)
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

      {selectedDocId && <DiagnosisPanel documentId={selectedDocId} />}

      {showDryRun && <DryRunDialog open={showDryRun} onClose={() => setShowDryRun(false)} />}
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
          <SummaryCard
            label="Text"
            value={data.summary.text}
            icon={<CheckCircle2 className="h-4 w-4 text-green-500" />}
          />
          <SummaryCard
            label="Scan"
            value={data.summary.scan}
            icon={<AlertTriangle className="h-4 w-4 text-yellow-500" />}
          />
          <SummaryCard
            label="Garbled"
            value={data.summary.garbled}
            icon={<XCircle className="h-4 w-4 text-red-500" />}
          />
          <SummaryCard
            label="Table"
            value={data.summary.table}
            icon={<Info className="h-4 w-4 text-blue-500" />}
          />
          <SummaryCard
            label="Empty"
            value={data.summary.empty}
            icon={<Info className="h-4 w-4 text-gray-400" />}
          />
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

function SummaryCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
}) {
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
    cell: ({ row }) => <span className="font-mono text-muted-foreground">{row.original.page}</span>,
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
