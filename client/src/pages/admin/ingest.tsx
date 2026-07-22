"use client";
import { useState } from "react";
import { useIngestAll, useIngestFile, useIngestRegistry } from "@/shared/api/hooks";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Badge } from "@/shared/ui/badge";
import { DataTable } from "@/shared/ui/data-table";
import { type ColumnDef } from "@tanstack/react-table";
import { Upload, RefreshCw, RotateCcw } from "lucide-react";
import toast from "react-hot-toast";
import type { IngestRegistryItem } from "@/shared/api/types";

export function AdminIngestPage() {
  const { data: registry, refetch } = useIngestRegistry();
  const allMut = useIngestAll();
  const fileMut = useIngestFile();
  const [docsDir, setDocsDir] = useState("data/docs_sample");
  const [filePath, setFilePath] = useState("");

  const [resetAll, setResetAll] = useState(false);

  const handleAll = async () => {
    try { await allMut.mutateAsync({ docsDir, reset: resetAll }); toast.success("Ingestion started" + (resetAll ? " (reset mode)" : "")); } catch { toast.error("Failed"); }
  };

  const handleFile = async () => {
    if (!filePath.trim()) return;
    try { await fileMut.mutateAsync({ filePath, force: false }); toast.success("File ingestion started"); } catch { toast.error("Failed"); }
  };

  const columns: ColumnDef<IngestRegistryItem>[] = [
    { accessorKey: "filename", header: "Filename", cell: ({ row }) => <span className="font-medium">{row.original.filename}</span> },
    { accessorKey: "chunks", header: "Chunks" },
    { accessorKey: "chars", header: "Chars", cell: ({ row }) => row.original.chars.toLocaleString() },
    { accessorKey: "source", header: "Source", cell: ({ row }) => <Badge variant="outline">{row.original.source}</Badge> },
    { accessorKey: "indexed_at", header: "Indexed At", cell: ({ row }) => new Date(row.original.indexed_at).toLocaleString() },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold">Ingestion</h1><p className="text-muted-foreground">Manage document ingestion</p></div>
        <Button variant="outline" onClick={() => refetch()}><RefreshCw className="h-4 w-4 mr-2" />Refresh</Button>
      </div>

      <Card>
        <CardHeader><CardTitle>Ingest Directory</CardTitle><CardDescription>Trigger ingestion for all files in a directory</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="flex-1"><Label>Directory Path</Label><Input value={docsDir} onChange={(e) => setDocsDir(e.target.value)} placeholder="data/docs_sample" /></div>
            <Button onClick={handleAll} disabled={allMut.isPending} className="mt-6"><Upload className="h-4 w-4 mr-2" />{allMut.isPending ? "Starting..." : "Ingest All"}</Button>
          </div>
          <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={resetAll}
              onChange={(e) => setResetAll(e.target.checked)}
              className="h-4 w-4 rounded border-input accent-primary"
            />
            <RotateCcw className="h-3.5 w-3.5" />
            Reset mode — re-index all files from scratch
          </label>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Ingest Single File</CardTitle><CardDescription>Ingest a specific file by path</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="flex-1"><Label>File Path</Label><Input value={filePath} onChange={(e) => setFilePath(e.target.value)} placeholder="data/docs_sample/example.pdf" /></div>
            <Button onClick={handleFile} disabled={!filePath.trim() || fileMut.isPending} className="mt-6">{fileMut.isPending ? "Starting..." : "Ingest File"}</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Ingestion Registry</CardTitle><CardDescription>{registry?.total_files || 0} files · {registry?.total_chunks?.toLocaleString() || 0} chunks</CardDescription></CardHeader>
        <CardContent><DataTable columns={columns} data={registry?.files || []} searchKey="filename" searchPlaceholder="Search registry..." /></CardContent>
      </Card>
    </div>
  );
}
