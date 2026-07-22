"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useDocuments, useUploadDocument, useDeleteDocument, useGroups } from "@/shared/api/hooks";
import { DataTable } from "@/shared/ui/data-table";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Progress } from "@/shared/ui/progress";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/shared/ui/alert-dialog";
import { Trash2, FileText, CloudUpload } from "lucide-react";
import { type ColumnDef } from "@tanstack/react-table";
import toast from "react-hot-toast";
import type { DocumentResponse, DocumentVisibility } from "@/shared/api/types";

export function DocumentsPage() {
  const { data: documents } = useDocuments();
  const uploadMut = useUploadDocument();
  const deleteMut = useDeleteDocument();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [vis, setVis] = useState<DocumentVisibility>("internal_private");
  const [groupId, setGroupId] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const { data: groups } = useGroups();

  const onDrop = useCallback((f: File[]) => { setFiles(f); setUploadOpen(true); }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"], "text/markdown": [".md"], "text/plain": [".txt"] },
  });

  const handleUpload = async () => {
    if (!files.length) return;
    setProgress(0);
    let done = 0;
    for (const f of files) {
      try {
        await uploadMut.mutateAsync({ file: f, visibility: vis, groupId: vis === "internal_group" ? groupId : undefined });
        done++;
        setProgress(Math.round((done / files.length) * 100));
      } catch {
        toast.error(`Failed: ${f.name}`);
      }
    }
    setUploadOpen(false);
    setFiles([]);
    setProgress(0);
    if (done > 0) {
      toast.success(`${done} file(s) queued for processing`);
    }
  };

  const handleDelete = async () => {
    if (deleteId === null) return;
    try { await deleteMut.mutateAsync(deleteId); toast.success("Deleted"); } catch { toast.error("Failed"); }
    setDeleteId(null);
  };

  const columns: ColumnDef<DocumentResponse>[] = [
    { accessorKey: "filename", header: "Filename", cell: ({ row }) => <div className="flex items-center gap-2"><FileText className="h-4 w-4 text-muted-foreground" /><span className="font-medium">{row.original.filename}</span></div> },
    { accessorKey: "visibility", header: "Visibility", cell: ({ row }) => <Badge variant="secondary">{row.original.visibility.replace(/_/g, " ")}</Badge> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <Badge variant={row.original.status === "done" ? "success" : row.original.status === "failed" ? "destructive" : row.original.status === "processing" ? "warning" : "secondary"}>{row.original.status}</Badge> },
    { accessorKey: "chunks", header: "Chunks", cell: ({ row }) => row.original.chunks ?? "—" },
    { accessorKey: "chars", header: "Chars", cell: ({ row }) => row.original.chars?.toLocaleString() ?? "—" },
    { id: "actions", header: "", cell: ({ row }) => <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setDeleteId(row.original.id)}><Trash2 className="h-4 w-4 text-destructive" /></Button> },
  ];

  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">Documents</h1><p className="text-muted-foreground">Manage your documents</p></div>
      <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-muted-foreground/50"}`}>
        <input {...getInputProps()} />
        <CloudUpload className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Drag & drop files here, or click to select</p>
        <p className="text-xs text-muted-foreground/70 mt-1">PDF, DOCX, MD, TXT</p>
      </div>
      <Card>
        <CardHeader><CardTitle>All Documents</CardTitle><CardDescription>{documents?.length || 0} document(s)</CardDescription></CardHeader>
        <CardContent><DataTable columns={columns} data={documents || []} searchKey="filename" searchPlaceholder="Search documents..." /></CardContent>
      </Card>

      <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Upload Documents</DialogTitle><DialogDescription>{files.length} file(s) selected</DialogDescription></DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Visibility</label>
              <Select value={vis} onValueChange={(v) => { setVis(v as DocumentVisibility); if (v !== "internal_group") setGroupId(null); }}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="internal_private">Private</SelectItem>
                  <SelectItem value="internal_public">Public</SelectItem>
                  <SelectItem value="internal_group">Group</SelectItem>
                  <SelectItem value="client_private">Client Private</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {vis === "internal_group" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Group</label>
                <Select value={groupId != null ? String(groupId) : ""} onValueChange={(v) => setGroupId(Number(v))}>
                  <SelectTrigger><SelectValue placeholder="Select a group" /></SelectTrigger>
                  <SelectContent>
                    {groups?.map((g) => (
                      <SelectItem key={g.id} value={String(g.id)}>{g.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            {files.length > 0 && <div className="space-y-1">{files.map((f, i) => <div key={i} className="flex items-center justify-between text-sm"><span className="truncate">{f.name}</span><span className="text-muted-foreground">{(f.size / 1024).toFixed(1)} KB</span></div>)}</div>}
            {progress > 0 && <Progress value={progress} />}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadOpen(false)}>Cancel</Button>
            <Button onClick={handleUpload} disabled={uploadMut.isPending || (vis === "internal_group" && groupId == null)}>{uploadMut.isPending ? "Uploading..." : "Upload"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteId !== null} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>Delete Document</AlertDialogTitle><AlertDialogDescription>This action cannot be undone.</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>Cancel</AlertDialogCancel><AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">Delete</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
