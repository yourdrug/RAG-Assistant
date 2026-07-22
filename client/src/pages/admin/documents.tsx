"use client";
import { useState } from "react";
import { useDocuments, useDeleteDocument } from "@/shared/api/hooks";
import { DataTable } from "@/shared/ui/data-table";
import { Card, CardContent } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/shared/ui/alert-dialog";
import { type ColumnDef } from "@tanstack/react-table";
import { FileText, Trash2 } from "lucide-react";
import toast from "react-hot-toast";
import type { DocumentResponse } from "@/shared/api/types";

export function AdminDocumentsPage() {
  const { data: documents } = useDocuments();
  const deleteMut = useDeleteDocument();
  const [deleteId, setDeleteId] = useState<number | null>(null);

  const handleDelete = async () => {
    if (deleteId === null) return;
    try { await deleteMut.mutateAsync(deleteId); toast.success("Deleted"); } catch { toast.error("Failed"); }
    setDeleteId(null);
  };

  const columns: ColumnDef<DocumentResponse>[] = [
    { accessorKey: "id", header: "ID", cell: ({ row }) => <span className="text-muted-foreground">#{row.original.id}</span> },
    { accessorKey: "filename", header: "Filename", cell: ({ row }) => <div className="flex items-center gap-2"><FileText className="h-4 w-4 text-muted-foreground" /><span className="font-medium">{row.original.filename}</span></div> },
    { accessorKey: "visibility", header: "Visibility", cell: ({ row }) => <Badge variant="secondary">{row.original.visibility.replace(/_/g, " ")}</Badge> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <Badge variant={row.original.status === "done" ? "success" : row.original.status === "failed" ? "destructive" : "secondary"}>{row.original.status}</Badge> },
    { accessorKey: "chunks", header: "Chunks", cell: ({ row }) => row.original.chunks ?? "—" },
    { accessorKey: "chars", header: "Chars", cell: ({ row }) => row.original.chars?.toLocaleString() ?? "—" },
    { id: "actions", header: "", cell: ({ row }) => <Button variant="ghost" size="icon" onClick={() => setDeleteId(row.original.id)}><Trash2 className="h-4 w-4 text-destructive" /></Button> },
  ];

  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">Documents (Admin)</h1><p className="text-muted-foreground">Full document management</p></div>
      <Card><CardContent className="pt-6"><DataTable columns={columns} data={documents || []} searchKey="filename" searchPlaceholder="Search documents..." /></CardContent></Card>
      <AlertDialog open={deleteId !== null} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>Delete Document</AlertDialogTitle><AlertDialogDescription>This action cannot be undone.</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>Cancel</AlertDialogCancel><AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground">Delete</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
