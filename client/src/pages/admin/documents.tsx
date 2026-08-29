"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { Edit3, FileText, Info, Loader2, Plus, Puzzle, Save, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import toast from "react-hot-toast";
import {
  useAddChunk,
  useChunks,
  useCreateManualDocument,
  useDeleteChunk,
  useDeleteDocument,
  useDocuments,
  useUpdateChunk,
} from "@/shared/api/hooks";
import type { ChunkResponse, DocumentResponse } from "@/shared/api/types";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/shared/ui/alert-dialog";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Textarea } from "@/shared/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/shared/ui/tooltip";

export function AdminDocumentsPage() {
  const [searchParams] = useSearchParams();
  const { data: documents } = useDocuments();
  const deleteMut = useDeleteDocument();
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const initialDocId = searchParams.get("docId");
  const highlightHashes = searchParams.get("highlight")?.split(",").filter(Boolean) ?? [];
  const [selectedDocId, setSelectedDocId] = useState<number | null>(
    initialDocId ? Number(initialDocId) : null,
  );
  const [showManualDocDialog, setShowManualDocDialog] = useState(false);

  useEffect(() => {
    const docId = searchParams.get("docId");
    if (docId) {
      setSelectedDocId(Number(docId));
    }
  }, [searchParams]);

  const handleDelete = async () => {
    if (deleteId === null) return;
    try {
      await deleteMut.mutateAsync(deleteId);
      toast.success("Deleted");
    } catch {
      toast.error("Failed");
    }
    setDeleteId(null);
  };

  const columns: ColumnDef<DocumentResponse>[] = [
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
          {row.original.source_type === "manual" ? (
            <Puzzle className="h-4 w-4 shrink-0 text-muted-foreground" />
          ) : (
            <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
          )}
          <span className="font-medium truncate">{row.original.filename}</span>
          {row.original.has_manual_edits && (
            <Badge variant="warning" className="text-xs">
              Edited
            </Badge>
          )}
        </div>
      ),
    },
    {
      accessorKey: "visibility",
      header: "Visibility",
      cell: ({ row }) => {
        const doc = row.original;
        return (
          <div className="flex items-center gap-1.5">
            <Badge variant="secondary">{doc.visibility.replace(/_/g, " ")}</Badge>
            {doc.in_search_scope === false && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Badge variant="warning" className="text-xs gap-1">
                      <Info className="h-3 w-3" />
                      Not in search
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    Document is visible for admin review but not used in RAG search
                    for this administrator — either another admin's private document or
                    a client document not assigned to you.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        );
      },
    },
    {
      accessorKey: "doc_domain",
      header: "Domain",
      cell: ({ row }) => (
        <Badge variant={row.original.doc_domain === "legal" ? "default" : "secondary"}>
          {row.original.doc_domain}
        </Badge>
      ),
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => (
        <Badge
          variant={
            row.original.status === "done"
              ? "success"
              : row.original.status === "failed"
                ? "destructive"
                : "secondary"
          }
        >
          {row.original.status}
        </Badge>
      ),
    },
    { accessorKey: "chunks", header: "Chunks", cell: ({ row }) => row.original.chunks ?? "—" },
    {
      accessorKey: "chars",
      header: "Chars",
      cell: ({ row }) => row.original.chars?.toLocaleString() ?? "—",
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => {
        const doc = row.original;
        const canManageChunks = doc.status === "done" && (doc.chunks ?? 0) > 0;
        return (
          <div className="flex gap-1">
            {canManageChunks && (
              <Button
                variant={selectedDocId === doc.id ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedDocId(selectedDocId === doc.id ? null : doc.id)}
              >
                {selectedDocId === doc.id ? "Hide" : "Chunks"}
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={() => setDeleteId(doc.id)}>
              <Trash2 className="h-4 w-4 text-destructive" />
            </Button>
          </div>
        );
      },
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Documents</h1>
          <p className="text-muted-foreground">Document and chunk management</p>
        </div>
        <Button onClick={() => setShowManualDocDialog(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Create Manual Document
        </Button>
      </div>

      <Card>
        <CardContent className="pt-6">
          <DataTable
            columns={columns}
            data={documents || []}
            searchKey="filename"
            searchPlaceholder="Search documents..."
          />
        </CardContent>
      </Card>

      {selectedDocId && <ChunkManager documentId={selectedDocId} highlightHashes={highlightHashes} />}

      <AlertDialog open={deleteId !== null} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Document</AlertDialogTitle>
            <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {showManualDocDialog && (
        <ManualDocumentDialog
          open={showManualDocDialog}
          onClose={() => setShowManualDocDialog(false)}
        />
      )}
    </div>
  );
}

function ChunkManager({
  documentId,
  highlightHashes,
}: {
  documentId: number;
  highlightHashes?: string[];
}) {
  const hasHighlight = (highlightHashes?.length ?? 0) > 0;
  const highlightParam = hasHighlight ? highlightHashes!.join(",") : undefined;
  const { data: chunkData, isLoading } = useChunks(documentId, undefined, undefined, highlightParam);
  const [editingChunk, setEditingChunk] = useState<ChunkResponse | null>(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [deleteChunkId, setDeleteChunkId] = useState<number | null>(null);

  const deleteMut = useDeleteChunk();

  const handleDeleteChunk = async () => {
    if (deleteChunkId === null) return;
    try {
      await deleteMut.mutateAsync({ documentId, chunkId: deleteChunkId });
      toast.success("Chunk deleted");
    } catch {
      toast.error("Failed to delete chunk");
    }
    setDeleteChunkId(null);
  };

  const columns: ColumnDef<ChunkResponse>[] = [
    {
      accessorKey: "chunk_index",
      header: "#",
      cell: ({ row }) => (
        <span className="text-muted-foreground font-mono">{row.original.chunk_index}</span>
      ),
    },
    {
      accessorKey: "content",
      header: "Content",
      cell: ({ row }) => {
        return (
          <div className="max-w-md">
            <p className={`text-sm line-clamp-2 ${hasHighlight ? "bg-yellow-100 dark:bg-yellow-900/30 rounded px-1 -mx-1" : ""}`}>
              {row.original.content}
            </p>
            <div className="flex gap-1 mt-1">
              {hasHighlight && (
                <Badge variant="warning" className="text-xs">
                  Source
                </Badge>
              )}
              {row.original.manual && (
                <Badge variant="outline" className="text-xs">
                  Manual
                </Badge>
              )}
              {row.original.edited_at && (
                <Badge variant="warning" className="text-xs">
                  Edited
                </Badge>
              )}
            </div>
          </div>
        );
      },
    },
    {
      accessorKey: "id",
      header: "Chunk ID",
      cell: ({ row }) => (
        <span className="text-muted-foreground font-mono text-xs">#{row.original.id}</span>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => (
        <div className="flex gap-1">
          <Button variant="ghost" size="icon" onClick={() => setEditingChunk(row.original)}>
            <Edit3 className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setDeleteChunkId(row.original.id)}>
            <Trash2 className="h-4 w-4 text-destructive" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Chunks ({chunkData?.total ?? 0})</CardTitle>
        <Button size="sm" onClick={() => setShowAddDialog(true)}>
          <Plus className="h-4 w-4 mr-1" />
          Add Chunk
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : (
          <DataTable
            columns={columns}
            data={chunkData?.chunks || []}
            searchKey="content"
            searchPlaceholder="Search chunks..."
          />
        )}
      </CardContent>

      {editingChunk && (
        <EditChunkDialog
          documentId={documentId}
          chunk={editingChunk}
          open={!!editingChunk}
          onClose={() => setEditingChunk(null)}
        />
      )}

      {showAddDialog && (
        <AddChunkDialog
          documentId={documentId}
          open={showAddDialog}
          onClose={() => setShowAddDialog(false)}
        />
      )}

      <AlertDialog open={deleteChunkId !== null} onOpenChange={() => setDeleteChunkId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Chunk</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this chunk from both Postgres and Qdrant.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteChunk}
              className="bg-destructive text-destructive-foreground"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}

function EditChunkDialog({
  documentId,
  chunk,
  open,
  onClose,
}: {
  documentId: number;
  chunk: ChunkResponse;
  open: boolean;
  onClose: () => void;
}) {
  const [content, setContent] = useState(chunk.content);
  const updateMut = useUpdateChunk();

  const handleSave = async () => {
    try {
      const result = await updateMut.mutateAsync({
        documentId,
        chunkId: chunk.id,
        data: { content },
      });
      if (result.warning) {
        toast(result.warning, { icon: "⚠️" });
      } else {
        toast.success("Chunk updated");
      }
      onClose();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to update chunk";
      toast.error(msg);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Edit Chunk #{chunk.chunk_index}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Content</Label>
            <ScrollArea className="h-64 mt-2">
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                className="min-h-[200px] font-mono text-sm"
              />
            </ScrollArea>
            <p className="text-xs text-muted-foreground mt-1">{content.length} characters</p>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose}>
              <X className="h-4 w-4 mr-1" />
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={updateMut.isPending}>
              {updateMut.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-1" />
              )}
              Save Changes
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function AddChunkDialog({
  documentId,
  open,
  onClose,
}: {
  documentId: number;
  open: boolean;
  onClose: () => void;
}) {
  const [content, setContent] = useState("");
  const [page, setPage] = useState("");
  const [section, setSection] = useState("");
  const addMut = useAddChunk();

  const handleAdd = async () => {
    if (!content.trim()) {
      toast.error("Content is required");
      return;
    }
    try {
      const result = await addMut.mutateAsync({
        documentId,
        data: {
          content,
          page: page ? Number(page) : undefined,
          section: section || undefined,
        },
      });
      if (result.warning) {
        toast(result.warning, { icon: "⚠️" });
      } else {
        toast.success("Chunk added");
      }
      onClose();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to add chunk";
      toast.error(msg);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Add New Chunk</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Content *</Label>
            <ScrollArea className="h-64 mt-2">
              <Textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Enter chunk text content..."
                className="min-h-[200px] font-mono text-sm"
              />
            </ScrollArea>
            <p className="text-xs text-muted-foreground mt-1">{content.length} characters</p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Page Number (optional)</Label>
              <Input
                type="number"
                value={page}
                onChange={(e) => setPage(e.target.value)}
                placeholder="e.g. 5"
                className="mt-1"
              />
            </div>
            <div>
              <Label>Section (optional)</Label>
              <Input
                value={section}
                onChange={(e) => setSection(e.target.value)}
                placeholder="e.g. Chapter 1"
                className="mt-1"
              />
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose}>
              <X className="h-4 w-4 mr-1" />
              Cancel
            </Button>
            <Button onClick={handleAdd} disabled={addMut.isPending}>
              {addMut.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Plus className="h-4 w-4 mr-1" />
              )}
              Add Chunk
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ManualDocumentDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [title, setTitle] = useState("");
  const [visibility, setVisibility] = useState("internal_private");
  const createMut = useCreateManualDocument();

  const handleCreate = async () => {
    if (!title.trim()) {
      toast.error("Title is required");
      return;
    }
    try {
      await createMut.mutateAsync({ title, visibility });
      toast.success("Manual document created");
      onClose();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to create document";
      toast.error(msg);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create Manual Document</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Title *</Label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Notes: Return Policy"
              className="mt-1"
            />
          </div>
          <div>
            <Label>Visibility</Label>
            <select
              value={visibility}
              onChange={(e) => setVisibility(e.target.value)}
              className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
            >
              <option value="internal_public">Public (Internal)</option>
              <option value="internal_private">Private</option>
              <option value="internal_group">Group</option>
            </select>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={createMut.isPending}>
              {createMut.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Plus className="h-4 w-4 mr-1" />
              )}
              Create
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
