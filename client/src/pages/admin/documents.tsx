"use client";
import { type ColumnDef } from "@tanstack/react-table";
import {
  Edit3,
  FileText,
  Hash,
  Info,
  Loader2,
  Pencil,
  Plus,
  Puzzle,
  Save,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { Link, useSearchParams } from "react-router-dom";
import {
  useAddChunk,
  useChunks,
  useChunksCursor,
  useCreateManualDocument,
  useCurrentUser,
  useDeleteChunk,
  useDeleteDocument,
  useDocuments,
  useRenameDocument,
  useUpdateChunk,
} from "@/shared/api/hooks";
import { documentStatuses } from "@/shared/api/hooks/use-documents";
import type { ChunkResponse, DocumentResponse } from "@/shared/api/types";
import { getErrorMessage } from "@/shared/lib/errors";
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
import { QueryError } from "@/shared/ui/query-error";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Textarea } from "@/shared/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/shared/ui/tooltip";
import { ChunkDetailsSheet } from "./chunk-details-sheet";

const filteredDocumentPageSize = 20;
const chunkPageSizes = [10, 25, 50, 100];

export function AdminDocumentsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const status = documentStatuses.find((value) => value === searchParams.get("status"));
  const offset = Number(searchParams.get("offset"));
  const pageOffset = status && Number.isSafeInteger(offset) && offset > 0 ? offset : 0;
  const documentsQuery = useDocuments(status, pageOffset, filteredDocumentPageSize + 1);
  const { data: documents, isLoading: documentsLoading } = documentsQuery;
  const changeOffset = (next: number) => {
    const params = new URLSearchParams(searchParams);
    if (next > 0) params.set("offset", String(next));
    else params.delete("offset");
    setSearchParams(params);
  };
  const deleteMut = useDeleteDocument();
  const renameMut = useRenameDocument();
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [renameId, setRenameId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const initialDocId = searchParams.get("docId");
  const highlightHashes = searchParams.get("highlight")?.split(",").filter(Boolean) ?? [];
  const [selectedDocId, setSelectedDocId] = useState<number | null>(
    initialDocId ? Number(initialDocId) : null,
  );
  const [showManualDocDialog, setShowManualDocDialog] = useState(false);
  const [chunkPageSize, setChunkPageSize] = useState(10);

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
      toast.success("Deletion queued");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not delete the document."));
    }
    setDeleteId(null);
  };

  const handleRename = async () => {
    if (renameId === null || !renameValue.trim()) return;
    try {
      await renameMut.mutateAsync({ id: renameId, filename: renameValue.trim() });
      toast.success("Renamed");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not rename the document."));
    }
    setRenameId(null);
    setRenameValue("");
  };

  const columns: ColumnDef<DocumentResponse>[] = [
    {
      accessorKey: "id",
      header: "ID",
      size: 60,
      minSize: 50,
      cell: ({ row }) => <span className="text-muted-foreground">#{row.original.id}</span>,
    },
    {
      accessorKey: "filename",
      header: "Filename",
      cell: ({ row }) => (
        <div className="flex items-start gap-2">
          {row.original.source_type === "manual" ? (
            <Puzzle className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
          ) : (
            <FileText className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
          )}
          <div className="flex flex-col gap-1">
            <span className="font-medium break-words">{row.original.filename}</span>
            {deleteMut.isDeleting(row.original.id) && <Badge variant="secondary">Deleting…</Badge>}
            {row.original.has_manual_edits && (
              <Badge variant="warning" className="text-xs w-fit">
                Edited
              </Badge>
            )}
          </div>
        </div>
      ),
    },
    {
      accessorKey: "visibility",
      header: "Visibility",
      size: 120,
      minSize: 100,
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
                    Document is visible for admin review but not used in RAG search for this
                    administrator — either another admin's private document or a client document not
                    assigned to you.
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
      size: 80,
      minSize: 70,
      cell: ({ row }) => (
        <Badge variant={row.original.doc_domain === "legal" ? "default" : "secondary"}>
          {row.original.doc_domain}
        </Badge>
      ),
    },
    {
      accessorKey: "status",
      header: "Status",
      size: 75,
      minSize: 65,
      maxSize: 90,
      cell: ({ row }) => {
        const doc = row.original;
        const isIndexing = doc.status === "indexing";
        const hasOutboxInfo =
          isIndexing && ((doc.outbox_pending ?? 0) > 0 || (doc.outbox_failed ?? 0) > 0);

        const badge = (
          <Badge
            className="w-fit"
            variant={
              doc.status === "done"
                ? "success"
                : doc.status === "failed"
                  ? "destructive"
                  : isIndexing
                    ? "default"
                    : "secondary"
            }
          >
            {doc.status}
          </Badge>
        );

        if (!hasOutboxInfo) {
          return badge;
        }

        const pending = doc.outbox_pending ?? 0;
        const failed = doc.outbox_failed ?? 0;
        const details = doc.outbox_failed_details;

        const shortLabel = [pending > 0 && `${pending} в очереди`, failed > 0 && `${failed} ошибок`]
          .filter(Boolean)
          .join(", ");

        const tooltipLines: string[] = [];
        if (pending > 0) {
          tooltipLines.push(
            `Очередь индексации: ${pending} операций ожидают обработки в векторной базе`,
          );
        }
        if (failed > 0) {
          tooltipLines.push(`Ошибки: ${failed} операций завершились ошибкой и будут повторены`);
          if (details && details.length > 0) {
            for (const d of details.slice(0, 3)) {
              const opLabel = d.operation === "upsert_chunks" ? "Индексация" : "Удаление";
              const errPreview = d.last_error ? ` — ${d.last_error.slice(0, 100)}` : "";
              tooltipLines.push(
                `• ${opLabel}: попытка ${d.attempts} из ${d.max_attempts}${errPreview}`,
              );
            }
            if (details.length > 3) {
              tooltipLines.push(`• … и ещё ${details.length - 3}`);
            }
          }
        }

        return (
          <div className="flex flex-col gap-0.5">
            {badge}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-muted-foreground whitespace-nowrap cursor-help underline decoration-dotted underline-offset-2">
                    {shortLabel}
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs whitespace-pre-line">
                  {tooltipLines.join("\n")}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        );
      },
    },
    {
      accessorKey: "chunks",
      header: "Chunks",
      size: 80,
      minSize: 60,
      cell: ({ row }) => row.original.chunks ?? "—",
    },
    {
      accessorKey: "chars",
      header: "Chars",
      size: 100,
      minSize: 80,
      cell: ({ row }) => row.original.chars?.toLocaleString() ?? "—",
    },
    {
      id: "actions",
      header: "",
      size: 140,
      minSize: 120,
      cell: ({ row }) => {
        const doc = row.original;
        const canManageChunks =
          (doc.status === "done" || doc.status === "indexing") &&
          ((doc.chunks ?? 0) > 0 || doc.source_type === "manual");
        return (
          <div className="flex gap-1">
            {canManageChunks ? (
              <Button
                variant={selectedDocId === doc.id ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedDocId(selectedDocId === doc.id ? null : doc.id)}
              >
                {selectedDocId === doc.id ? "Hide" : "Chunks"}
              </Button>
            ) : (
              <div className="w-[72px]" />
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                setRenameId(doc.id);
                setRenameValue(doc.filename);
              }}
            >
              <Pencil className="h-4 w-4 text-muted-foreground" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              disabled={deleteMut.isDeleting(doc.id)}
              title={deleteMut.isDeleting(doc.id) ? "Deleting…" : "Delete"}
              onClick={() => setDeleteId(doc.id)}
            >
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
        <CardContent className="space-y-4 pt-6">
          {status && (
            <div className="flex items-center gap-3">
              <Badge variant={status === "failed" ? "destructive" : "secondary"}>{status}</Badge>
              <Button variant="ghost" size="sm" asChild>
                <Link to="/admin/documents">Clear status filter</Link>
              </Button>
            </div>
          )}
          {documentsQuery.isError && (
            <QueryError query={documentsQuery} fallback="Could not load documents." />
          )}
          {(documents || !documentsQuery.isError) && (
            <DataTable
              columns={columns}
              data={
                status ? (documents ?? []).slice(0, filteredDocumentPageSize) : (documents ?? [])
              }
              isLoading={documentsLoading}
              searchKey="filename"
              searchPlaceholder={status ? "Search this page..." : "Search documents..."}
              serverPagination={
                status
                  ? {
                      canPreviousPage: pageOffset > 0,
                      canNextPage: (documents?.length ?? 0) > filteredDocumentPageSize,
                      onPreviousPage: () => changeOffset(pageOffset - filteredDocumentPageSize),
                      onNextPage: () => changeOffset(pageOffset + filteredDocumentPageSize),
                    }
                  : undefined
              }
            />
          )}
        </CardContent>
      </Card>

      {selectedDocId && (
        <ChunkManager
          key={`${selectedDocId}:${highlightHashes.join(",")}`}
          documentId={selectedDocId}
          highlightHashes={highlightHashes}
          pageSize={chunkPageSize}
          onPageSizeChange={setChunkPageSize}
        />
      )}

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

      <Dialog open={renameId !== null} onOpenChange={() => setRenameId(null)}>
        <DialogContent className="w-full max-w-md overflow-hidden">
          <DialogHeader>
            <DialogTitle>Rename Document</DialogTitle>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            placeholder="filename.pdf"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleRename();
            }}
            autoFocus
          />
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setRenameId(null)}>
              Cancel
            </Button>
            <Button onClick={handleRename} disabled={!renameValue.trim() || renameMut.isPending}>
              {renameMut.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-1" />
              )}
              Rename
            </Button>
          </div>
        </DialogContent>
      </Dialog>

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
  pageSize,
  onPageSizeChange,
}: {
  documentId: number;
  highlightHashes: string[];
  pageSize: number;
  onPageSizeChange: (size: number) => void;
}) {
  const hasHighlight = highlightHashes.length > 0;
  const useCursor = !hasHighlight;

  const [cursorHistory, setCursorHistory] = useState<string[]>([]);
  const [offset, setOffset] = useState(0);

  const cursorQuery = useChunksCursor(documentId, {
    cursor: cursorHistory.at(-1),
    limit: pageSize,
    enabled: useCursor,
  });
  const offsetQuery = useChunks(
    documentId,
    pageSize,
    offset,
    hasHighlight ? highlightHashes.join(",") : undefined,
    hasHighlight,
  );

  const { data: cursorData } = cursorQuery;
  const { data: offsetData } = offsetQuery;
  const chunksQuery = useCursor ? cursorQuery : offsetQuery;
  const chunks = useCursor ? (cursorData?.chunks ?? []) : (offsetData?.chunks ?? []);
  const total = offsetData?.total ?? 0;
  const { isLoading } = chunksQuery;

  const handleNextPage = () => {
    if (hasHighlight) {
      setOffset((value) => value + pageSize);
      return;
    }
    const nextCursor = cursorData?.next_cursor;
    if (nextCursor) setCursorHistory((history) => [...history, nextCursor]);
  };

  const handlePrevPage = () => {
    if (hasHighlight) {
      setOffset((value) => Math.max(0, value - pageSize));
      return;
    }
    setCursorHistory((history) => history.slice(0, -1));
  };

  const handlePageSizeChange = (value: string) => {
    setCursorHistory([]);
    setOffset(0);
    onPageSizeChange(Number(value));
  };

  const [editingChunk, setEditingChunk] = useState<ChunkResponse | null>(null);
  const [viewingChunk, setViewingChunk] = useState<ChunkResponse | null>(null);
  // The index-state endpoint is admin-only; other roles see Postgres metadata only.
  const { data: currentUser } = useCurrentUser();
  const isAdmin = currentUser?.role === "admin";
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [deleteChunkId, setDeleteChunkId] = useState<number | null>(null);

  const deleteMut = useDeleteChunk();

  const handleDeleteChunk = async () => {
    if (deleteChunkId === null) return;
    try {
      await deleteMut.mutateAsync({ documentId, chunkId: deleteChunkId });
      toast.success("Chunk deleted");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not delete the chunk."));
    }
    setDeleteChunkId(null);
  };

  const columns: ColumnDef<ChunkResponse>[] = [
    {
      accessorKey: "chunk_index",
      header: "#",
      cell: ({ row }) => (
        <span className="inline-flex h-7 min-w-7 items-center justify-center rounded-md bg-muted px-1.5 font-mono text-xs">
          {row.original.chunk_index}
        </span>
      ),
    },
    {
      accessorKey: "content",
      header: "Content",
      cell: ({ row }) => {
        return (
          <div className="min-w-0 space-y-2 py-1">
            {(row.original.heading || row.original.section) && (
              <p className="text-xs font-medium text-muted-foreground line-clamp-1 [overflow-wrap:anywhere]">
                {row.original.heading || row.original.section}
              </p>
            )}
            <p
              className={`text-sm leading-relaxed text-foreground/90 line-clamp-3 whitespace-normal [overflow-wrap:anywhere] ${hasHighlight ? "bg-yellow-100 dark:bg-yellow-900/30 rounded px-1 -mx-1" : ""}`}
            >
              {row.original.content
                .replace(/<br\s*\/?>/gi, " ")
                .replace(/\s+/g, " ")
                .trim()}
            </p>
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span>{row.original.content.length.toLocaleString()} chars</span>
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
      header: "ID",
      cell: ({ row }) => (
        <span className="whitespace-nowrap font-mono text-xs text-muted-foreground">
          #{row.original.id}
        </span>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => (
        <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Edit chunk"
            onClick={() => setEditingChunk(row.original)}
          >
            <Edit3 className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Delete chunk"
            onClick={() => setDeleteChunkId(row.original.id)}
          >
            <Trash2 className="h-4 w-4 text-destructive" />
          </Button>
        </div>
      ),
    },
  ];

  const titleText = useCursor
    ? `Chunks (${chunks.length}${cursorData?.next_cursor ? "+" : ""})`
    : `Chunks (${total})`;

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <CardTitle>{titleText}</CardTitle>
          <p className="text-sm text-muted-foreground">
            Select a chunk to read its full text and metadata.
          </p>
        </div>
        <Button size="sm" onClick={() => setShowAddDialog(true)}>
          <Plus className="h-4 w-4 mr-1" />
          Add Chunk
        </Button>
      </CardHeader>
      <CardContent className="px-3 sm:px-6 [&_table]:table-fixed [&_th]:bg-muted/40 [&_th]:px-3 [&_td]:px-3 [&_td]:py-3 [&_th[data-column=chunk\_index]]:w-16 [&_th[data-column=id]]:w-20 [&_th[data-column=actions]]:w-26 [&_[data-column=id]]:hidden sm:[&_[data-column=id]]:table-cell">
        {chunksQuery.isError && (
          <QueryError query={chunksQuery} fallback="Could not load document chunks." />
        )}
        {chunksQuery.isError && !chunksQuery.data ? (
          (cursorHistory.length > 0 || offset > 0) && (
            <Button variant="outline" onClick={handlePrevPage}>
              Previous page
            </Button>
          )
        ) : (
          <DataTable
            columns={columns}
            data={chunks}
            isLoading={isLoading}
            searchKey="content"
            searchPlaceholder="Search chunks..."
            topPagination={
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Chunks per page</span>
                <Select value={String(pageSize)} onValueChange={handlePageSizeChange}>
                  <SelectTrigger aria-label="Chunks per page" className="h-8 w-20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {chunkPageSizes.map((size) => (
                      <SelectItem key={size} value={String(size)}>
                        {size}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            }
            serverPagination={{
              canPreviousPage: useCursor ? cursorHistory.length > 0 : offset > 0,
              canNextPage: useCursor ? !!cursorData?.next_cursor : offset + pageSize < total,
              onPreviousPage: handlePrevPage,
              onNextPage: handleNextPage,
            }}
            onRowClick={setViewingChunk}
          />
        )}
      </CardContent>

      <ChunkDetailsSheet
        documentId={documentId}
        chunk={viewingChunk}
        showIndexState={isAdmin}
        onClose={() => setViewingChunk(null)}
      />

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
      const msg = getErrorMessage(e, "Failed to update chunk");
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
      const msg = getErrorMessage(e, "Failed to add chunk");
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
  const [chunks, setChunks] = useState<{ content: string; page: string; section: string }[]>([]);
  const [chunkContent, setChunkContent] = useState("");
  const [chunkPage, setChunkPage] = useState("");
  const [chunkSection, setChunkSection] = useState("");
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editContent, setEditContent] = useState("");
  const [editPage, setEditPage] = useState("");
  const [editSection, setEditSection] = useState("");
  const createMut = useCreateManualDocument();
  const addChunkMut = useAddChunk();

  const handleAddChunk = () => {
    if (!chunkContent.trim()) {
      toast.error("Chunk content is required");
      return;
    }
    setChunks([...chunks, { content: chunkContent, page: chunkPage, section: chunkSection }]);
    setChunkContent("");
    setChunkPage("");
    setChunkSection("");
  };

  const handleRemoveChunk = (index: number) => {
    setChunks(chunks.filter((_, i) => i !== index));
    if (editingIndex === index) setEditingIndex(null);
  };

  const handleEditChunk = (index: number) => {
    setEditingIndex(index);
    setEditContent(chunks[index].content);
    setEditPage(chunks[index].page);
    setEditSection(chunks[index].section);
  };

  const handleSaveEdit = () => {
    if (editingIndex === null) return;
    if (!editContent.trim()) {
      toast.error("Chunk content is required");
      return;
    }
    const updated = [...chunks];
    updated[editingIndex] = { content: editContent, page: editPage, section: editSection };
    setChunks(updated);
    setEditingIndex(null);
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
  };

  const handleCreate = async () => {
    if (!title.trim()) {
      toast.error("Title is required");
      return;
    }
    const pendingChunks = [...chunks];
    if (chunkContent.trim()) {
      pendingChunks.push({ content: chunkContent, page: chunkPage, section: chunkSection });
    }
    if (pendingChunks.length === 0) {
      toast.error("Add at least one chunk");
      return;
    }
    try {
      const doc = await createMut.mutateAsync({ title, visibility });
      let added = 0;
      for (const chunk of pendingChunks) {
        try {
          await addChunkMut.mutateAsync({
            documentId: doc.id,
            data: {
              content: chunk.content,
              page: chunk.page ? Number(chunk.page) : undefined,
              section: chunk.section || undefined,
            },
          });
          added++;
        } catch (e: unknown) {
          const msg = getErrorMessage(e, "Failed to add chunk");
          toast.error(`Chunk #${added + 1}: ${msg}`);
        }
      }
      toast.success(
        `Document created with ${added}/${pendingChunks.length} chunk${pendingChunks.length > 1 ? "s" : ""}`,
      );
      onClose();
    } catch (e: unknown) {
      const msg = getErrorMessage(e, "Failed to create document");
      toast.error(msg);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="w-[700px] max-w-none max-h-[85vh] overflow-hidden [&>div]:min-w-0">
        <div className="min-w-0 overflow-hidden space-y-4">
          <DialogHeader>
            <DialogTitle className="px-1">Create Manual Document</DialogTitle>
          </DialogHeader>
          <div className="max-h-[50vh] overflow-auto">
            <div className="space-y-4 px-1 min-w-0 overflow-hidden">
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
                <Select value={visibility} onValueChange={setVisibility}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="Select visibility" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="internal_public">Public (Internal)</SelectItem>
                    <SelectItem value="internal_private">Private</SelectItem>
                    <SelectItem value="internal_group">Group</SelectItem>
                    <SelectItem value="client_private">Client Private</SelectItem>
                    <SelectItem value="client_public">Client Public</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-sm font-medium">
                  Chunks {chunks.length > 0 && <Badge variant="secondary">{chunks.length}</Badge>}
                </Label>
                <p className="text-xs text-muted-foreground mb-3">
                  Add content chunks to this document. You can add more later.
                </p>

                {chunks.length > 0 && (
                  <div className="space-y-2 mb-3">
                    {chunks.map((chunk, i) => (
                      <div
                        key={i}
                        className="group relative rounded-lg border bg-card hover:bg-accent/30 transition-colors"
                      >
                        {editingIndex === i ? (
                          <div className="p-3 space-y-3">
                            <div className="space-y-1.5">
                              <Label className="text-[11px] text-muted-foreground">Content</Label>
                              <Textarea
                                value={editContent}
                                onChange={(e) => setEditContent(e.target.value)}
                                className="min-h-[100px] font-mono text-sm resize-none"
                                autoFocus
                              />
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="space-y-1.5">
                                <Label className="text-[11px] text-muted-foreground">Page</Label>
                                <Input
                                  type="number"
                                  value={editPage}
                                  onChange={(e) => setEditPage(e.target.value)}
                                  placeholder="e.g. 5"
                                  className="h-8 text-xs"
                                />
                              </div>
                              <div className="space-y-1.5">
                                <Label className="text-[11px] text-muted-foreground">Section</Label>
                                <Input
                                  value={editSection}
                                  onChange={(e) => setEditSection(e.target.value)}
                                  placeholder="e.g. Introduction"
                                  className="h-8 text-xs"
                                />
                              </div>
                            </div>
                            <div className="flex items-center justify-end gap-1.5">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs"
                                onClick={handleCancelEdit}
                              >
                                Cancel
                              </Button>
                              <Button
                                size="sm"
                                className="h-7 px-3 text-xs"
                                onClick={handleSaveEdit}
                              >
                                <Save className="h-3 w-3 mr-1" />
                                Save
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <button
                            type="button"
                            className="w-full text-left p-3 flex items-start gap-3"
                            onClick={() => handleEditChunk(i)}
                          >
                            <span className="inline-flex items-center justify-center h-5 min-w-[20px] rounded bg-primary/10 text-primary text-[10px] font-semibold shrink-0 mt-0.5">
                              {i + 1}
                            </span>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm leading-relaxed break-words text-foreground">
                                {chunk.content}
                              </p>
                              {(chunk.page || chunk.section) && (
                                <div className="flex items-center gap-1.5 mt-1.5 text-[11px] text-muted-foreground">
                                  {chunk.page && (
                                    <span className="inline-flex items-center gap-0.5 rounded bg-muted px-1.5 py-0.5">
                                      <Hash className="h-2.5 w-2.5" />
                                      p. {chunk.page}
                                    </span>
                                  )}
                                  {chunk.section && (
                                    <span className="inline-flex items-center gap-0.5 rounded bg-muted px-1.5 py-0.5">
                                      § {chunk.section}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                            <button
                              type="button"
                              className="opacity-0 group-hover:opacity-100 absolute right-2 top-2 p-1 rounded-md hover:bg-destructive/10 hover:text-destructive text-muted-foreground transition-all"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRemoveChunk(i);
                              }}
                            >
                              <X className="h-3.5 w-3.5" />
                            </button>
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                <div className="space-y-2">
                  <Textarea
                    value={chunkContent}
                    onChange={(e) => setChunkContent(e.target.value)}
                    placeholder="Enter chunk text content..."
                    className="min-h-[80px] font-mono text-sm"
                  />
                  <div className="grid grid-cols-2 gap-2">
                    <Input
                      type="number"
                      value={chunkPage}
                      onChange={(e) => setChunkPage(e.target.value)}
                      placeholder="Page (optional)"
                    />
                    <Input
                      value={chunkSection}
                      onChange={(e) => setChunkSection(e.target.value)}
                      placeholder="Section (optional)"
                    />
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleAddChunk}
                    disabled={!chunkContent.trim()}
                  >
                    <Plus className="h-3 w-3 mr-1" />
                    Add Chunk
                  </Button>
                </div>
              </div>
            </div>
          </div>
          <div className="flex justify-end gap-2 pt-2 px-1 border-t">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={createMut.isPending || addChunkMut.isPending}>
              {createMut.isPending || addChunkMut.isPending ? (
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
