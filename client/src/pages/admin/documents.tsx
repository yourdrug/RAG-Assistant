"use client";
import {type ColumnDef} from "@tanstack/react-table";
import {Edit3, FileText, Hash, Info, Loader2, Pencil, Plus, Puzzle, Save, Trash2, X} from "lucide-react";
import {useEffect, useState} from "react";
import {useSearchParams} from "react-router-dom";
import toast from "react-hot-toast";
import {
    useAddChunk,
    useChunks,
    useCreateManualDocument,
    useDeleteChunk,
    useDeleteDocument,
    useDocuments,
    useRenameDocument,
    useUpdateChunk,
} from "@/shared/api/hooks";
import type {ChunkResponse, DocumentResponse} from "@/shared/api/types";
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
import {Badge} from "@/shared/ui/badge";
import {Button} from "@/shared/ui/button";
import {Card, CardContent, CardHeader, CardTitle} from "@/shared/ui/card";
import {DataTable} from "@/shared/ui/data-table";
import {Dialog, DialogContent, DialogHeader, DialogTitle} from "@/shared/ui/dialog";
import {Input} from "@/shared/ui/input";
import {Label} from "@/shared/ui/label";
import {ScrollArea} from "@/shared/ui/scroll-area";
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from "@/shared/ui/select";
import {Textarea} from "@/shared/ui/textarea";
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@/shared/ui/tooltip";

export function AdminDocumentsPage() {
    const [searchParams] = useSearchParams();
    const {data: documents} = useDocuments();
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

    const handleRename = async () => {
        if (renameId === null || !renameValue.trim()) return;
        try {
            await renameMut.mutateAsync({ id: renameId, filename: renameValue.trim() });
            toast.success("Renamed");
        } catch {
            toast.error("Failed to rename");
        }
        setRenameId(null);
        setRenameValue("");
    };

    const columns: ColumnDef<DocumentResponse>[] = [
        {
            accessorKey: "id",
            header: "ID",
            cell: ({row}) => <span className="text-muted-foreground">#{row.original.id}</span>,
        },
        {
            accessorKey: "filename",
            header: "Filename",
            cell: ({row}) => (
                <div className="flex items-center gap-2 max-w-xs">
                    {row.original.source_type === "manual" ? (
                        <Puzzle className="h-4 w-4 shrink-0 text-muted-foreground"/>
                    ) : (
                        <FileText className="h-4 w-4 shrink-0 text-muted-foreground"/>
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
            cell: ({row}) => {
                const doc = row.original;
                return (
                    <div className="flex items-center gap-1.5">
                        <Badge variant="secondary">{doc.visibility.replace(/_/g, " ")}</Badge>
                        {doc.in_search_scope === false && (
                            <TooltipProvider>
                                <Tooltip>
                                    <TooltipTrigger>
                                        <Badge variant="warning" className="text-xs gap-1">
                                            <Info className="h-3 w-3"/>
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
            cell: ({row}) => (
                <Badge variant={row.original.doc_domain === "legal" ? "default" : "secondary"}>
                    {row.original.doc_domain}
                </Badge>
            ),
        },
        {
            accessorKey: "status",
            header: "Status",
            cell: ({row}) => (
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
        {accessorKey: "chunks", header: "Chunks", cell: ({row}) => row.original.chunks ?? "—"},
        {
            accessorKey: "chars",
            header: "Chars",
            cell: ({row}) => row.original.chars?.toLocaleString() ?? "—",
        },
        {
            id: "actions",
            header: "",
            cell: ({row}) => {
                const doc = row.original;
                const canManageChunks =
                    doc.status === "done" && ((doc.chunks ?? 0) > 0 || doc.source_type === "manual");
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
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => {
                                setRenameId(doc.id);
                                setRenameValue(doc.filename);
                            }}
                        >
                            <Pencil className="h-4 w-4 text-muted-foreground"/>
                        </Button>
                        <Button variant="ghost" size="icon" onClick={() => setDeleteId(doc.id)}>
                            <Trash2 className="h-4 w-4 text-destructive"/>
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
                    <Plus className="h-4 w-4 mr-2"/>
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

            {selectedDocId && <ChunkManager documentId={selectedDocId} highlightHashes={highlightHashes}/>}

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
                                <Loader2 className="h-4 w-4 mr-1 animate-spin"/>
                            ) : (
                                <Save className="h-4 w-4 mr-1"/>
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
                      }: {
    documentId: number;
    highlightHashes?: string[];
}) {
    const hasHighlight = (highlightHashes?.length ?? 0) > 0;
    const highlightParam = hasHighlight ? highlightHashes!.join(",") : undefined;
    const {data: chunkData, isLoading} = useChunks(documentId, undefined, undefined, highlightParam);
    const [editingChunk, setEditingChunk] = useState<ChunkResponse | null>(null);
    const [showAddDialog, setShowAddDialog] = useState(false);
    const [deleteChunkId, setDeleteChunkId] = useState<number | null>(null);

    const deleteMut = useDeleteChunk();

    const handleDeleteChunk = async () => {
        if (deleteChunkId === null) return;
        try {
            await deleteMut.mutateAsync({documentId, chunkId: deleteChunkId});
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
            cell: ({row}) => (
                <span className="text-muted-foreground font-mono">{row.original.chunk_index}</span>
            ),
        },
        {
            accessorKey: "content",
            header: "Content",
            cell: ({row}) => {
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
            cell: ({row}) => (
                <span className="text-muted-foreground font-mono text-xs">#{row.original.id}</span>
            ),
        },
        {
            id: "actions",
            header: "",
            cell: ({row}) => (
                <div className="flex gap-1">
                    <Button variant="ghost" size="icon" onClick={() => setEditingChunk(row.original)}>
                        <Edit3 className="h-4 w-4"/>
                    </Button>
                    <Button variant="ghost" size="icon" onClick={() => setDeleteChunkId(row.original.id)}>
                        <Trash2 className="h-4 w-4 text-destructive"/>
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
                    <Plus className="h-4 w-4 mr-1"/>
                    Add Chunk
                </Button>
            </CardHeader>
            <CardContent>
                {isLoading ? (
                    <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin"/>
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
                data: {content},
            });
            if (result.warning) {
                toast(result.warning, {icon: "⚠️"});
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
                            <X className="h-4 w-4 mr-1"/>
                            Cancel
                        </Button>
                        <Button onClick={handleSave} disabled={updateMut.isPending}>
                            {updateMut.isPending ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin"/>
                            ) : (
                                <Save className="h-4 w-4 mr-1"/>
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
                toast(result.warning, {icon: "⚠️"});
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
                            <X className="h-4 w-4 mr-1"/>
                            Cancel
                        </Button>
                        <Button onClick={handleAdd} disabled={addMut.isPending}>
                            {addMut.isPending ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin"/>
                            ) : (
                                <Plus className="h-4 w-4 mr-1"/>
                            )}
                            Add Chunk
                        </Button>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}

function ManualDocumentDialog({open, onClose}: { open: boolean; onClose: () => void }) {
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
        setChunks([...chunks, {content: chunkContent, page: chunkPage, section: chunkSection}]);
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
        updated[editingIndex] = {content: editContent, page: editPage, section: editSection};
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
            pendingChunks.push({content: chunkContent, page: chunkPage, section: chunkSection});
        }
        if (pendingChunks.length === 0) {
            toast.error("Add at least one chunk");
            return;
        }
        try {
            const doc = await createMut.mutateAsync({title, visibility});
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
                    const msg = e instanceof Error ? e.message : "Failed to add chunk";
                    toast.error(`Chunk #${added + 1}: ${msg}`);
                }
            }
            toast.success(`Document created with ${added}/${pendingChunks.length} chunk${pendingChunks.length > 1 ? "s" : ""}`);
            onClose();
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : "Failed to create document";
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
                                        <SelectValue placeholder="Select visibility"/>
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="internal_public">Public (Internal)</SelectItem>
                                        <SelectItem value="internal_private">Private</SelectItem>
                                        <SelectItem value="internal_group">Group</SelectItem>
                                        <SelectItem value="client_private">Client Private</SelectItem>
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
                                            <div key={i}
                                                 className="group relative rounded-lg border bg-card hover:bg-accent/30 transition-colors">
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
                                                                <Save className="h-3 w-3 mr-1"/>
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
                                                                            <Hash className="h-2.5 w-2.5"/>
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
                                                            <X className="h-3.5 w-3.5"/>
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
                                        <Plus className="h-3 w-3 mr-1"/>
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
                                <Loader2 className="h-4 w-4 mr-1 animate-spin"/>
                            ) : (
                                <Plus className="h-4 w-4 mr-1"/>
                            )}
                            Create
                        </Button>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}
