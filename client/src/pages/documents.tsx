"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { CloudUpload, FileText, Pencil, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import {
  useDeleteDocument,
  useDocuments,
  useGroups,
  useRenameDocument,
  useUploadableClients,
  useUploadDocument,
} from "@/shared/api/hooks";
import type { DocumentDomain, DocumentResponse, DocumentVisibility } from "@/shared/api/types";
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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import { Progress } from "@/shared/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Input } from "@/shared/ui/input";
import { useAuthStore } from "@/stores/auth-store";

export function DocumentsPage() {
  const { data: documents } = useDocuments();
  const uploadMut = useUploadDocument();
  const deleteMut = useDeleteDocument();
  const renameMut = useRenameDocument();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [renameId, setRenameId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const user = useAuthStore((s) => s.user);
  const isClient = user?.kind === "client";
  const isAdmin = user?.role === "admin";
  const [vis, setVis] = useState<DocumentVisibility>("internal_private");
  const [groupId, setGroupId] = useState<number | null>(null);
  const [clientId, setClientId] = useState<number | null>(null);
  const [docDomain, setDocDomain] = useState<DocumentDomain | "auto">("auto");
  const [progress, setProgress] = useState(0);
  const { data: groups } = useGroups();
  const { data: uploadableClients } = useUploadableClients();

  useEffect(() => {
    if (user) {
      setVis(isClient ? "client_private" : "internal_private");
      setGroupId(null);
      setClientId(null);
    }
  }, [user?.kind, user?.role, isClient]);

  // Conflict resolution state
  const [conflictOpen, setConflictOpen] = useState(false);
  const [conflictFile, setConflictFile] = useState<File | null>(null);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);

  const existingNames = new Set((documents || []).map((d) => d.filename));

  const onDrop = useCallback((f: File[]) => {
    setFiles(f);
    setUploadOpen(true);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/markdown": [".md"],
      "text/plain": [".txt"],
    },
  });

  const uploadFile = async (file: File, renameOnConflict: boolean) => {
    await uploadMut.mutateAsync({
      file,
      visibility: vis,
      groupId: vis === "internal_group" ? groupId : undefined,
      clientId: vis === "client_private" ? clientId : undefined,
      renameOnConflict,
      docDomain: docDomain === "auto" ? undefined : docDomain,
    });
  };

  const handleUpload = async () => {
    if (!files.length) return;

    // Check for name conflicts
    const conflicts = files.filter((f) => existingNames.has(f.name));
    if (conflicts.length > 0) {
      setConflictFile(conflicts[0]);
      setPendingFiles(files);
      setUploadOpen(false);
      setConflictOpen(true);
      return;
    }

    await doUpload(files, false);
  };

  const doUpload = async (filesToUpload: File[], renameOnConflict: boolean) => {
    setUploadOpen(false);
    setConflictOpen(false);
    setProgress(0);
    let done = 0;
    for (const f of filesToUpload) {
      try {
        await uploadFile(f, renameOnConflict);
        done++;
        setProgress(Math.round((done / filesToUpload.length) * 100));
      } catch {
        toast.error(`Failed: ${f.name}`);
      }
    }
    setFiles([]);
    setProgress(0);
    if (done > 0) {
      toast.success(`${done} file(s) queued for processing`);
    }
  };

  const handleConflictChoice = async (choice: "replace" | "add_new") => {
    if (!conflictFile) return;
    setConflictOpen(false);

    if (choice === "add_new") {
      await doUpload([conflictFile], true);
    } else {
      await doUpload([conflictFile], false);
    }

    // Upload remaining files after resolving this conflict
    const remaining = pendingFiles.filter((f) => f !== conflictFile);
    if (remaining.length > 0) {
      // Re-check remaining files against current document list for new conflicts
      await doUpload(remaining, false);
    }

    setConflictFile(null);
    setPendingFiles([]);
  };

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
      accessorKey: "filename",
      header: "Filename",
      cell: ({ row }) => (
        <div className="flex items-start gap-2">
          <FileText className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
          <span className="font-medium break-words">{row.original.filename}</span>
        </div>
      ),
    },
    {
      accessorKey: "visibility",
      header: "Visibility",
      size: 120,
      minSize: 100,
      cell: ({ row }) => {
        const vis = row.original.visibility;
        const ownerId = row.original.owner_id;
        const notInSearch = ownerId != null && user != null && ownerId !== user.id;
        return (
          <div className="flex items-center gap-1.5">
            <Badge variant="secondary">{vis.replace(/_/g, " ")}</Badge>
            {notInSearch && (
              <Badge variant="outline" className="text-xs text-muted-foreground">
                not in your search
              </Badge>
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
      cell: ({ row }) => {
        const domain = row.original.doc_domain;
        return <Badge variant={domain === "legal" ? "default" : "secondary"}>{domain}</Badge>;
      },
    },
    {
      accessorKey: "owner_id",
      header: "Owner",
      size: 120,
      minSize: 100,
      cell: ({ row }) => {
        const owner = uploadableClients?.find((c) => c.id === row.original.owner_id);
        if (owner) return <span className="text-sm">{owner.email}</span>;
        if (row.original.owner_id === user?.id)
          return <span className="text-sm text-muted-foreground">You</span>;
        if (row.original.owner_id != null)
          return <span className="text-sm text-muted-foreground">#{row.original.owner_id}</span>;
        return <span className="text-muted-foreground">—</span>;
      },
    },
    {
      accessorKey: "status",
      header: "Status",
      size: 90,
      minSize: 70,
      maxSize: 120,
      cell: ({ row }) => (
        <Badge
          variant={
            row.original.status === "done"
              ? "success"
              : row.original.status === "failed"
                ? "destructive"
                : row.original.status === "processing" || row.original.status === "indexing"
                  ? "warning"
                  : "secondary"
          }
        >
          {row.original.status}
        </Badge>
      ),
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
      size: 100,
      minSize: 80,
      cell: ({ row }) => (
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => {
              setRenameId(row.original.id);
              setRenameValue(row.original.filename);
            }}
          >
            <Pencil className="h-4 w-4 text-muted-foreground" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setDeleteId(row.original.id)}
          >
            <Trash2 className="h-4 w-4 text-destructive" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Documents</h1>
        <p className="text-muted-foreground">Manage your documents</p>
      </div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-muted-foreground/50"}`}
      >
        <input {...getInputProps()} />
        <CloudUpload className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Drag & drop files here, or click to select</p>
        <p className="text-xs text-muted-foreground/70 mt-1">PDF, DOCX, MD, TXT</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>All Documents</CardTitle>
          <CardDescription>{documents?.length || 0} document(s)</CardDescription>
        </CardHeader>
        <CardContent>
          <DataTable
            columns={columns}
            data={documents || []}
            searchKey="filename"
            searchPlaceholder="Search documents..."
          />
        </CardContent>
      </Card>

      {/* Upload settings dialog */}
      <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
        <DialogContent className="overflow-hidden w-full max-w-lg">
          <DialogHeader>
            <DialogTitle>Upload Documents</DialogTitle>
            <DialogDescription>{files.length} file(s) selected</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 min-w-0">
            <div className="space-y-2">
              <label className="text-sm font-medium">Visibility</label>
              <Select
                value={vis}
                onValueChange={(v) => {
                  setVis(v as DocumentVisibility);
                  if (v !== "internal_group") setGroupId(null);
                  if (v !== "client_private") setClientId(null);
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
            {vis === "internal_group" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Group</label>
                <Select
                  value={groupId != null ? String(groupId) : ""}
                  onValueChange={(v) => setGroupId(Number(v))}
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
            {vis === "client_private" && !isClient && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Client</label>
                <Select
                  value={clientId != null ? String(clientId) : ""}
                  onValueChange={(v) => setClientId(Number(v))}
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
                value={docDomain}
                onValueChange={(v) => setDocDomain(v as DocumentDomain | "auto")}
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
              <p className="text-xs text-muted-foreground">
                Legal documents get article-aware chunking and retrieval
              </p>
            </div>
            {files.length > 0 && (
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm min-w-0 overflow-hidden">
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <span className="truncate min-w-0 flex-1">{f.name}</span>
                    <span className="shrink-0 text-muted-foreground">
                      {(f.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                ))}
              </div>
            )}
            {progress > 0 && <Progress value={progress} />}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={
                uploadMut.isPending ||
                (vis === "internal_group" && groupId == null) ||
                (vis === "client_private" && !isClient && clientId == null)
              }
            >
              {uploadMut.isPending ? "Uploading..." : "Upload"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Conflict resolution dialog */}
      <Dialog open={conflictOpen} onOpenChange={setConflictOpen}>
        <DialogContent className="w-full max-w-lg overflow-hidden [&>div]:min-w-0">
          <DialogHeader>
            <DialogTitle>File Already Exists</DialogTitle>
            <DialogDescription className="break-words">
              A document named <strong className="break-words">{conflictFile?.name}</strong> already
              exists. What would you like to do?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0 min-w-0">
            <Button
              variant="outline"
              className="min-w-0 max-w-full"
              onClick={() => handleConflictChoice("add_new")}
            >
              <span className="truncate">
                Add as New ({conflictFile?.name?.replace(/(\.[^.]+)$/, "(1)$1")})
              </span>
            </Button>
            <Button onClick={() => handleConflictChoice("replace")}>Replace Existing</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteId !== null} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent className="overflow-hidden">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Document</AlertDialogTitle>
            <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <Dialog open={renameId !== null} onOpenChange={() => setRenameId(null)}>
        <DialogContent className="overflow-hidden w-full max-w-md">
          <DialogHeader>
            <DialogTitle>Rename Document</DialogTitle>
            <DialogDescription>Enter a new filename for this document.</DialogDescription>
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
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameId(null)}>
              Cancel
            </Button>
            <Button onClick={handleRename} disabled={!renameValue.trim() || renameMut.isPending}>
              {renameMut.isPending ? "Renaming..." : "Rename"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
