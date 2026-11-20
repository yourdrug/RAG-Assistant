"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { RefreshCw, RotateCcw, Upload } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  defaultSpaceId,
  useDocumentDomains,
  useGroups,
  useIngestAll,
  useIngestFile,
  useIngestRegistry,
  useSpaces,
  useUploadableClients,
} from "@/shared/api/hooks";
import type { IngestRegistryItem } from "@/shared/api/types";
import { formatDateTime } from "@/shared/lib/date-time";
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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";

export function AdminIngestPage() {
  const registryQuery = useIngestRegistry();
  const registry = registryQuery.data;
  const allMut = useIngestAll();
  const fileMut = useIngestFile();
  const groupsQuery = useGroups();
  const clientsQuery = useUploadableClients();
  const { data: groups } = groupsQuery;
  const { data: uploadableClients } = clientsQuery;
  const [docsDir, setDocsDir] = useState("docs/");
  const [filePath, setFilePath] = useState("");
  const [domain, setDomain] = useState("auto");
  const [visibility, setVisibility] = useState("internal_public");
  const [groupId, setGroupId] = useState<number | null>(null);
  const [clientId, setClientId] = useState<number | null>(null);
  const spacesQuery = useSpaces();
  const { data: spaces } = spacesQuery;
  const [spaceChoice, setSpaceChoice] = useState<number | null>(null);
  // New documents go to one knowledge space; profiles come from its domain pack.
  const spaceId = spaceChoice ?? defaultSpaceId(spaces);
  const domainsQuery = useDocumentDomains(spaceId, spaceId != null);
  const { data: documentDomains } = domainsQuery;

  const [resetAll, setResetAll] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  const handleAll = async () => {
    if (resetAll) {
      setShowResetConfirm(true);
      return;
    }
    try {
      await allMut.mutateAsync({
        docsDir,
        reset: false,
        domain,
        visibility,
        groupId,
        clientId,
        spaceId,
      });
      toast.success("Ingestion started");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not schedule directory indexing."));
    }
  };

  const confirmResetIngest = async () => {
    setShowResetConfirm(false);
    try {
      await allMut.mutateAsync({
        docsDir,
        reset: true,
        domain,
        visibility,
        groupId,
        clientId,
        spaceId,
      });
      toast.success("Ingestion started (reset mode)");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not schedule a full reindex."));
    }
  };

  const handleFile = async () => {
    if (!filePath.trim()) return;
    try {
      await fileMut.mutateAsync({
        filePath,
        force: false,
        domain,
        visibility,
        groupId,
        clientId,
        spaceId,
      });
      toast.success("File ingestion started");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not schedule file indexing."));
    }
  };

  const columns: ColumnDef<IngestRegistryItem>[] = [
    {
      accessorKey: "ingestion_key",
      header: "Source key",
      cell: ({ row }) => (
        <span className="font-medium truncate max-w-xs block">{row.original.ingestion_key}</span>
      ),
    },
    { accessorKey: "chunks", header: "Chunks" },
    {
      accessorKey: "chars",
      header: "Chars",
      cell: ({ row }) => row.original.chars.toLocaleString(),
    },
    {
      accessorKey: "source",
      header: "Source",
      cell: ({ row }) => <Badge variant="outline">{row.original.source}</Badge>,
    },
    {
      accessorKey: "indexed_at",
      header: "Indexed At",
      cell: ({ row }) => formatDateTime(row.original.indexed_at),
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Ingestion</h1>
          <p className="text-muted-foreground">Manage document ingestion</p>
        </div>
        <Button variant="outline" onClick={() => void registryQuery.refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {spacesQuery.isError && (
        <QueryError query={spacesQuery} fallback="Could not load knowledge spaces." />
      )}
      {domainsQuery.isError && (
        <QueryError query={domainsQuery} fallback="Could not load document types." />
      )}
      {visibility === "internal_group" && groupsQuery.isError && (
        <QueryError query={groupsQuery} fallback="Could not load groups." />
      )}
      {visibility === "client_private" && clientsQuery.isError && (
        <QueryError query={clientsQuery} fallback="Could not load clients." />
      )}
      <Card>
        <CardHeader>
          <CardTitle>Ingest Directory</CardTitle>
          <CardDescription>Trigger ingestion for all files under an S3 prefix</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="flex-1">
              <Label>S3 Prefix</Label>
              <Input
                value={docsDir}
                onChange={(e) => setDocsDir(e.target.value)}
                placeholder="docs/"
              />
            </div>
            <div className="w-40">
              <Label>Domain</Label>
              <Select value={domain} onValueChange={setDomain}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  {documentDomains?.map((d) => (
                    <SelectItem key={d.key} value={d.key}>
                      {d.display_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              onClick={handleAll}
              disabled={allMut.isPending || spaceId == null}
              className="mt-6"
            >
              <Upload className="h-4 w-4 mr-2" />
              {allMut.isPending ? "Starting..." : "Ingest All"}
            </Button>
          </div>
          <div className="w-72">
            <Label>Knowledge Space</Label>
            <Select
              value={spaceId != null ? String(spaceId) : ""}
              onValueChange={(v) => {
                setSpaceChoice(Number(v));
                setDomain("auto");
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a space" />
              </SelectTrigger>
              <SelectContent>
                {spaces?.map((s) => (
                  <SelectItem key={s.id} value={String(s.id)}>
                    {s.name} ({s.pack_key})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex gap-2">
            <div className="w-48">
              <Label>Visibility</Label>
              <Select
                value={visibility}
                onValueChange={(v) => {
                  setVisibility(v);
                  if (v !== "internal_group") setGroupId(null);
                  if (v !== "client_private") setClientId(null);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="internal_public">Public</SelectItem>
                  <SelectItem value="internal_group">Group</SelectItem>
                  <SelectItem value="client_private">Client Private</SelectItem>
                  <SelectItem value="client_public">Client Public</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {visibility === "internal_group" && (
              <div className="w-48">
                <Label>Group</Label>
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
            {visibility === "client_private" && (
              <div className="w-48">
                <Label>Client</Label>
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
                {uploadableClients?.length === 0 && (
                  <p className="text-xs text-muted-foreground">No clients assigned</p>
                )}
              </div>
            )}
          </div>
          <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={resetAll}
              onChange={(e) => setResetAll(e.target.checked)}
              className="h-4 w-4 rounded border-input accent-primary"
            />
            <RotateCcw className="h-3.5 w-3.5" />
            Reset selected prefix — preserve existing assignments
          </label>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Ingest Single File</CardTitle>
          <CardDescription>Ingest a specific file by path</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="flex-1">
              <Label>File Path (S3 key)</Label>
              <Input
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
                placeholder="docs/report.pdf"
              />
            </div>
            <div className="w-40">
              <Label>Domain</Label>
              <Select value={domain} onValueChange={setDomain}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  {documentDomains?.map((d) => (
                    <SelectItem key={d.key} value={d.key}>
                      {d.display_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              onClick={handleFile}
              disabled={!filePath.trim() || fileMut.isPending || spaceId == null}
              className="mt-6"
            >
              {fileMut.isPending ? "Starting..." : "Ingest File"}
            </Button>
          </div>
          <div className="flex gap-2">
            <div className="w-48">
              <Label>Visibility</Label>
              <Select
                value={visibility}
                onValueChange={(v) => {
                  setVisibility(v);
                  if (v !== "internal_group") setGroupId(null);
                  if (v !== "client_private") setClientId(null);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="internal_public">Public</SelectItem>
                  <SelectItem value="internal_group">Group</SelectItem>
                  <SelectItem value="client_private">Client Private</SelectItem>
                  <SelectItem value="client_public">Client Public</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {visibility === "internal_group" && (
              <div className="w-48">
                <Label>Group</Label>
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
            {visibility === "client_private" && (
              <div className="w-48">
                <Label>Client</Label>
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
                {uploadableClients?.length === 0 && (
                  <p className="text-xs text-muted-foreground">No clients assigned</p>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Ingestion Registry</CardTitle>
          {registry && (
            <CardDescription>
              {registry?.total_files || 0} files · {registry?.total_chunks?.toLocaleString() || 0}{" "}
              chunks
            </CardDescription>
          )}
        </CardHeader>
        <CardContent>
          {registryQuery.isError && (
            <QueryError query={registryQuery} fallback="Could not load the ingestion registry." />
          )}
          {registry && (
            <DataTable
              columns={columns}
              data={registry?.files || []}
              searchKey="ingestion_key"
              searchPlaceholder="Search registry..."
            />
          )}
          {registryQuery.isPending && <p role="status">Loading the ingestion registry…</p>}
        </CardContent>
      </Card>

      <AlertDialog open={showResetConfirm} onOpenChange={setShowResetConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Reset Ingestion</AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <p>
                Re-index files in <strong>{docsDir}</strong> and its subdirectories, preserving
                existing document IDs and access assignments. Files removed from this prefix will be
                removed from the index.
              </p>
              <p>
                Other prefixes, user uploads, and manual documents will be preserved. Visibility,
                client, and group options apply only to new documents.
              </p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmResetIngest}>Reset and Re-index</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
