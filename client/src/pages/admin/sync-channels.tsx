"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { FolderInput, History, Pencil, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  useAdoptIngested,
  useCreateSyncChannel,
  useCurrentUser,
  useDeleteSyncChannel,
  useSpaces,
  useStartSyncRun,
  useSyncChannels,
  useSyncRuns,
  useSyncStorageKinds,
  useUpdateSyncChannel,
} from "@/shared/api/hooks";
import type {
  DocumentVisibility,
  SyncAdoptionReport,
  SyncChannel,
  SyncChannelSettings,
  SyncDeletionMode,
  SyncRun,
  SyncStorageKind,
} from "@/shared/api/types";
import { getErrorMessage } from "@/shared/lib/errors";
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
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Textarea } from "@/shared/ui/textarea";

const VISIBILITIES: DocumentVisibility[] = [
  "internal_public",
  "internal_group",
  "internal_private",
  "client_private",
  "client_public",
];

const RUN_STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  succeeded: "default",
  partial: "secondary",
  queued: "outline",
  running: "outline",
  cancelled: "outline",
  failed: "destructive",
  rejected: "destructive",
};

interface ChannelForm {
  id: number | null;
  spaceId: number | null;
  storage_kind: SyncStorageKind;
  repository_id: number;
  repository: string;
  branch: string;
  folder: string;
  exclusions: string[];
  visibility: DocumentVisibility;
  owner_id: number | null;
  group_id: number | null;
  responsible_user_id: number;
}

const STORAGE_LABELS: Record<SyncStorageKind, string> = {
  github: "GitHub repository",
  s3: "S3 prefix of the deployment bucket",
};

const EMPTY_FORM: ChannelForm = {
  id: null,
  spaceId: null,
  storage_kind: "s3",
  repository_id: 0,
  repository: "",
  branch: "main",
  folder: "",
  exclusions: [],
  visibility: "internal_public",
  owner_id: null,
  group_id: null,
  responsible_user_id: 0,
};

function formFor(channel: SyncChannel): ChannelForm {
  return {
    id: channel.id,
    spaceId: channel.space_id,
    storage_kind: channel.storage_kind,
    repository_id: channel.repository_id ?? 0,
    repository: channel.repository ?? "",
    branch: channel.branch ?? "main",
    folder: channel.folder,
    exclusions: channel.exclusions,
    visibility: channel.visibility,
    owner_id: channel.owner_id,
    group_id: channel.group_id,
    responsible_user_id: channel.responsible_user_id ?? 0,
  };
}

function settingsOf(form: ChannelForm): SyncChannelSettings {
  const github = form.storage_kind === "github";
  return {
    storage_kind: form.storage_kind,
    repository_id: github ? form.repository_id : null,
    repository: github ? form.repository.trim() : null,
    branch: github ? form.branch.trim() : null,
    folder: form.folder.trim(),
    exclusions: form.exclusions.map((line) => line.trim()).filter(Boolean),
    visibility: form.visibility,
    owner_id: form.owner_id,
    group_id: form.group_id,
    responsible_user_id: form.responsible_user_id,
  };
}

/** The location as the table shows it; the search matches this text. */
function locationLabel(channel: SyncChannel): string {
  const where =
    channel.storage_kind === "s3"
      ? `s3://${channel.bucket ?? ""}`
      : `${channel.repository ?? ""} ${channel.branch ?? ""}`;
  return `${where} ${channel.folder || "/"}`;
}

function numberOrNull(value: string): number | null {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function RunSummary({ run }: { run: SyncRun | null }) {
  if (!run) return <span className="text-xs text-muted-foreground">Never synced</span>;
  const counts = Object.entries(run.counts)
    .filter(([kind]) => kind !== "unchanged")
    .map(([kind, count]) => `${kind} ${count}`)
    .join(", ");
  return (
    <div className="space-y-1">
      <Badge variant={RUN_STATUS_VARIANT[run.status] ?? "outline"}>{run.status}</Badge>
      {run.storage_revision && (
        <div className="font-mono text-xs text-muted-foreground">
          {run.storage_revision.slice(0, 10)}
        </div>
      )}
      {(counts || run.error) && (
        <div
          className="max-w-xs truncate text-xs text-muted-foreground"
          title={run.error ?? counts}
        >
          {run.error ?? counts}
        </div>
      )}
    </div>
  );
}

function AdoptionResult({
  report,
  onClose,
}: {
  report: SyncAdoptionReport | null;
  onClose: () => void;
}) {
  return (
    <Dialog open={report !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Ingested documents moved</DialogTitle>
          <DialogDescription>
            {report?.adopted.length ?? 0} document(s) are now managed by the channel. The others
            stay manual documents.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[60vh] space-y-3 overflow-y-auto text-sm">
          {(report?.different_access.length ?? 0) > 0 && (
            <div>
              <div className="font-medium">Different access than the channel</div>
              <ul className="mt-1 space-y-1 text-xs">
                {report?.different_access.map((path) => (
                  <li key={path} className="font-mono">
                    {path}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {(report?.skipped.length ?? 0) > 0 && (
            <div>
              <div className="font-medium">Skipped</div>
              <ul className="mt-1 space-y-1 text-xs">
                {report?.skipped.map((item) => (
                  <li key={item.path}>
                    <span className="font-mono">{item.path}</span> — {item.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function RunHistory({ channelId, onClose }: { channelId: number | null; onClose: () => void }) {
  const runsQuery = useSyncRuns(channelId);
  return (
    <Dialog open={channelId !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Sync runs</DialogTitle>
          <DialogDescription>
            Newest first; failed files and conflicts are listed.
          </DialogDescription>
        </DialogHeader>
        {runsQuery.isError && <QueryError query={runsQuery} fallback="Could not load runs." />}
        <div className="max-h-[60vh] space-y-3 overflow-y-auto">
          {(runsQuery.data ?? []).map((run) => (
            <div key={run.id} className="rounded-md border p-3 text-sm">
              <div className="flex items-center justify-between gap-2">
                <RunSummary run={run} />
                <span className="text-xs text-muted-foreground">
                  {run.trigger} · {new Date(run.creation_date).toLocaleString()}
                </span>
              </div>
              {run.problems.length > 0 && (
                <ul className="mt-2 space-y-1 text-xs">
                  {run.problems.map((problem) => (
                    <li key={`${problem.kind}:${problem.path}`}>
                      <span className="font-mono">{problem.path}</span> — {problem.kind}
                      {problem.detail ? `: ${problem.detail}` : ""}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function AdminSyncChannelsPage() {
  const channelsQuery = useSyncChannels();
  const { data: user } = useCurrentUser();
  const { data: spaces } = useSpaces();
  const createMut = useCreateSyncChannel();
  const updateMut = useUpdateSyncChannel();
  const deleteMut = useDeleteSyncChannel();
  const runMut = useStartSyncRun();
  const adoptMut = useAdoptIngested();
  const { data: storageKinds } = useSyncStorageKinds();
  const [form, setForm] = useState<ChannelForm | null>(null);
  const [adoption, setAdoption] = useState<SyncAdoptionReport | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<SyncChannel | null>(null);
  const [historyId, setHistoryId] = useState<number | null>(null);
  const isAdmin = user?.role === "admin";
  const spaceNames = new Map((spaces ?? []).map((space) => [space.id, space.name]));

  const handleSave = async () => {
    if (!form) return;
    try {
      if (form.id != null) {
        await updateMut.mutateAsync({ id: form.id, ...settingsOf(form) });
        toast.success("Channel updated");
      } else if (form.spaceId != null) {
        await createMut.mutateAsync({ space_id: form.spaceId, ...settingsOf(form) });
        toast.success("Channel created");
      }
      setForm(null);
    } catch (error) {
      toast.error(getErrorMessage(error, "Failed to save the channel"));
    }
  };

  const handleAdopt = async (channel: SyncChannel) => {
    try {
      const report = await adoptMut.mutateAsync(channel.id);
      const kept = report.different_access.length + report.skipped.length;
      toast.success(
        `${report.adopted.length} document(s) moved into the channel` +
          (kept ? `; ${kept} stay manual` : ""),
      );

      if (kept) setAdoption(report);
    } catch (error) {
      toast.error(getErrorMessage(error, "Failed to move ingested documents"));
    }
  };

  const handleRun = async (channel: SyncChannel) => {
    try {
      await runMut.mutateAsync(channel.id);
      toast.success("Sync queued");
    } catch (error) {
      toast.error(getErrorMessage(error, "Failed to start the sync"));
    }
  };

  const handleDelete = async (mode: SyncDeletionMode) => {
    if (!deleteTarget) return;
    try {
      await deleteMut.mutateAsync({ id: deleteTarget.id, mode });
      toast.success(mode === "unlink" ? "Channel deleted, documents kept" : "Channel deleted");
    } catch (error) {
      toast.error(getErrorMessage(error, "Failed to delete the channel"));
    }
    setDeleteTarget(null);
  };

  const columns: ColumnDef<SyncChannel>[] = [
    { accessorKey: "id", header: "ID", size: 60 },
    {
      id: "location",
      header: "Location",
      accessorFn: locationLabel,
      cell: ({ row }) => (
        <div>
          <div className="font-medium">
            {row.original.storage_kind === "s3"
              ? `s3://${row.original.bucket}`
              : row.original.repository}
          </div>
          <div className="font-mono text-xs text-muted-foreground">
            {row.original.storage_kind === "s3" ? "prefix" : row.original.branch}:
            {row.original.folder || "/"}
          </div>
        </div>
      ),
    },
    {
      id: "space",
      header: "Space",
      cell: ({ row }) => spaceNames.get(row.original.space_id) ?? `#${row.original.space_id}`,
    },
    {
      accessorKey: "visibility",
      header: "Audience",
      cell: ({ row }) => <Badge variant="secondary">{row.original.visibility}</Badge>,
    },
    {
      id: "last_run",
      header: "Last run",
      cell: ({ row }) => <RunSummary run={row.original.last_run} />,
    },
    {
      id: "actions",
      header: "",
      size: 150,
      cell: ({ row }) => (
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title="Sync now"
            onClick={() => handleRun(row.original)}
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title="Runs"
            onClick={() => setHistoryId(row.original.id)}
          >
            <History className="h-3.5 w-3.5" />
          </Button>
          {isAdmin && row.original.storage_kind === "s3" && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              title="Move documents ingested from this prefix into the channel"
              onClick={() => handleAdopt(row.original)}
            >
              <FolderInput className="h-3.5 w-3.5" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title={isAdmin ? "Edit" : "Narrow the folder or add exclusions"}
            onClick={() => setForm(formFor(row.original))}
          >
            <Pencil className="h-3.5 w-3.5" />
          </Button>
          {isAdmin && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              title="Delete"
              onClick={() => setDeleteTarget(row.original)}
            >
              <Trash2 className="h-3.5 w-3.5 text-destructive" />
            </Button>
          )}
        </div>
      ),
    },
  ];

  const isEdit = form?.id != null;
  // Responsible users may only narrow: the location and audience fields are the administrator's.
  const locked = isEdit && !isAdmin;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Sync Channels</h1>
          <p className="text-muted-foreground">
            A storage folder (an S3 prefix or a repository folder) kept in line with the documents
            of a space. Exclude files with channel patterns or a <code>.ragignore</code> file.
          </p>
        </div>
        {isAdmin && (
          <Button
            disabled={!storageKinds?.length}
            title={storageKinds?.length ? undefined : "No storage is configured for sync channels"}
            onClick={() =>
              setForm({
                ...EMPTY_FORM,
                storage_kind: storageKinds?.[0] ?? EMPTY_FORM.storage_kind,
                spaceId: spaces?.[0]?.id ?? null,
                responsible_user_id: user?.id ?? 0,
              })
            }
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Channel
          </Button>
        )}
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Channels</CardTitle>
          {channelsQuery.data && (
            <CardDescription>{channelsQuery.data.length} channel(s)</CardDescription>
          )}
        </CardHeader>
        <CardContent>
          {channelsQuery.isError && (
            <QueryError query={channelsQuery} fallback="Could not load sync channels." />
          )}
          {channelsQuery.data && (
            <DataTable
              columns={columns}
              data={channelsQuery.data}
              searchKey="location"
              searchPlaceholder="Search locations and folders..."
            />
          )}
          {channelsQuery.isPending && <p role="status">Loading sync channels…</p>}
        </CardContent>
      </Card>

      <Dialog open={form !== null} onOpenChange={(open) => !open && setForm(null)}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{isEdit ? "Edit Channel" : "Create Channel"}</DialogTitle>
            <DialogDescription>
              {locked
                ? "As the responsible user you may narrow the folder and add exclusions."
                : "Only an administrator binds a repository or widens a channel."}
            </DialogDescription>
          </DialogHeader>
          {form && (
            <div className="grid grid-cols-2 gap-3">
              {!isEdit && (
                <div className="col-span-2 space-y-1">
                  <Label>Space</Label>
                  <Select
                    value={form.spaceId ? String(form.spaceId) : ""}
                    onValueChange={(v) => setForm({ ...form, spaceId: Number(v) })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a space" />
                    </SelectTrigger>
                    <SelectContent>
                      {(spaces ?? []).map((space) => (
                        <SelectItem key={space.id} value={String(space.id)}>
                          {space.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              {!isEdit && (
                <div className="col-span-2 space-y-1">
                  <Label>Storage</Label>
                  <Select
                    value={form.storage_kind}
                    onValueChange={(v) => setForm({ ...form, storage_kind: v as SyncStorageKind })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(storageKinds ?? []).map((kind) => (
                        <SelectItem key={kind} value={kind}>
                          {STORAGE_LABELS[kind]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              {form.storage_kind === "github" && (
                <>
                  <div className="space-y-1">
                    <Label>Repository (owner/name)</Label>
                    <Input
                      value={form.repository}
                      disabled={locked}
                      onChange={(e) => setForm({ ...form, repository: e.target.value })}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Repository ID</Label>
                    <Input
                      inputMode="numeric"
                      value={form.repository_id || ""}
                      disabled={locked}
                      onChange={(e) =>
                        setForm({ ...form, repository_id: numberOrNull(e.target.value) ?? 0 })
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Branch</Label>
                    <Input
                      value={form.branch}
                      disabled={locked}
                      onChange={(e) => setForm({ ...form, branch: e.target.value })}
                    />
                  </div>
                </>
              )}
              <div className="space-y-1">
                <Label>Folder{form.storage_kind === "s3" ? " (required)" : ""}</Label>
                <Input
                  value={form.folder}
                  placeholder="docs"
                  onChange={(e) => setForm({ ...form, folder: e.target.value })}
                />
                {form.storage_kind === "s3" && (
                  <p className="text-xs text-muted-foreground">
                    The key prefix to import. The bucket also holds uploads and previews, so a
                    channel never reads the whole bucket or the uploads/ and previews/ areas.
                  </p>
                )}
              </div>
              <div className="col-span-2 space-y-1">
                <Label>Exclusions (.gitignore syntax, one per line)</Label>
                <Textarea
                  rows={3}
                  value={form.exclusions.join("\n")}
                  onChange={(e) => setForm({ ...form, exclusions: e.target.value.split("\n") })}
                />
              </div>
              <div className="space-y-1">
                <Label>Visibility</Label>
                <Select
                  value={form.visibility}
                  disabled={locked}
                  onValueChange={(v) => setForm({ ...form, visibility: v as DocumentVisibility })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VISIBILITIES.map((visibility) => (
                      <SelectItem key={visibility} value={visibility}>
                        {visibility}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label>Responsible user ID</Label>
                <Input
                  inputMode="numeric"
                  value={form.responsible_user_id || ""}
                  disabled={locked}
                  onChange={(e) =>
                    setForm({ ...form, responsible_user_id: numberOrNull(e.target.value) ?? 0 })
                  }
                />
              </div>
              <div className="space-y-1">
                <Label>Owner user ID (private/client)</Label>
                <Input
                  inputMode="numeric"
                  value={form.owner_id ?? ""}
                  disabled={locked}
                  onChange={(e) => setForm({ ...form, owner_id: numberOrNull(e.target.value) })}
                />
              </div>
              <div className="space-y-1">
                <Label>Group ID (internal_group)</Label>
                <Input
                  inputMode="numeric"
                  value={form.group_id ?? ""}
                  disabled={locked}
                  onChange={(e) => setForm({ ...form, group_id: numberOrNull(e.target.value) })}
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setForm(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              disabled={
                (form?.storage_kind === "github" &&
                  (!form?.repository.trim() || !form?.repository_id)) ||
                (form?.storage_kind === "s3" && !form?.folder.trim()) ||
                !form?.responsible_user_id ||
                (!isEdit && form?.spaceId == null) ||
                createMut.isPending ||
                updateMut.isPending
              }
            >
              {isEdit ? "Save" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteTarget !== null} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Channel</DialogTitle>
            <DialogDescription>
              Delete the channel's documents with it, or keep them as manual documents with the same
              access.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => handleDelete("unlink")}>
              Keep documents
            </Button>
            <Button variant="destructive" onClick={() => handleDelete("delete_documents")}>
              Delete documents
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <RunHistory channelId={historyId} onClose={() => setHistoryId(null)} />
      <AdoptionResult report={adoption} onClose={() => setAdoption(null)} />
    </div>
  );
}
