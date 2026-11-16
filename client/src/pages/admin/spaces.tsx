"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { isAxiosError } from "axios";
import { Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  useCreateSpace,
  useDeleteSpace,
  useDomainPacks,
  useGroups,
  useSpaces,
  useUpdateSpace,
} from "@/shared/api/hooks";
import type { GroupResponse, KnowledgeSpace } from "@/shared/api/types";
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
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";

interface SpaceForm {
  id: number | null;
  name: string;
  packKey: string;
  groupIds: number[];
}

const EMPTY_FORM: SpaceForm = { id: null, name: "", packKey: "", groupIds: [] };

function errorMessage(error: unknown, fallback: string): string {
  if (isAxiosError(error)) {
    const data = error.response?.data as { message?: string; detail?: string } | undefined;
    return data?.message || data?.detail || fallback;
  }
  return fallback;
}

function GroupPicker({
  groups,
  selected,
  onChange,
}: {
  groups: GroupResponse[];
  selected: number[];
  onChange: (ids: number[]) => void;
}) {
  if (groups.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">No groups yet: only administrators fill it</p>
    );
  }
  return (
    <div className="max-h-48 space-y-1 overflow-y-auto rounded-md border p-2">
      {groups.map((g) => (
        <label key={g.id} className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={selected.includes(g.id)}
            onChange={(e) =>
              onChange(
                e.target.checked ? [...selected, g.id] : selected.filter((id) => id !== g.id),
              )
            }
          />
          {g.name}
        </label>
      ))}
    </div>
  );
}

export function AdminSpacesPage() {
  const { data: spaces } = useSpaces();
  const { data: groups } = useGroups();
  const { data: packKeys } = useDomainPacks();
  const createMut = useCreateSpace();
  const updateMut = useUpdateSpace();
  const deleteMut = useDeleteSpace();
  const [form, setForm] = useState<SpaceForm | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeSpace | null>(null);
  const groupNames = new Map((groups || []).map((g) => [g.id, g.name]));
  const isEdit = form?.id != null;

  const handleSave = async () => {
    if (!form || !form.name.trim()) return;
    try {
      if (form.id != null) {
        await updateMut.mutateAsync({
          id: form.id,
          name: form.name.trim(),
          upload_group_ids: form.groupIds,
        });
        toast.success("Space updated");
      } else {
        await createMut.mutateAsync({
          name: form.name.trim(),
          pack_key: form.packKey,
          upload_group_ids: form.groupIds,
        });
        toast.success("Space created");
      }
      setForm(null);
    } catch (error) {
      toast.error(errorMessage(error, "Failed to save the space"));
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteMut.mutateAsync(deleteTarget.id);
      toast.success("Space deleted");
    } catch (error) {
      toast.error(errorMessage(error, "Failed to delete the space"));
    }
    setDeleteTarget(null);
  };

  const columns: ColumnDef<KnowledgeSpace>[] = [
    { accessorKey: "id", header: "ID", size: 60 },
    {
      accessorKey: "name",
      header: "Name",
      cell: ({ row }) => <span className="font-medium">{row.original.name}</span>,
    },
    {
      accessorKey: "pack_key",
      header: "Domain pack",
      cell: ({ row }) => <Badge variant="secondary">{row.original.pack_key}</Badge>,
    },
    {
      id: "groups",
      header: "May fill",
      cell: ({ row }) => {
        const ids = row.original.upload_group_ids ?? [];
        return ids.length === 0 ? (
          <span className="text-xs text-muted-foreground">Administrators only</span>
        ) : (
          <div className="flex flex-wrap gap-1">
            {ids.map((id) => (
              <Badge key={id} variant="outline">
                {groupNames.get(id) ?? `#${id}`}
              </Badge>
            ))}
          </div>
        );
      },
    },
    {
      id: "actions",
      header: "",
      size: 90,
      cell: ({ row }) => (
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title="Edit"
            onClick={() =>
              setForm({
                id: row.original.id,
                name: row.original.name,
                packKey: row.original.pack_key,
                groupIds: row.original.upload_group_ids ?? [],
              })
            }
          >
            <Pencil className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title="Delete"
            onClick={() => setDeleteTarget(row.original)}
          >
            <Trash2 className="h-3.5 w-3.5 text-destructive" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Knowledge Spaces</h1>
          <p className="text-muted-foreground">
            A question searches the documents of one space and is answered by its domain pack.
            Reading is still decided by visibility.
          </p>
        </div>
        <Button onClick={() => setForm({ ...EMPTY_FORM, packKey: packKeys?.[0] ?? "" })}>
          <Plus className="h-4 w-4 mr-2" />
          Create Space
        </Button>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Spaces</CardTitle>
          <CardDescription>{spaces?.length || 0} space(s)</CardDescription>
        </CardHeader>
        <CardContent>
          <DataTable
            columns={columns}
            data={spaces || []}
            searchKey="name"
            searchPlaceholder="Search spaces..."
          />
        </CardContent>
      </Card>

      <Dialog open={form !== null} onOpenChange={(open) => !open && setForm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{isEdit ? "Edit Space" : "Create Space"}</DialogTitle>
            <DialogDescription>
              {isEdit
                ? "The domain pack of a space cannot be changed."
                : "Choose a loaded domain pack; it cannot be changed later."}
            </DialogDescription>
          </DialogHeader>
          {form && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Name</Label>
                <Input
                  value={form.name}
                  maxLength={100}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="e.g., Customer support"
                />
              </div>
              <div className="space-y-2">
                <Label>Domain pack</Label>
                <Select
                  value={form.packKey}
                  onValueChange={(v) => setForm({ ...form, packKey: v })}
                  disabled={isEdit}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a pack" />
                  </SelectTrigger>
                  <SelectContent>
                    {(isEdit ? [form.packKey] : packKeys || []).map((key) => (
                      <SelectItem key={key} value={key}>
                        {key}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Groups that may fill the space</Label>
                <GroupPicker
                  groups={groups || []}
                  selected={form.groupIds}
                  onChange={(groupIds) => setForm({ ...form, groupIds })}
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
                !form?.name.trim() || !form?.packKey || createMut.isPending || updateMut.isPending
              }
            >
              {isEdit ? "Save" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteTarget !== null} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Space</AlertDialogTitle>
            <AlertDialogDescription>
              Only an empty space can be deleted; its conversations are deleted with it.
            </AlertDialogDescription>
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
    </div>
  );
}
