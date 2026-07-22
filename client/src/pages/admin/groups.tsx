"use client";
import { useState } from "react";
import { useGroups, useCreateGroup, useGroupMembers, useAddGroupMember, useRemoveGroupMember, useUsers } from "@/shared/api/hooks";
import { DataTable } from "@/shared/ui/data-table";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Plus, Trash2, Users } from "lucide-react";
import { type ColumnDef } from "@tanstack/react-table";
import toast from "react-hot-toast";
import type { GroupResponse, GroupMemberResponse } from "@/shared/api/types";

export function AdminGroupsPage() {
  const { data: groups } = useGroups();
  const { data: users } = useUsers();
  const createMut = useCreateGroup();
  const addMut = useAddGroupMember();
  const removeMut = useRemoveGroupMember();
  const [createOpen, setCreateOpen] = useState(false);
  const [selGroup, setSelGroup] = useState<number | null>(null);
  const [gName, setGName] = useState("");
  const [addUserId, setAddUserId] = useState("");
  const { data: members } = useGroupMembers(selGroup ?? 0);

  const handleCreate = async () => {
    if (!gName.trim()) return;
    try { await createMut.mutateAsync({ name: gName }); toast.success("Created"); setCreateOpen(false); setGName(""); } catch { toast.error("Failed"); }
  };

  const handleAdd = async () => {
    if (!selGroup || !addUserId) return;
    try { await addMut.mutateAsync({ groupId: selGroup, userId: Number(addUserId) }); toast.success("Added"); setAddUserId(""); } catch { toast.error("Failed"); }
  };

  const handleRemove = async (uid: number) => {
    if (!selGroup) return;
    try { await removeMut.mutateAsync({ groupId: selGroup, userId: uid }); toast.success("Removed"); } catch { toast.error("Failed"); }
  };

  const internalUsers = users?.filter((u) => u.kind === "internal") || [];
  const columns: ColumnDef<GroupResponse>[] = [
    { accessorKey: "id", header: "ID" },
    { accessorKey: "name", header: "Name", cell: ({ row }) => <span className="font-medium">{row.original.name}</span> },
    { id: "actions", header: "", cell: ({ row }) => <Button variant="ghost" size="sm" onClick={() => setSelGroup(row.original.id)}><Users className="h-4 w-4 mr-1" />Members</Button> },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold">Groups</h1><p className="text-muted-foreground">Manage user groups</p></div>
        <Button onClick={() => setCreateOpen(true)}><Plus className="h-4 w-4 mr-2" />Create Group</Button>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card><CardHeader><CardTitle>Groups</CardTitle></CardHeader><CardContent><DataTable columns={columns} data={groups || []} searchKey="name" searchPlaceholder="Search groups..." /></CardContent></Card>
        <Card>
          <CardHeader><CardTitle>{selGroup ? `Members (Group #${selGroup})` : "Select a group"}</CardTitle></CardHeader>
          <CardContent>
            {selGroup ? (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <Select value={addUserId} onValueChange={setAddUserId}><SelectTrigger className="flex-1"><SelectValue placeholder="Select user" /></SelectTrigger><SelectContent>{internalUsers.map((u) => <SelectItem key={u.id} value={String(u.id)}>{u.email}</SelectItem>)}</SelectContent></Select>
                  <Button onClick={handleAdd} disabled={!addUserId}>Add</Button>
                </div>
                {members && members.length > 0 ? (
                  <div className="space-y-2">{members.map((m: GroupMemberResponse) => (
                    <div key={m.id} className="flex items-center justify-between rounded-md border p-2">
                      <span className="text-sm">{m.email}</span>
                      <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => handleRemove(m.id)}><Trash2 className="h-3 w-3 text-destructive" /></Button>
                    </div>
                  ))}</div>
                ) : <p className="text-sm text-muted-foreground">No members</p>}
              </div>
            ) : <p className="text-sm text-muted-foreground">Select a group to manage members</p>}
          </CardContent>
        </Card>
      </div>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Create Group</DialogTitle><DialogDescription>Add a new user group</DialogDescription></DialogHeader>
          <div className="space-y-2"><Label>Group Name</Label><Input value={gName} onChange={(e) => setGName(e.target.value)} placeholder="e.g., Marketing Team" /></div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!gName.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
