"use client";
import { useState } from "react";
import { useUsers, useCreateUser, useToggleUserActive } from "@/shared/api/hooks";
import { DataTable } from "@/shared/ui/data-table";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";
import { Card, CardContent } from "@/shared/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Plus, UserPlus } from "lucide-react";
import { type ColumnDef } from "@tanstack/react-table";
import toast from "react-hot-toast";
import type { UserResponse } from "@/shared/api/types";

export function AdminUsersPage() {
  const { data: users } = useUsers();
  const createMut = useCreateUser();
  const toggleMut = useToggleUserActive();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ email: "", password: "", role: "user", kind: "internal" });

  const handleCreate = async () => {
    try { await createMut.mutateAsync(form); toast.success("User created"); setOpen(false); setForm({ email: "", password: "", role: "user", kind: "internal" }); } catch { toast.error("Failed"); }
  };

  const handleToggle = async (id: number, active: boolean) => {
    try { await toggleMut.mutateAsync({ userId: id, isActive: !active }); toast.success(active ? "Deactivated" : "Activated"); } catch { toast.error("Failed"); }
  };

  const columns: ColumnDef<UserResponse>[] = [
    { accessorKey: "id", header: "ID", cell: ({ row }) => <span className="text-muted-foreground">#{row.original.id}</span> },
    { accessorKey: "email", header: "Email", cell: ({ row }) => <span className="font-medium">{row.original.email}</span> },
    { accessorKey: "role", header: "Role", cell: ({ row }) => <Badge variant={row.original.role === "admin" ? "default" : "secondary"}>{row.original.role}</Badge> },
    { accessorKey: "kind", header: "Kind", cell: ({ row }) => <Badge variant="outline">{row.original.kind}</Badge> },
    { accessorKey: "is_active", header: "Status", cell: ({ row }) => <Badge variant={row.original.is_active ? "success" : "destructive"}>{row.original.is_active ? "Active" : "Inactive"}</Badge> },
    { id: "actions", header: "", cell: ({ row }) => <Button variant="ghost" size="sm" onClick={() => handleToggle(row.original.id, row.original.is_active)}>{row.original.is_active ? "Deactivate" : "Activate"}</Button> },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold">Users</h1><p className="text-muted-foreground">Manage system users</p></div>
        <Button onClick={() => setOpen(true)}><Plus className="h-4 w-4 mr-2" />Add User</Button>
      </div>
      <Card><CardContent className="pt-6"><DataTable columns={columns} data={users || []} searchKey="email" searchPlaceholder="Search by email..." /></CardContent></Card>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2"><UserPlus className="h-5 w-5" />Create User</DialogTitle>
            <DialogDescription>Add a new user to the system</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2"><Label>Email</Label><Input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} placeholder="user@example.com" /></div>
            <div className="space-y-2"><Label>Password</Label><Input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} placeholder="••••••••" /></div>
            <div className="space-y-2"><Label>Role</Label><Select value={form.role} onValueChange={(v) => setForm({ ...form, role: v })}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="user">User</SelectItem><SelectItem value="admin">Admin</SelectItem></SelectContent></Select></div>
            <div className="space-y-2"><Label>Kind</Label><Select value={form.kind} onValueChange={(v) => setForm({ ...form, kind: v })}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="internal">Internal</SelectItem><SelectItem value="client">Client</SelectItem></SelectContent></Select></div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={!form.email || !form.password || createMut.isPending}>{createMut.isPending ? "Creating..." : "Create"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
