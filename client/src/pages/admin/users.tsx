"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { ChevronDown, Plus, UserPlus } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  useAssignClientSpace,
  useChangeUserRole,
  useCreateUser,
  useSpaces,
  useToggleUserActive,
  useUsers,
} from "@/shared/api/hooks";
import type { UserResponse } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent } from "@/shared/ui/card";
import { DataTable } from "@/shared/ui/data-table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/shared/ui/dropdown-menu";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";

const roleBadgeVariant = (role: string) => {
  switch (role) {
    case "admin":
      return "default" as const;
    case "curator":
      return "outline" as const;
    default:
      return "secondary" as const;
  }
};

export function AdminUsersPage() {
  const { data: users } = useUsers();
  const { data: spaces } = useSpaces();
  const assignSpaceMut = useAssignClientSpace();
  const createMut = useCreateUser();
  const toggleMut = useToggleUserActive();
  const changeRoleMut = useChangeUserRole();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ email: "", password: "", role: "user", kind: "internal" });
  const [roleDialogUser, setRoleDialogUser] = useState<UserResponse | null>(null);
  const [newRole, setNewRole] = useState("");

  const handleCreate = async () => {
    try {
      await createMut.mutateAsync(form);
      toast.success("User created");
      setOpen(false);
      setForm({ email: "", password: "", role: "user", kind: "internal" });
    } catch {
      toast.error("Failed");
    }
  };

  const handleToggle = async (id: number, active: boolean) => {
    try {
      await toggleMut.mutateAsync({ userId: id, isActive: !active });
      toast.success(active ? "Deactivated" : "Activated");
    } catch {
      toast.error("Failed");
    }
  };

  const handleChangeRole = async () => {
    if (!roleDialogUser) return;
    try {
      await changeRoleMut.mutateAsync({ userId: roleDialogUser.id, role: newRole });
      toast.success(`Role changed to ${newRole}`);
      setRoleDialogUser(null);
    } catch {
      toast.error("Failed to change role");
    }
  };

  const openRoleDialog = (user: UserResponse) => {
    setRoleDialogUser(user);
    setNewRole(user.role);
  };

  const columns: ColumnDef<UserResponse>[] = [
    {
      accessorKey: "id",
      header: "ID",
      cell: ({ row }) => <span className="text-muted-foreground">#{row.original.id}</span>,
    },
    {
      accessorKey: "email",
      header: "Email",
      cell: ({ row }) => <span className="font-medium">{row.original.email}</span>,
    },
    {
      accessorKey: "role",
      header: "Role",
      cell: ({ row }) => (
        <Badge variant={roleBadgeVariant(row.original.role)}>{row.original.role}</Badge>
      ),
    },
    {
      accessorKey: "kind",
      header: "Kind",
      cell: ({ row }) => <Badge variant="outline">{row.original.kind}</Badge>,
    },
    {
      id: "client_spaces",
      header: "Support spaces",
      cell: ({ row }) => {
        if (row.original.kind !== "client") {
          return <span className="text-muted-foreground">Not applicable</span>;
        }
        const assigned = row.original.client_space_ids ?? [];
        const names = (spaces ?? []).filter((s) => assigned.includes(s.id)).map((s) => s.name);
        const toggle = async (spaceId: number, checked: boolean) => {
          try {
            await assignSpaceMut.mutateAsync({
              clientId: row.original.id,
              spaceIds: checked ? [...assigned, spaceId] : assigned.filter((id) => id !== spaceId),
            });
            toast.success("Support spaces updated");
          } catch {
            toast.error("Failed to assign support spaces");
          }
        };
        return (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="w-56 justify-between font-normal"
                disabled={assignSpaceMut.isPending}
              >
                <span className="truncate">
                  {names.length === 0 ? "Unassigned" : names.join(", ")}
                </span>
                <ChevronDown className="h-3.5 w-3.5 opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              {(spaces ?? []).length === 0 && (
                <p className="px-2 py-1.5 text-xs text-muted-foreground">No spaces yet</p>
              )}
              {spaces?.map((space) => (
                <DropdownMenuCheckboxItem
                  key={space.id}
                  checked={assigned.includes(space.id)}
                  disabled={assignSpaceMut.isPending}
                  onSelect={(e) => e.preventDefault()}
                  onCheckedChange={(checked) => void toggle(space.id, checked === true)}
                >
                  {space.name}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        );
      },
    },
    {
      accessorKey: "is_active",
      header: "Status",
      cell: ({ row }) => (
        <Badge variant={row.original.is_active ? "success" : "destructive"}>
          {row.original.is_active ? "Active" : "Inactive"}
        </Badge>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => (
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => openRoleDialog(row.original)}>
            Role
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleToggle(row.original.id, row.original.is_active)}
          >
            {row.original.is_active ? "Deactivate" : "Activate"}
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Users</h1>
          <p className="text-muted-foreground">Manage system users</p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Add User
        </Button>
      </div>
      <Card>
        <CardContent className="pt-6">
          <DataTable
            columns={columns}
            data={users?.users || []}
            searchKey="email"
            searchPlaceholder="Search by email..."
          />
        </CardContent>
      </Card>

      {/* Create User Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <UserPlus className="h-5 w-5" />
              Create User
            </DialogTitle>
            <DialogDescription>Add a new user to the system</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Email</Label>
              <Input
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                placeholder="user@example.com"
              />
            </div>
            <div className="space-y-2">
              <Label>Password</Label>
              <Input
                type="password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                placeholder="••••••••"
              />
            </div>
            <div className="space-y-2">
              <Label>Kind</Label>
              <Select
                value={form.kind}
                onValueChange={(v) =>
                  setForm({ ...form, kind: v, ...(v === "client" ? { role: "user" } : {}) })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="internal">Internal</SelectItem>
                  <SelectItem value="client">Client</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Role</Label>
              <Select
                value={form.role}
                onValueChange={(v) => setForm({ ...form, role: v })}
                disabled={form.kind === "client"}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">User</SelectItem>
                  {form.kind === "internal" && <SelectItem value="curator">Curator</SelectItem>}
                  {form.kind === "internal" && <SelectItem value="admin">Admin</SelectItem>}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!form.email || !form.password || createMut.isPending}
            >
              {createMut.isPending ? "Creating..." : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Change Role Dialog */}
      <Dialog open={!!roleDialogUser} onOpenChange={() => setRoleDialogUser(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Change Role</DialogTitle>
            <DialogDescription>Change role for {roleDialogUser?.email}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={newRole} onValueChange={setNewRole}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">User</SelectItem>
                  {roleDialogUser?.kind === "internal" && (
                    <SelectItem value="curator">Curator</SelectItem>
                  )}
                  {roleDialogUser?.kind === "internal" && (
                    <SelectItem value="admin">Admin</SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRoleDialogUser(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleChangeRole}
              disabled={newRole === roleDialogUser?.role || changeRoleMut.isPending}
            >
              {changeRoleMut.isPending ? "Saving..." : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
