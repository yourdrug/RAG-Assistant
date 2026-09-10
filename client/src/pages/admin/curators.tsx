"use client";
import { UserMinus, UserPlus } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  useAssignGroupToCurator,
  useAssignUserToCurator,
  useCuratorScope,
  useGroups,
  useUnassignGroupFromCurator,
  useUnassignUserFromCurator,
  useUsers,
} from "@/shared/api/hooks";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Separator } from "@/shared/ui/separator";

export function AdminCuratorsPage() {
  const { data: users } = useUsers();
  const { data: groups } = useGroups();
  const [selectedCuratorId, setSelectedCuratorId] = useState<number | null>(null);
  const { data: scope } = useCuratorScope(selectedCuratorId);

  const assignUserMut = useAssignUserToCurator();
  const unassignUserMut = useUnassignUserFromCurator();
  const assignGroupMut = useAssignGroupToCurator();
  const unassignGroupMut = useUnassignGroupFromCurator();

  const [assignUserOpen, setAssignUserOpen] = useState(false);
  const [assignGroupOpen, setAssignGroupOpen] = useState(false);
  const [targetUserId, setTargetUserId] = useState<string>("");
  const [targetGroupId, setTargetGroupId] = useState<string>("");

  const curators = users?.users?.filter((u) => u.role === "curator") || [];
  const allUsers = users?.users || [];
  const allGroups = groups || [];

  const assignedClientIds = scope?.managed_client_ids || [];
  const assignedInternalIds = scope?.managed_internal_ids || [];
  const assignedGroupIds = scope?.managed_group_ids || [];

  const assignedClients = allUsers.filter(
    (u) => u.kind === "client" && assignedClientIds.includes(u.id),
  );
  const assignedInternals = allUsers.filter(
    (u) => u.kind === "internal" && assignedInternalIds.includes(u.id),
  );
  const assignedGroups = allGroups.filter((g) => assignedGroupIds.includes(g.id));

  const availableUsers = allUsers.filter(
    (u) =>
      u.role === "user" &&
      u.id !== selectedCuratorId &&
      !assignedClientIds.includes(u.id) &&
      !assignedInternalIds.includes(u.id),
  );
  const availableGroups = allGroups.filter((g) => !assignedGroupIds.includes(g.id));

  const handleAssignUser = async () => {
    if (!selectedCuratorId || !targetUserId) return;
    try {
      await assignUserMut.mutateAsync({
        curatorId: selectedCuratorId,
        targetUserId: Number(targetUserId),
      });
      toast.success("User assigned");
      setAssignUserOpen(false);
      setTargetUserId("");
    } catch {
      toast.error("Failed to assign user");
    }
  };

  const handleUnassignUser = async (userId: number) => {
    if (!selectedCuratorId) return;
    try {
      await unassignUserMut.mutateAsync({ curatorId: selectedCuratorId, targetUserId: userId });
      toast.success("User unassigned");
    } catch {
      toast.error("Failed to unassign user");
    }
  };

  const handleAssignGroup = async () => {
    if (!selectedCuratorId || !targetGroupId) return;
    try {
      await assignGroupMut.mutateAsync({
        curatorId: selectedCuratorId,
        groupId: Number(targetGroupId),
      });
      toast.success("Group assigned");
      setAssignGroupOpen(false);
      setTargetGroupId("");
    } catch {
      toast.error("Failed to assign group");
    }
  };

  const handleUnassignGroup = async (groupId: number) => {
    if (!selectedCuratorId) return;
    try {
      await unassignGroupMut.mutateAsync({ curatorId: selectedCuratorId, groupId });
      toast.success("Group unassigned");
    } catch {
      toast.error("Failed to unassign group");
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Curators</h1>
        <p className="text-muted-foreground">
          Manage curator assignments — which users and groups each curator can access
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Curator list */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Curators</CardTitle>
          </CardHeader>
          <CardContent>
            {curators.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No curators found. Create a user with role "curator" first.
              </p>
            ) : (
              <div className="space-y-1">
                {curators.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setSelectedCuratorId(c.id)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                      selectedCuratorId === c.id
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted"
                    }`}
                  >
                    <div className="font-medium">{c.email}</div>
                    <div className="text-xs opacity-70">ID: {c.id}</div>
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Scope details */}
        <div className="lg:col-span-2">
          {!selectedCuratorId ? (
            <Card>
              <CardContent className="pt-6">
                <p className="text-muted-foreground text-center py-8">
                  Select a curator to view and manage their assignments
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {/* Assigned Users */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle className="text-lg">Assigned Users</CardTitle>
                  <Button size="sm" onClick={() => setAssignUserOpen(true)}>
                    <UserPlus className="h-4 w-4 mr-1" />
                    Assign
                  </Button>
                </CardHeader>
                <CardContent>
                  {assignedClients.length === 0 && assignedInternals.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No users assigned</p>
                  ) : (
                    <div className="space-y-2">
                      {assignedClients.map((u) => (
                        <div key={u.id} className="flex items-center justify-between py-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm">{u.email}</span>
                            <Badge variant="outline" className="text-xs">
                              client
                            </Badge>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleUnassignUser(u.id)}
                          >
                            <UserMinus className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                      {assignedInternals.map((u) => (
                        <div key={u.id} className="flex items-center justify-between py-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm">{u.email}</span>
                            <Badge variant="secondary" className="text-xs">
                              internal
                            </Badge>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleUnassignUser(u.id)}
                          >
                            <UserMinus className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Separator />

              {/* Assigned Groups */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle className="text-lg">Assigned Groups</CardTitle>
                  <Button size="sm" onClick={() => setAssignGroupOpen(true)}>
                    <UserPlus className="h-4 w-4 mr-1" />
                    Assign
                  </Button>
                </CardHeader>
                <CardContent>
                  {assignedGroups.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No groups assigned</p>
                  ) : (
                    <div className="space-y-2">
                      {assignedGroups.map((g) => (
                        <div key={g.id} className="flex items-center justify-between py-1">
                          <span className="text-sm">{g.name}</span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleUnassignGroup(g.id)}
                          >
                            <UserMinus className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>

      {/* Assign User Dialog */}
      <Dialog open={assignUserOpen} onOpenChange={setAssignUserOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Assign User to Curator</DialogTitle>
            <DialogDescription>Select a user to assign to this curator's scope</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>User</Label>
              <Select value={targetUserId} onValueChange={setTargetUserId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a user" />
                </SelectTrigger>
                <SelectContent>
                  {availableUsers.map((u) => (
                    <SelectItem key={u.id} value={String(u.id)}>
                      {u.email} ({u.kind})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAssignUserOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAssignUser} disabled={!targetUserId || assignUserMut.isPending}>
              {assignUserMut.isPending ? "Assigning..." : "Assign"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Assign Group Dialog */}
      <Dialog open={assignGroupOpen} onOpenChange={setAssignGroupOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Assign Group to Curator</DialogTitle>
            <DialogDescription>Select a group to assign to this curator's scope</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Group</Label>
              <Select value={targetGroupId} onValueChange={setTargetGroupId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a group" />
                </SelectTrigger>
                <SelectContent>
                  {availableGroups.map((g) => (
                    <SelectItem key={g.id} value={String(g.id)}>
                      {g.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAssignGroupOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleAssignGroup}
              disabled={!targetGroupId || assignGroupMut.isPending}
            >
              {assignGroupMut.isPending ? "Assigning..." : "Assign"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
