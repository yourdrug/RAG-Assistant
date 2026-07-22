"use client";
import { useState } from "react";
import { useUsers, useClientAssignments, useAssignClient, useUnassignClient } from "@/shared/api/hooks";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Plus, Trash2, Link as LinkIcon } from "lucide-react";
import toast from "react-hot-toast";

export function AdminClientsPage() {
  const { data: users } = useUsers();
  const [selClient, setSelClient] = useState("");
  const [selInternal, setSelInternal] = useState("");
  const assignMut = useAssignClient();
  const unassignMut = useUnassignClient();
  const { data: assignments, refetch } = useClientAssignments(Number(selClient) || 0);

  const clientUsers = users?.filter((u) => u.kind === "client") || [];
  const internalUsers = users?.filter((u) => u.kind === "internal") || [];

  const handleAssign = async () => {
    if (!selClient || !selInternal) return;
    try { await assignMut.mutateAsync({ clientUserId: Number(selClient), internalUserId: Number(selInternal) }); toast.success("Assigned"); setSelInternal(""); refetch(); } catch { toast.error("Failed"); }
  };

  const handleUnassign = async (uid: number) => {
    if (!selClient) return;
    try { await unassignMut.mutateAsync({ clientUserId: Number(selClient), internalUserId: uid }); toast.success("Unassigned"); refetch(); } catch { toast.error("Failed"); }
  };

  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">Client Assignments</h1><p className="text-muted-foreground">Assign internal users to client accounts</p></div>
      <Card>
        <CardHeader><CardTitle>Select Client</CardTitle><CardDescription>Choose a client user</CardDescription></CardHeader>
        <CardContent>
          <Select value={selClient} onValueChange={setSelClient}>
            <SelectTrigger className="w-full max-w-md"><SelectValue placeholder="Select a client user" /></SelectTrigger>
            <SelectContent>{clientUsers.map((u) => <SelectItem key={u.id} value={String(u.id)}>{u.email} ({u.is_active ? "Active" : "Inactive"})</SelectItem>)}</SelectContent>
          </Select>
        </CardContent>
      </Card>
      {selClient && (
        <Card>
          <CardHeader><CardTitle>Assigned Users</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Select value={selInternal} onValueChange={setSelInternal}><SelectTrigger className="flex-1 max-w-md"><SelectValue placeholder="Select internal user" /></SelectTrigger><SelectContent>{internalUsers.map((u) => <SelectItem key={u.id} value={String(u.id)}>{u.email}</SelectItem>)}</SelectContent></Select>
              <Button onClick={handleAssign} disabled={!selInternal}><Plus className="h-4 w-4 mr-1" />Assign</Button>
            </div>
            {assignments && assignments.length > 0 ? (
              <div className="space-y-2">{assignments.map((a) => (
                <div key={a.internal_user_id} className="flex items-center justify-between rounded-md border p-3">
                  <div className="flex items-center gap-2"><LinkIcon className="h-4 w-4 text-muted-foreground" /><span className="text-sm font-medium">{a.email}</span><Badge variant="outline">#{a.internal_user_id}</Badge></div>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => handleUnassign(a.internal_user_id)}><Trash2 className="h-3 w-3 text-destructive" /></Button>
                </div>
              ))}</div>
            ) : <p className="text-sm text-muted-foreground">No users assigned</p>}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
