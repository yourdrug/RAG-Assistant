"use client";
import { Check, Copy, Key, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import {
  useClientApiKeys,
  useCreateClientApiKey,
  useRevokeClientApiKey,
  useUsers,
} from "@/shared/api/hooks";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";

export function AdminApiKeysPage() {
  const { data: users } = useUsers();
  const [selClient, setSelClient] = useState<string>("");
  const [newKeyName, setNewKeyName] = useState<string>("");
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const clientUserId = Number(selClient) || 0;
  const { data: apiKeys, isLoading, refetch } = useClientApiKeys(clientUserId);
  const createMut = useCreateClientApiKey(clientUserId);
  const revokeMut = useRevokeClientApiKey(clientUserId);

  const clientUsers = users?.filter((u) => u.kind === "client") || [];

  const handleCreateKey = async () => {
    if (!selClient) return;
    try {
      const result = await createMut.mutateAsync({ name: newKeyName || undefined });
      setCreatedKey(result.api_key);
      setNewKeyName("");
      refetch();
    } catch (e) {
      toast.error("Failed to create API key");
    }
  };

  const handleCopyKey = async () => {
    if (!createdKey) return;
    await navigator.clipboard.writeText(createdKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRevokeKey = async (apiKeyId: number) => {
    try {
      await revokeMut.mutateAsync({ apiKeyId });
      toast.success("API key revoked");
      refetch();
    } catch (e) {
      toast.error("Failed to revoke API key");
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Client API Keys</h1>
        <p className="text-muted-foreground">Manage API keys for client users</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Select Client</CardTitle>
          <CardDescription>Choose a client user to manage their API keys</CardDescription>
        </CardHeader>
        <CardContent>
          <Select value={selClient} onValueChange={setSelClient}>
            <SelectTrigger className="w-full max-w-md">
              <SelectValue placeholder="Select a client user" />
            </SelectTrigger>
            <SelectContent>
              {clientUsers.map((u) => (
                <SelectItem key={u.id} value={String(u.id)}>
                  {u.email} ({u.is_active ? "Active" : "Inactive"})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {selClient && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Create New API Key</CardTitle>
              <CardDescription>Generate a new API key for the selected client</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Key name (optional)"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  className="flex-1 px-3 py-2 border rounded-md"
                />
                <Button onClick={handleCreateKey} disabled={createMut.isPending || !selClient}>
                  <Plus className="h-4 w-4 mr-1" />
                  Create
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" /> API Keys for{" "}
                {clientUserId > 0
                  ? clientUsers.find((u) => u.id === clientUserId)?.email
                  : "Selected Client"}
              </CardTitle>
              <CardDescription>
                Client API keys for authentication in their applications
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="p-3 border rounded-md">
                      <div className="h-4 bg-gray-200 rounded w-1/4 animate-pulse" />
                      <div className="h-3 bg-gray-200 rounded w-1/2 mt-2 animate-pulse" />
                    </div>
                  ))}
                </div>
              ) : apiKeys && apiKeys.length > 0 ? (
                <div className="space-y-3">
                  {apiKeys.map((key) => (
                    <div
                      key={key.id}
                      className="flex items-center justify-between rounded-md border p-4"
                    >
                      <div className="flex items-center gap-3">
                        <Key className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="font-medium">{key.name || "Unnamed key"}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              Prefix: {key.key_prefix}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              Created: {new Date(key.created_at).toLocaleDateString()}
                            </Badge>
                            <Badge
                              variant={key.is_active ? "success" : "destructive"}
                              className="text-xs"
                            >
                              {key.is_active ? "Active" : "Revoked"}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive"
                        onClick={() => handleRevokeKey(key.id)}
                        disabled={revokeMut.isPending || !key.is_active}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No API keys found for this client</p>
              )}
            </CardContent>
          </Card>
        </>
      )}

      <Dialog
        open={!!createdKey}
        onOpenChange={(open) => {
          if (!open) {
            setCreatedKey(null);
            setCopied(false);
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>API Key Created</DialogTitle>
            <DialogDescription>Copy this key now. It will not be shown again.</DialogDescription>
          </DialogHeader>
          <div className="relative">
            <input
              type="text"
              readOnly
              value={createdKey || ""}
              className="w-full px-3 py-2 pr-10 border rounded-md bg-muted font-mono text-sm"
            />
            <button
              onClick={handleCopyKey}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-accent"
            >
              {copied ? (
                <Check className="h-4 w-4 text-emerald-500" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </button>
          </div>
          <DialogFooter>
            <Button
              onClick={() => {
                setCreatedKey(null);
                setCopied(false);
              }}
            >
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
