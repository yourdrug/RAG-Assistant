"use client";
import { type ColumnDef } from "@tanstack/react-table";
import { Pencil } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import { useActVersions, useUpdateActVersion } from "@/shared/api/hooks";
import type { ActSummary, ActVersionReviewItem } from "@/shared/api/types";
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
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";

const sourceBadgeVariant = (source: string) => {
  switch (source) {
    case "extracted_trusted":
      return "success" as const;
    case "extracted":
      return "warning" as const;
    case "manual":
      return "default" as const;
    default:
      return "secondary" as const;
  }
};

const formatConfidence = (v: number | null) => (v != null ? `${Math.round(v * 100)}%` : "—");

export function AdminActVersionsPage() {
  const [offset, setOffset] = useState(0);
  const { data } = useActVersions(offset);
  const updateMut = useUpdateActVersion();
  const [editing, setEditing] = useState<ActVersionReviewItem | null>(null);
  const [form, setForm] = useState({ effective_from: "", effective_to: "", act_id: "" });

  const openEdit = (v: ActVersionReviewItem) => {
    setEditing(v);
    setForm({
      effective_from: v.effective_from ?? "",
      effective_to: v.effective_to ?? "",
      act_id: v.act_id != null ? String(v.act_id) : "",
    });
  };

  const handleSave = async () => {
    if (!editing) return;
    if (form.effective_from && form.effective_to && form.effective_to <= form.effective_from) {
      toast.error("Effective To must be later than Effective From");
      return;
    }
    try {
      await updateMut.mutateAsync({
        versionId: editing.id,
        data: {
          effective_from: form.effective_from || null,
          effective_to: form.effective_to || null,
          act_id: form.act_id ? Number(form.act_id) : null,
        },
      });
      toast.success("Version updated");
      setEditing(null);
    } catch {
      toast.error("Failed to update version");
    }
  };

  const columns: ColumnDef<ActVersionReviewItem>[] = [
    {
      accessorKey: "id",
      header: "ID",
      cell: ({ row }) => <span className="text-muted-foreground">#{row.original.id}</span>,
    },
    {
      accessorKey: "document_filename",
      header: "Document",
      cell: ({ row }) => (
        <span className="font-medium" title={`ID: ${row.original.document_id}`}>
          {row.original.document_filename}
        </span>
      ),
    },
    {
      accessorKey: "act_id",
      header: "Act",
      cell: ({ row }) => {
        const act = data?.acts.find((a) => a.id === row.original.act_id);
        return act ? (
          <span className="font-medium">{act.act_number || act.title}</span>
        ) : (
          <span className="text-muted-foreground italic">Unlinked</span>
        );
      },
    },
    {
      accessorKey: "effective_from",
      header: "From",
      cell: ({ row }) => row.original.effective_from ?? "—",
    },
    {
      accessorKey: "effective_to",
      header: "To",
      cell: ({ row }) => row.original.effective_to ?? "—",
    },
    {
      accessorKey: "date_source",
      header: "Source",
      cell: ({ row }) => (
        <Badge variant={sourceBadgeVariant(row.original.date_source)}>
          {row.original.date_source}
        </Badge>
      ),
    },
    {
      accessorKey: "date_confidence",
      header: "Confidence",
      cell: ({ row }) => <span>{formatConfidence(row.original.date_confidence)}</span>,
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => (
        <Button variant="ghost" size="sm" onClick={() => openEdit(row.original)}>
          <Pencil className="h-4 w-4 mr-1" />
          Review
        </Button>
      ),
    },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Act Versions</h1>
          <p className="text-muted-foreground">
            All versions
            {data != null && (
              <Badge variant="warning" className="ml-2">
                {data.total}
              </Badge>
            )}
          </p>
        </div>
      </div>
      <Card>
        <CardContent className="pt-6">
          <DataTable
            columns={columns}
            data={data?.versions || []}
            serverPagination={{
              canPreviousPage: offset > 0,
              canNextPage: data?.next_offset != null,
              onPreviousPage: () => setOffset((value) => Math.max(0, value - 100)),
              onNextPage: () => {
                if (data?.next_offset != null) setOffset(data.next_offset);
              },
            }}
            searchKey="date_source"
            searchPlaceholder="Search by source..."
          />
        </CardContent>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={!!editing} onOpenChange={() => setEditing(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Pencil className="h-5 w-5" />
              Review Act Version #{editing?.id}
            </DialogTitle>
            <DialogDescription>Set effective dates and link to a regulatory act.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Effective From</Label>
              <Input
                type="date"
                value={form.effective_from}
                onChange={(e) => setForm({ ...form, effective_from: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label>Effective To (exclusive)</Label>
              <Input
                type="date"
                value={form.effective_to}
                onChange={(e) => setForm({ ...form, effective_to: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label>Regulatory Act</Label>
              <Select
                value={form.act_id}
                onValueChange={(v) => setForm({ ...form, act_id: v === "__none__" ? "" : v })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select act..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">No act (unlink)</SelectItem>
                  {(data?.acts || []).map((a: ActSummary) => (
                    <SelectItem key={a.id} value={String(a.id)}>
                      {a.act_number ? `${a.act_number} — ` : ""}
                      {a.title}
                      {a.visibility_scope ? ` (${a.visibility_scope})` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(null)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={updateMut.isPending}>
              {updateMut.isPending ? "Saving..." : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
