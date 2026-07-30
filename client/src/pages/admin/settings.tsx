"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/shared/api/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";
import { Save, RotateCcw, Settings } from "lucide-react";
import toast from "react-hot-toast";

interface ConfigParam {
  key: string;
  value: string;
  value_type: string;
  category: string;
  description: string | null;
  min_value: number | null;
  max_value: number | null;
}

const CATEGORY_LABELS: Record<string, string> = {
  rag: "RAG Pipeline",
  hybrid: "Hybrid Search",
  ingestion: "Ingestion",
};

export function AdminSettingsPage() {
  const queryClient = useQueryClient();
  const [edits, setEdits] = useState<Record<string, string>>({});

  const { data: params, isLoading } = useQuery({
    queryKey: ["admin", "config"],
    queryFn: async () => (await apiClient.get<ConfigParam[]>("/admin/config")).data,
  });

  const updateMutation = useMutation({
    mutationFn: async ({ key, value }: { key: string; value: string }) => {
      return (await apiClient.put(`/admin/config/${key}`, { value })).data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["admin", "config"] });
      setEdits((prev) => {
        const next = { ...prev };
        delete next[variables.key];
        return next;
      });
      toast.success(`"${variables.key}" updated`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || "Update failed");
    },
  });

  const handleSave = (key: string) => {
    const val = edits[key];
    if (val === undefined) return;
    updateMutation.mutate({ key, value: val });
  };

  const handleReset = (key: string) => {
    setEdits((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const categories = [...new Set(params?.map((p) => p.category) ?? [])];

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div>
          <Skeleton className="h-8 w-48 mb-2" />
          <Skeleton className="h-4 w-72" />
        </div>
        {[1, 2].map((i) => (
          <Card key={i}>
            <CardContent className="pt-6">
              <Skeleton className="h-40 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground">All dynamic configuration parameters</p>
      </div>

      {categories.map((cat) => {
        const catParams = params?.filter((p) => p.category === cat) ?? [];
        return (
          <Card key={cat}>
            <CardHeader>
              <CardTitle>{CATEGORY_LABELS[cat] ?? cat}</CardTitle>
              <CardDescription>
                {catParams.length} parameter{catParams.length !== 1 ? "s" : ""}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[180px]">Key</TableHead>
                    <TableHead className="w-[280px]">Value</TableHead>
                    <TableHead>Description</TableHead>
                    <TableHead className="w-[80px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {catParams.map((p) => {
                    const isEdited = edits[p.key] !== undefined;
                    const displayValue = isEdited ? edits[p.key] : p.value;
                    return (
                      <TableRow key={p.key}>
                        <TableCell className="font-mono text-sm">{p.key}</TableCell>
                        <TableCell>
                          {p.value_type === "bool" ? (
                            <div className="flex items-center gap-2">
                              <Button
                                size="sm"
                                variant={displayValue === "true" ? "default" : "outline"}
                                onClick={() => setEdits((prev) => ({ ...prev, [p.key]: "true" }))}
                              >
                                ON
                              </Button>
                              <Button
                                size="sm"
                                variant={displayValue === "false" ? "destructive" : "outline"}
                                onClick={() => setEdits((prev) => ({ ...prev, [p.key]: "false" }))}
                              >
                                OFF
                              </Button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2">
                              <Input
                                type="number"
                                step={p.value_type === "float" ? "0.1" : "1"}
                                value={displayValue}
                                onChange={(e) => setEdits((prev) => ({ ...prev, [p.key]: e.target.value }))}
                                className="w-28"
                              />
                              {p.min_value !== null && p.max_value !== null && (
                                <span className="text-xs text-muted-foreground whitespace-nowrap">
                                  {p.min_value} – {p.max_value}
                                </span>
                              )}
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">{p.description}</TableCell>
                        <TableCell>
                          {isEdited && (
                            <div className="flex items-center gap-1">
                              <Button size="sm" variant="ghost" onClick={() => handleSave(p.key)} disabled={updateMutation.isPending}>
                                <Save className="h-4 w-4" />
                              </Button>
                              <Button size="sm" variant="ghost" onClick={() => handleReset(p.key)}>
                                <RotateCcw className="h-4 w-4" />
                              </Button>
                            </div>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
