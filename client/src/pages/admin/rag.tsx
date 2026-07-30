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
import { Save, RotateCcw } from "lucide-react";
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

export function AdminRAGPage() {
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
      toast.success(`Parameter "${variables.key}" updated`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || "Failed to update parameter");
    },
  });

  const ragParams = params?.filter((p) => p.category === "rag") ?? [];
  const hybridParams = params?.filter((p) => p.category === "hybrid") ?? [];

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

  const renderValue = (param: ConfigParam) => {
    const isEdited = edits[param.key] !== undefined;
    const displayValue = isEdited ? edits[param.key] : param.value;

    if (param.value_type === "bool") {
      return (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={displayValue === "true" ? "default" : "outline"}
            onClick={() => setEdits((prev) => ({ ...prev, [param.key]: "true" }))}
          >
            ON
          </Button>
          <Button
            size="sm"
            variant={displayValue === "false" ? "destructive" : "outline"}
            onClick={() => setEdits((prev) => ({ ...prev, [param.key]: "false" }))}
          >
            OFF
          </Button>
        </div>
      );
    }

    return (
      <div className="flex items-center gap-2">
        <Input
          type={param.value_type === "int" ? "number" : "number"}
          step={param.value_type === "float" ? "0.1" : "1"}
          value={displayValue}
          onChange={(e) => setEdits((prev) => ({ ...prev, [param.key]: e.target.value }))}
          className="w-28"
        />
        {param.min_value !== null && param.max_value !== null && (
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {param.min_value} – {param.max_value}
          </span>
        )}
      </div>
    );
  };

  const renderActions = (param: ConfigParam) => {
    const isEdited = edits[param.key] !== undefined;
    return (
      <div className="flex items-center gap-1">
        {isEdited && (
          <>
            <Button size="sm" variant="ghost" onClick={() => handleSave(param.key)} disabled={updateMutation.isPending}>
              <Save className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="ghost" onClick={() => handleReset(param.key)}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </>
        )}
        {isEdited && <Badge variant="outline">modified</Badge>}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div>
          <Skeleton className="h-8 w-48 mb-2" />
          <Skeleton className="h-4 w-72" />
        </div>
        <Card>
          <CardContent className="pt-6">
            <Skeleton className="h-64 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">RAG Settings</h1>
        <p className="text-muted-foreground">Configure RAG pipeline parameters (changes apply instantly)</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Retrieval Parameters</CardTitle>
          <CardDescription>Control how many documents are fetched and ranked</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Parameter</TableHead>
                <TableHead className="w-[300px]">Value</TableHead>
                <TableHead>Description</TableHead>
                <TableHead className="w-[100px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ragParams.map((p) => (
                <TableRow key={p.key}>
                  <TableCell className="font-mono text-sm">{p.key}</TableCell>
                  <TableCell>{renderValue(p)}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{p.description}</TableCell>
                  <TableCell>{renderActions(p)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Hybrid Search</CardTitle>
          <CardDescription>BM25 + dense RRF merge settings</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Parameter</TableHead>
                <TableHead className="w-[300px]">Value</TableHead>
                <TableHead>Description</TableHead>
                <TableHead className="w-[100px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {hybridParams.map((p) => (
                <TableRow key={p.key}>
                  <TableCell className="font-mono text-sm">{p.key}</TableCell>
                  <TableCell>{renderValue(p)}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{p.description}</TableCell>
                  <TableCell>{renderActions(p)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
