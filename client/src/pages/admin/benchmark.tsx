"use client";
import {
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  Filter,
  GitCompare,
  History,
  Layers,
  ListChecks,
  Play,
  Plus,
  RefreshCw,
  Settings2,
  Target,
  TrendingUp,
  X,
  XCircle,
} from "lucide-react";
import { Fragment, useMemo, useRef, useState } from "react";
import toast from "react-hot-toast";
import { apiClient } from "@/shared/api/client";
import {
  useApplyRunConfig,
  useBenchmarkHistory,
  useBenchmarkQuestions,
  useBenchmarkResult,
  useBenchmarkRuns,
  useCancelSweep,
  useCompareRuns,
  useCreateBenchmarkQuestion,
  useCreateSweep,
  useDeleteBenchmarkQuestion,
  useImportBenchmarkQuestions,
  useRegressionCheck,
  useSourceFiles,
  useSweeps,
  useUpdateBenchmarkQuestion,
} from "@/shared/api/hooks";
import type {
  BenchmarkQuestion,
  BenchmarkQuestionCreate,
  BenchmarkRun,
} from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/shared/ui/dialog";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Progress } from "@/shared/ui/progress";
import { Skeleton } from "@/shared/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/table";
import { Textarea } from "@/shared/ui/textarea";

// ─── Reusable widgets ─────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  icon: Icon,
  color = "text-foreground",
}: {
  label: string;
  value: string | number | null;
  icon: typeof TrendingUp;
  color?: string;
}) {
  return (
    <Card>
      <CardContent className="pt-4">
        <div className="flex items-center gap-2 mb-1">
          <Icon className={`h-4 w-4 ${color}`} />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
        <div className={`text-2xl font-bold font-mono ${color}`}>
          {value !== null && value !== undefined ? value : "-"}
        </div>
      </CardContent>
    </Card>
  );
}

function ScoreBadge({ score, max = 10 }: { score: number | null | undefined; max?: number }) {
  if (score === null || score === undefined) return <Badge variant="secondary">-</Badge>;
  const pct = (score / max) * 100;
  if (pct >= 70) return <Badge variant="success">{typeof score === "number" ? score.toFixed(1) : score}</Badge>;
  if (pct >= 40) return <Badge variant="warning">{typeof score === "number" ? score.toFixed(1) : score}</Badge>;
  return <Badge variant="destructive">{typeof score === "number" ? score.toFixed(1) : score}</Badge>;
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "done"
      ? "success"
      : status === "running"
        ? "default"
        : status === "failed"
          ? "destructive"
          : status === "cancelled"
            ? "warning"
            : "secondary";
  return <Badge variant={variant as any}>{status}</Badge>;
}

function SourceHintPicker({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (val: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const { data: files } = useSourceFiles(query || undefined);
  const [highlightIdx, setHighlightIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = files || [];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && highlightIdx >= 0 && filtered[highlightIdx]) {
      e.preventDefault();
      onChange(filtered[highlightIdx]);
      setQuery("");
      setOpen(false);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-1">
        <Input
          ref={inputRef}
          value={value || ""}
          onChange={(e) => {
            const v = e.target.value || null;
            onChange(v);
            setQuery(v || "");
            setOpen(true);
            setHighlightIdx(-1);
          }}
          onFocus={() => {
            setOpen(true);
            setQuery(value || "");
          }}
          onBlur={() => {
            // Delay to allow click on dropdown item
            setTimeout(() => setOpen(false), 200);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Select or type filename"
          className="flex-1"
        />
        {value && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 shrink-0"
            onClick={() => {
              onChange(null);
              setQuery("");
              inputRef.current?.focus();
            }}
          >
            <X className="h-3 w-3" />
          </Button>
        )}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0 shrink-0"
          onClick={() => {
            setOpen(true);
            setQuery("");
            inputRef.current?.focus();
          }}
        >
          <Filter className="h-3 w-3" />
        </Button>
      </div>
      {open && filtered.length > 0 && (
        <div
          ref={listRef}
          className="absolute z-50 top-full mt-1 w-full bg-popover border rounded-md shadow-md max-h-[200px] overflow-auto"
        >
          {filtered.map((file, idx) => (
            <div
              key={file}
              className={`px-3 py-1.5 text-sm cursor-pointer truncate ${
                idx === highlightIdx ? "bg-accent text-accent-foreground" : "hover:bg-muted"
              }`}
              onMouseDown={(e) => {
                e.preventDefault();
                onChange(file);
                setQuery("");
                setOpen(false);
              }}
              onMouseEnter={() => setHighlightIdx(idx)}
            >
              {file}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Tab 1: Questions ─────────────────────────────────────────────────────

function QuestionsTab() {
  const [search, setSearch] = useState("");
  const [datasetFilter, setDatasetFilter] = useState<string>("");
  const { data, isLoading, refetch } = useBenchmarkQuestions({
    search: search || undefined,
    dataset: datasetFilter || undefined,
    limit: 500,
  });
  const createQuestion = useCreateBenchmarkQuestion();
  const updateQuestion = useUpdateBenchmarkQuestion();
  const deleteQuestion = useDeleteBenchmarkQuestion();
  const importQuestions = useImportBenchmarkQuestions();

  const [dialogOpen, setDialogOpen] = useState(false);
  const [editQuestion, setEditQuestion] = useState<BenchmarkQuestion | null>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [importText, setImportText] = useState("");
  const [form, setForm] = useState<BenchmarkQuestionCreate>({
    question: "",
    expected_answer: "",
    source_hint: "",
    tags: [],
    dataset: "main",
    notes: "",
  });

  const handleCreate = async () => {
    if (!form.question.trim()) return;
    try {
      if (editQuestion) {
        await updateQuestion.mutateAsync({ id: editQuestion.id, data: form });
        toast.success("Question updated");
      } else {
        await createQuestion.mutateAsync(form);
        toast.success("Question created");
      }
      setDialogOpen(false);
      setEditQuestion(null);
      setForm({ question: "", expected_answer: "", source_hint: "", tags: [], dataset: "main", notes: "" });
      refetch();
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Failed");
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this question?")) return;
    try {
      await deleteQuestion.mutateAsync(id);
      toast.success("Deleted");
      refetch();
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Failed");
    }
  };

  const handleImport = async () => {
    try {
      const parsed = JSON.parse(importText);
      const questions = Array.isArray(parsed) ? parsed : [parsed];
      await importQuestions.mutateAsync({ questions });
      toast.success(`Imported ${questions.length} questions`);
      setImportDialogOpen(false);
      setImportText("");
      refetch();
    } catch (e: any) {
      toast.error(e instanceof SyntaxError ? "Invalid JSON" : e.response?.data?.detail || "Failed");
    }
  };

  const handleExport = async () => {
    try {
      const { data } = await apiClient.get(
        `/admin/benchmark/questions/export${datasetFilter ? `?dataset=${datasetFilter}` : ""}`
      );
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `benchmark-questions${datasetFilter ? `-${datasetFilter}` : ""}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Exported");
    } catch {
      toast.error("Export failed");
    }
  };

  const readFile = (file: File) => {
    if (!file.name.endsWith(".json") && file.type !== "application/json") {
      toast.error("Only .json files are supported");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text === "string") {
        setImportText(text);
        toast.success(`Loaded ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
      }
    };
    reader.onerror = () => toast.error("Failed to read file");
    reader.readAsText(file);
  };

  const openEdit = (q: BenchmarkQuestion) => {
    setEditQuestion(q);
    setForm({
      question: q.question,
      expected_answer: q.expected_answer || "",
      source_hint: q.source_hint || "",
      tags: q.tags || [],
      dataset: q.dataset,
      notes: q.notes || "",
    });
    setDialogOpen(true);
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center gap-3 flex-wrap">
        <Input
          placeholder="Search questions..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
        />
        <Input
          placeholder="Dataset filter"
          value={datasetFilter}
          onChange={(e) => setDatasetFilter(e.target.value)}
          className="max-w-[150px]"
        />
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-1" /> Refresh
        </Button>
        <Button variant="outline" size="sm" onClick={handleExport}>
          <Download className="h-4 w-4 mr-1" /> Export
        </Button>
        <Button variant="outline" size="sm" onClick={() => setImportDialogOpen(true)}>
          <Layers className="h-4 w-4 mr-1" /> Import
        </Button>
        <Button
          size="sm"
          onClick={() => {
            setEditQuestion(null);
            setForm({ question: "", expected_answer: "", source_hint: "", tags: [], dataset: "main", notes: "" });
            setDialogOpen(true);
          }}
        >
          <Plus className="h-4 w-4 mr-1" /> Add Question
        </Button>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : (
        <div className="max-h-[500px] overflow-auto border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[40px]">#</TableHead>
                <TableHead>Question</TableHead>
                <TableHead>Expected Answer</TableHead>
                <TableHead>Source Hint</TableHead>
                <TableHead>Tags</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead className="w-[60px]">Active</TableHead>
                <TableHead className="w-[80px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.questions.map((q, idx) => (
                <TableRow key={q.id}>
                  <TableCell className="text-xs text-muted-foreground">{idx + 1}</TableCell>
                  <TableCell className="text-sm max-w-[250px]">
                    <div className="truncate" title={q.question}>{q.question}</div>
                  </TableCell>
                  <TableCell className="text-xs max-w-[200px]">
                    <div className="truncate text-muted-foreground" title={q.expected_answer || ""}>
                      {q.expected_answer || "-"}
                    </div>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">{q.source_hint || "-"}</TableCell>
                  <TableCell>
                    <div className="flex gap-1 flex-wrap">
                      {(q.tags || []).map((t) => (
                        <Badge key={t} variant="secondary" className="text-xs">{t}</Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell className="text-xs">{q.dataset}</TableCell>
                  <TableCell>
                    {q.is_active ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <XCircle className="h-4 w-4 text-red-500" />
                    )}
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="sm" onClick={() => openEdit(q)}>
                        Edit
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => handleDelete(q.id)}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {data?.questions.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                    No questions yet. Add your first question above.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      )}
      <div className="text-xs text-muted-foreground">
        {data?.total ?? 0} questions total
      </div>

      {/* Create/Edit Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{editQuestion ? "Edit Question" : "Add Question"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <Label>Question *</Label>
              <Textarea
                value={form.question}
                onChange={(e) => setForm({ ...form, question: e.target.value })}
                placeholder="Enter the test question..."
              />
            </div>
            <div>
              <Label>Expected Answer</Label>
              <Textarea
                value={form.expected_answer || ""}
                onChange={(e) => setForm({ ...form, expected_answer: e.target.value || null })}
                placeholder="Optional expected answer for correctness scoring"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>Source Hint</Label>
                <SourceHintPicker
                  value={form.source_hint || null}
                  onChange={(v) => setForm({ ...form, source_hint: v })}
                />
              </div>
              <div>
                <Label>Dataset</Label>
                <Input
                  value={form.dataset}
                  onChange={(e) => setForm({ ...form, dataset: e.target.value })}
                />
              </div>
            </div>
            <div>
              <Label>Tags (comma-separated)</Label>
              <Input
                value={(form.tags || []).join(", ")}
                onChange={(e) =>
                  setForm({ ...form, tags: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })
                }
                placeholder="legal, narrow, regression"
              />
            </div>
            <div>
              <Label>Notes</Label>
              <Input
                value={form.notes || ""}
                onChange={(e) => setForm({ ...form, notes: e.target.value || null })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleCreate} disabled={createQuestion.isPending || updateQuestion.isPending}>
              {editQuestion ? "Update" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Import Dialog */}
      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Import Questions</DialogTitle>
          </DialogHeader>

          {/* File upload zone */}
          <div
            className="border-2 border-dashed rounded-md p-6 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("border-primary", "bg-primary/5"); }}
            onDragLeave={(e) => { e.currentTarget.classList.remove("border-primary", "bg-primary/5"); }}
            onDrop={(e) => {
              e.preventDefault();
              e.currentTarget.classList.remove("border-primary", "bg-primary/5");
              const file = e.dataTransfer.files?.[0];
              if (file) readFile(file);
            }}
            onClick={() => document.getElementById("import-file-input")?.click()}
          >
            <Layers className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Drag & drop a JSON file here, or click to browse
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              .json format — array of question objects
            </p>
            <input
              id="import-file-input"
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) readFile(file);
                e.target.value = "";
              }}
            />
          </div>

          <div className="text-center text-xs text-muted-foreground">— or paste JSON below —</div>

          <Textarea
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            placeholder='[{"question": "...", "expected_answer": "...", "source_hint": "..."}]'
            className="h-32 font-mono text-xs"
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setImportDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleImport} disabled={importQuestions.isPending || !importText.trim()}>
              {importQuestions.isPending ? "Importing..." : "Import"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ─── Tab 2: Sweep Builder ─────────────────────────────────────────────────

function SweepBuilderTab({ onSweepCreated }: { onSweepCreated: (id: number) => void }) {
  const createSweep = useCreateSweep();
  const [strategy, setStrategy] = useState<"grid" | "random" | "successive_halving">("grid");
  const [topNLlm, setTopNLlm] = useState(3);
  const [dataset, setDataset] = useState("main");

  // Parameter configs
  const [params, setParams] = useState<
    Record<string, { enabled: boolean; values: string; min: string; max: string; step: string }>
  >({
    top_k: { enabled: false, values: "4,6,8,10", min: "", max: "", step: "" },
    fetch_k: { enabled: false, values: "20,30,40", min: "", max: "", step: "" },
    dense_weight: { enabled: false, values: "0.5,1.0,1.5", min: "", max: "", step: "" },
    sparse_weight: { enabled: false, values: "0.5,1.0,1.5", min: "", max: "", step: "" },
    rrf_k: { enabled: false, values: "30,60,90", min: "", max: "", step: "" },
    rerank_min_score: { enabled: false, values: "0.05,0.10,0.15", min: "", max: "", step: "" },
  });

  // Objective weights
  const [weights, setWeights] = useState({
    hit_rate: 0.4,
    faithfulness: 0.3,
    relevancy: 0.3,
  });

  const estimatedConfigs = useMemo(() => {
    let total = 1;
    for (const [, cfg] of Object.entries(params)) {
      if (!cfg.enabled) continue;
      if (strategy === "random") return 50; // fixed for random
      const values = cfg.values.split(",").map((s) => s.trim()).filter(Boolean);
      total *= values.length || 1;
    }
    return total;
  }, [params, strategy]);

  const handleStart = async () => {
    const searchSpace: Record<string, any> = {};
    for (const [key, cfg] of Object.entries(params)) {
      if (!cfg.enabled) continue;
      const values = cfg.values.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
      if (values.length > 0) {
        searchSpace[key] = { values };
      }
    }

    if (Object.keys(searchSpace).length === 0) {
      toast.error("Enable at least one parameter");
      return;
    }

    try {
      const result = await createSweep.mutateAsync({
        strategy,
        search_space: searchSpace,
        objective_weights: weights,
        dataset,
        top_n_llm: topNLlm,
      });
      toast.success(`Sweep #${result.id} started`);
      onSweepCreated(result.id);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Failed to create sweep");
    }
  };

  return (
    <div className="space-y-6">
      {/* Strategy */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings2 className="h-5 w-5" /> Strategy
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            {(["grid", "random", "successive_halving"] as const).map((s) => (
              <Button
                key={s}
                variant={strategy === s ? "default" : "outline"}
                onClick={() => setStrategy(s)}
              >
                {s}
              </Button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {strategy === "grid" && "Cartesian product of all values. Best for 2-4 parameters."}
            {strategy === "random" && "50 random points from the search space. Good for 5+ parameters."}
            {strategy === "successive_halving" && "Evaluate all on subset, keep top 50%, repeat."}
          </p>
        </CardContent>
      </Card>

      {/* Parameters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" /> Parameters
          </CardTitle>
          <CardDescription>
            Enable parameters and provide values (comma-separated) or ranges.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(params).map(([key, cfg]) => (
              <div
                key={key}
                className={`border rounded-md p-3 ${cfg.enabled ? "border-primary bg-primary/5" : ""}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={cfg.enabled}
                      onChange={(e) => setParams({ ...params, [key]: { ...cfg, enabled: e.target.checked } })}
                      className="rounded"
                    />
                    {key}
                  </label>
                  <Badge variant="secondary" className="text-xs">cheap</Badge>
                </div>
                {cfg.enabled && (
                  <Input
                    value={cfg.values}
                    onChange={(e) => setParams({ ...params, [key]: { ...cfg, values: e.target.value } })}
                    placeholder="values: 4,6,8,10"
                    className="text-xs font-mono"
                  />
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Weights + Config */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Objective Weights</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {Object.entries(weights).map(([key, val]) => (
              <div key={key} className="flex items-center gap-3">
                <Label className="w-28 text-xs">{key}</Label>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={val}
                  onChange={(e) => setWeights({ ...weights, [key]: parseFloat(e.target.value) || 0 })}
                  className="w-20"
                />
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-3">
              <Label className="w-28 text-xs">Dataset</Label>
              <Input value={dataset} onChange={(e) => setDataset(e.target.value)} className="flex-1" />
            </div>
            <div className="flex items-center gap-3">
              <Label className="w-28 text-xs">Top-N for LLM</Label>
              <Input
                type="number"
                min="0"
                max="20"
                value={topNLlm}
                onChange={(e) => setTopNLlm(parseInt(e.target.value) || 0)}
                className="w-20"
              />
            </div>
            <div className="mt-4 p-3 bg-muted rounded-md text-sm">
              <div className="font-medium">Estimated combinations</div>
              <div className="text-2xl font-bold font-mono mt-1">{estimatedConfigs}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {strategy === "grid"
                  ? `~${Math.ceil(estimatedConfigs * 0.5)}min retrieval-only + ~${topNLlm * 2}min LLM-judge`
                  : `${estimatedConfigs} random evaluations`}
              </div>
            </div>
            <Button className="w-full mt-2" onClick={handleStart} disabled={createSweep.isPending}>
              <Play className="h-4 w-4 mr-2" />
              {createSweep.isPending ? "Starting..." : "Start Sweep"}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ─── Tab 3: Live Progress / Sweeps ────────────────────────────────────────

function SweepProgressTab({
  activeSweepId,
  onSweepSelect,
}: {
  activeSweepId: number | null;
  onSweepSelect: (id: number) => void;
}) {
  const { data: sweepsData, isLoading } = useSweeps();
  const cancelSweep = useCancelSweep();

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Sweeps</h3>
        <Button variant="outline" size="sm" onClick={() => onSweepSelect(0)}>
          <RefreshCw className="h-4 w-4 mr-1" /> Refresh
        </Button>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      ) : (
        <div className="border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Strategy</TableHead>
                <TableHead>Progress</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead>Best Run</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sweepsData?.sweeps.map((s) => (
                <TableRow
                  key={s.id}
                  className={activeSweepId === s.id ? "bg-muted/50" : "cursor-pointer hover:bg-muted/30"}
                  onClick={() => onSweepSelect(s.id)}
                >
                  <TableCell className="font-mono">#{s.id}</TableCell>
                  <TableCell>
                    <StatusBadge status={s.status} />
                  </TableCell>
                  <TableCell className="text-xs">{s.strategy}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Progress
                        value={s.total_configs > 0 ? (s.evaluated_configs / s.total_configs) * 100 : 0}
                        className="w-24 h-2"
                      />
                      <span className="text-xs text-muted-foreground">
                        {s.evaluated_configs}/{s.total_configs}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-xs">{s.dataset}</TableCell>
                  <TableCell className="text-xs font-mono">
                    {s.best_run_id ? `#${s.best_run_id}` : "-"}
                  </TableCell>
                  <TableCell>
                    {(s.status === "pending" || s.status === "running") && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          cancelSweep.mutate(s.id);
                        }}
                      >
                        Cancel
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {sweepsData?.sweeps.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                    No sweeps yet. Create one in the Sweep Builder tab.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}

// ─── Tab 4: Leaderboard ───────────────────────────────────────────────────

function LeaderboardTab() {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [expandedRunId, setExpandedRunId] = useState<number | null>(null);
  const { data: runsData, isLoading } = useBenchmarkRuns({ limit: 50 });
  const applyConfig = useApplyRunConfig();
  const { data: compareData } = useCompareRuns(selectedIds);
  const { data: runDetail } = useBenchmarkResult(expandedRunId);

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id].slice(-5)
    );
  };

  const handleApply = async (run: BenchmarkRun) => {
    if (!confirm(`Apply config from run #${run.id}? This will update live settings.`)) return;
    try {
      const result = await applyConfig.mutateAsync(run.id);
      if (result.failed.length > 0) {
        toast(
          `Applied ${result.applied}, failed ${result.failed.length}: ${result.failed.map((f) => f.key).join(", ")}`,
          { icon: "\u26a0\ufe0f" }
        );
      } else {
        toast.success(`Applied ${result.applied} config parameters: ${result.keys.join(", ")}`);
      }
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Failed to apply");
    }
  };

  return (
    <div className="space-y-4">
      {selectedIds.length >= 2 && (
        <Card className="border-primary">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <GitCompare className="h-4 w-4" /> Comparing {selectedIds.length} runs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-auto max-h-[200px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Key</TableHead>
                    {compareData?.runs.map((r) => (
                      <TableHead key={r.id}>Run #{r.id}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(compareData?.diff || {}).map(([key, values]) => (
                    <TableRow key={key}>
                      <TableCell className="font-mono text-xs">{key}</TableCell>
                      {values.map((v, i) => (
                        <TableCell key={i} className="font-mono text-xs">
                          {v.value !== null && v.value !== undefined ? String(v.value) : "-"}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            <Button variant="outline" size="sm" className="mt-2" onClick={() => setSelectedIds([])}>
              Clear selection
            </Button>
          </CardContent>
        </Card>
      )}

      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : (
        <div className="max-h-[500px] overflow-auto border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[30px]"></TableHead>
                <TableHead className="w-[30px]"></TableHead>
                <TableHead>Run</TableHead>
                <TableHead>Config</TableHead>
                <TableHead>Hit Rate</TableHead>
                <TableHead>Faithfulness</TableHead>
                <TableHead>Relevancy</TableHead>
                <TableHead>Composite</TableHead>
                <TableHead>LLM</TableHead>
                <TableHead>Time</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runsData?.runs.map((run) => {
                const isExpanded = expandedRunId === run.id;
                return (
                  <Fragment key={run.id}>
                    <TableRow className={isExpanded ? "bg-muted/50" : ""}>
                      <TableCell>
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(run.id)}
                          onChange={() => toggleSelect(run.id)}
                          className="rounded"
                        />
                      </TableCell>
                      <TableCell>
                        <button
                          className="p-0.5 hover:bg-muted rounded"
                          onClick={() => setExpandedRunId(isExpanded ? null : run.id)}
                        >
                          {isExpanded ? (
                            <ChevronDown className="h-3.5 w-3.5" />
                          ) : (
                            <ChevronRight className="h-3.5 w-3.5" />
                          )}
                        </button>
                      </TableCell>
                      <TableCell className="font-mono text-xs">#{run.id}</TableCell>
                      <TableCell className="text-xs max-w-[200px]">
                        <div className="truncate font-mono">
                          {Object.entries(run.config_json)
                            .slice(0, 3)
                            .map(([k, v]) => `${k}=${v}`)
                            .join(" ")}
                        </div>
                      </TableCell>
                      <TableCell>
                        {run.summary_metrics.hit_rate !== null && run.summary_metrics.hit_rate !== undefined ? (
                          <span className={run.summary_metrics.hit_rate >= 0.5 ? "text-green-500" : "text-red-500"}>
                            {(run.summary_metrics.hit_rate * 100).toFixed(0)}%
                          </span>
                        ) : (
                          "-"
                        )}
                      </TableCell>
                      <TableCell>
                        <ScoreBadge score={run.summary_metrics.faithfulness ?? run.summary_metrics.avg_faithfulness} />
                      </TableCell>
                      <TableCell>
                        <ScoreBadge score={run.summary_metrics.relevancy ?? run.summary_metrics.avg_relevancy} />
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {run.summary_metrics.composite?.toFixed(3) ?? "-"}
                      </TableCell>
                      <TableCell>
                        {run.llm_evaluated ? (
                          <Badge variant="success" className="text-xs">LLM</Badge>
                        ) : (
                          <Badge variant="secondary" className="text-xs">Retrieval</Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {run.duration_sec < 60
                          ? `${run.duration_sec.toFixed(0)}s`
                          : `${(run.duration_sec / 60).toFixed(1)}m`}
                      </TableCell>
                      <TableCell>
                        <Button variant="outline" size="sm" onClick={() => handleApply(run)}>
                          Apply
                        </Button>
                      </TableCell>
                    </TableRow>
                    {isExpanded && (
                      <TableRow>
                        <TableCell colSpan={11} className="p-0">
                          <div className="px-4 py-3 bg-muted/30">
                            {runDetail?.per_question_results ? (
                              <div className="space-y-2">
                                <h4 className="text-sm font-medium flex items-center gap-1">
                                  <ListChecks className="h-4 w-4" /> Per-question results (
                                  {runDetail.per_question_results.length})
                                </h4>
                                <div className="overflow-auto max-h-[300px] border rounded-md">
                                  <Table>
                                    <TableHeader>
                                      <TableRow>
                                        <TableHead className="text-xs">#</TableHead>
                                        <TableHead className="text-xs">Question</TableHead>
                                        <TableHead className="text-xs">Hit</TableHead>
                                        <TableHead className="text-xs">MRR</TableHead>
                                        <TableHead className="text-xs">Faith</TableHead>
                                        <TableHead className="text-xs">Rel</TableHead>
                                        <TableHead className="text-xs">Latency</TableHead>
                                      </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                      {runDetail.per_question_results.map((qr, idx) => (
                                        <TableRow key={qr.id ?? idx}>
                                          <TableCell className="text-xs text-muted-foreground">{idx + 1}</TableCell>
                                          <TableCell className="text-xs max-w-[300px] truncate">
                                            {qr.question}
                                          </TableCell>
                                          <TableCell className="text-xs">
                                            {qr.hit_rate != null ? (
                                              <span className={qr.hit_rate >= 0.5 ? "text-green-500" : "text-red-500"}>
                                                {qr.hit_rate >= 0.5 ? "Hit" : "Miss"}
                                              </span>
                                            ) : (
                                              "-"
                                            )}
                                          </TableCell>
                                          <TableCell className="text-xs font-mono">
                                            {qr.mrr?.toFixed(2) ?? "-"}
                                          </TableCell>
                                          <TableCell className="text-xs">
                                            <ScoreBadge score={qr.faithfulness} />
                                          </TableCell>
                                          <TableCell className="text-xs">
                                            <ScoreBadge score={qr.relevancy} />
                                          </TableCell>
                                          <TableCell className="text-xs text-muted-foreground">
                                            {qr.latency_sec != null ? `${qr.latency_sec.toFixed(1)}s` : "-"}
                                          </TableCell>
                                        </TableRow>
                                      ))}
                                    </TableBody>
                                  </Table>
                                </div>
                              </div>
                            ) : (
                              <p className="text-sm text-muted-foreground">
                                No per-question data available for this run.
                              </p>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}

// ─── Tab 5: Trends ────────────────────────────────────────────────────────

function TrendsTab() {
  const { data: historyData, isLoading } = useBenchmarkHistory({ days: 30 });
  const latestRunId = historyData?.points?.length ? historyData.points[historyData.points.length - 1].run_id : null;
  const { data: regressionData } = useRegressionCheck(latestRunId);

  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  const points = historyData?.points || [];

  if (points.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <History className="h-12 w-12 mx-auto mb-3 opacity-30" />
        <p>No benchmark history yet. Run benchmarks to see trends.</p>
      </div>
    );
  }

  // Simple text-based trend display
  const metrics = ["hit_rate", "faithfulness", "relevancy", "composite"];
  const latest = points[points.length - 1];
  const previous = points.length > 1 ? points[points.length - 2] : null;

  return (
    <div className="space-y-4">
      {/* Latest snapshot */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {metrics.map((m) => {
          const current = latest?.metrics?.[m] ?? latest?.metrics?.[`avg_${m}`] ?? null;
          const prev = previous?.metrics?.[m] ?? previous?.metrics?.[`avg_${m}`] ?? null;
          const delta = current !== null && prev !== null ? current - prev : null;
          return (
            <MetricCard
              key={m}
              label={m.replace("avg_", "")}
              value={current !== null ? (typeof current === "number" ? current.toFixed(3) : String(current)) : null}
              icon={TrendingUp}
              color={
                delta !== null
                  ? delta > 0
                    ? "text-green-500"
                    : delta < 0
                      ? "text-red-500"
                      : "text-foreground"
                  : "text-foreground"
              }
            />
          );
        })}
      </div>

      {/* Regression check */}
      {regressionData && regressionData.results.length > 0 && (
        <Card className={regressionData.passed ? "border-green-200" : "border-red-200"}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              {regressionData.passed ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              Regression Check — {regressionData.passed ? "PASSED" : "FAILED"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs">Metric</TableHead>
                    <TableHead className="text-xs">Baseline</TableHead>
                    <TableHead className="text-xs">Current</TableHead>
                    <TableHead className="text-xs">Delta</TableHead>
                    <TableHead className="text-xs">Threshold</TableHead>
                    <TableHead className="text-xs">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {regressionData.results.map((r) => (
                    <TableRow key={r.metric}>
                      <TableCell className="text-xs font-mono">{r.metric}</TableCell>
                      <TableCell className="text-xs">{r.baseline != null ? r.baseline.toFixed(4) : "-"}</TableCell>
                      <TableCell className="text-xs">{r.current != null ? r.current.toFixed(4) : "-"}</TableCell>
                      <TableCell className="text-xs font-mono">
                        {r.delta != null ? (
                          <span className={r.failed ? "text-red-500" : "text-green-500"}>
                            {r.delta > 0 ? "+" : ""}{r.delta.toFixed(4)}
                          </span>
                        ) : (
                          "-"
                        )}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">{r.threshold}</TableCell>
                      <TableCell className="text-xs">
                        {r.failed ? (
                          <Badge variant="destructive" className="text-xs">FAIL</Badge>
                        ) : (
                          <Badge variant="success" className="text-xs">OK</Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* History table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-[400px] overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Hit Rate</TableHead>
                  <TableHead>Faithfulness</TableHead>
                  <TableHead>Relevancy</TableHead>
                  <TableHead>Composite</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>LLM</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {points.map((p) => (
                  <TableRow key={p.run_id}>
                    <TableCell className="font-mono text-xs">#{p.run_id}</TableCell>
                    <TableCell className="text-xs">
                      {p.creation_date ? new Date(p.creation_date).toLocaleDateString() : "-"}
                    </TableCell>
                    <TableCell>
                      {p.metrics.hit_rate != null ? `${(p.metrics.hit_rate * 100).toFixed(0)}%` : "-"}
                    </TableCell>
                    <TableCell>
                      <ScoreBadge score={p.metrics.faithfulness ?? p.metrics.avg_faithfulness} />
                    </TableCell>
                    <TableCell>
                      <ScoreBadge score={p.metrics.relevancy ?? p.metrics.avg_relevancy} />
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {p.metrics.composite?.toFixed(3) ?? "-"}
                    </TableCell>
                    <TableCell className="text-xs">{p.dataset}</TableCell>
                    <TableCell>
                      {p.llm_evaluated ? (
                        <Badge variant="success" className="text-xs">Yes</Badge>
                      ) : (
                        <Badge variant="secondary" className="text-xs">No</Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────

export function AdminBenchmarkPage() {
  const [activeTab, setActiveTab] = useState("questions");
  const [activeSweepId, setActiveSweepId] = useState<number | null>(null);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Benchmark Lab</h1>
          <p className="text-muted-foreground">
            RAG quality evaluation — manage questions, run sweeps, compare results
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="questions">
            <ListChecks className="h-4 w-4 mr-1" /> Questions
          </TabsTrigger>
          <TabsTrigger value="sweep-builder">
            <Settings2 className="h-4 w-4 mr-1" /> Sweep Builder
          </TabsTrigger>
          <TabsTrigger value="progress">
            <BarChart3 className="h-4 w-4 mr-1" /> Sweeps
          </TabsTrigger>
          <TabsTrigger value="leaderboard">
            <TrendingUp className="h-4 w-4 mr-1" /> Leaderboard
          </TabsTrigger>
          <TabsTrigger value="trends">
            <History className="h-4 w-4 mr-1" /> Trends
          </TabsTrigger>
        </TabsList>

        <TabsContent value="questions" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Test Questions</CardTitle>
              <CardDescription>
                Manage the question dataset used for benchmarking. Import/export JSON, edit inline.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <QuestionsTab />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sweep-builder" className="mt-4">
          <SweepBuilderTab
            onSweepCreated={(id) => {
              setActiveSweepId(id);
              setActiveTab("progress");
            }}
          />
        </TabsContent>

        <TabsContent value="progress" className="mt-4">
          <Card>
            <CardContent className="pt-4">
              <SweepProgressTab
                activeSweepId={activeSweepId}
                onSweepSelect={setActiveSweepId}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="leaderboard" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Leaderboard</CardTitle>
              <CardDescription>
                Compare benchmark runs. Select 2-5 runs to diff configs side-by-side. Apply the best config with one click.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <LeaderboardTab />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Trends</CardTitle>
              <CardDescription>
                Metric history over time. Track regressions and improvements.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TrendsTab />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
