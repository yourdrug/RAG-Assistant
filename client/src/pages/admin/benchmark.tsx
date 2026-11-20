"use client";
import {
  BarChart3,
  CheckCircle2,
  Download,
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
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import toast from "react-hot-toast";
import { useSearchParams } from "react-router-dom";
import { apiClient } from "@/shared/api/client";
import {
  useApplyRunConfig,
  useBenchmarkDatasets,
  useBenchmarkQuestions,
  useBenchmarkRuns,
  useCancelSweep,
  useCompareRuns,
  useCreateBenchmarkQuestion,
  useCreateSweep,
  useDeleteBenchmarkQuestion,
  useImportBenchmarkQuestions,
  useJob,
  useResumeSweep,
  useSpaces,
  useSweeps,
  useUpdateBenchmarkQuestion,
} from "@/shared/api/hooks";
import {
  useBenchmarkReviewSummary,
  useExportBenchmarkQuestions,
} from "@/shared/api/hooks/use-benchmark";
import type {
  BenchmarkQuestion,
  BenchmarkQuestionCreate,
  BenchmarkRun,
  LLMReplayMode,
  SweepResponse,
} from "@/shared/api/types";
import { getErrorMessage } from "@/shared/lib/errors";
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
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Progress } from "@/shared/ui/progress";
import { QueryError } from "@/shared/ui/query-error";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Skeleton } from "@/shared/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/tabs";
import { Textarea } from "@/shared/ui/textarea";
import { BenchmarkJudgeModel } from "./benchmark-judge-model";
import { BenchmarkLLMReplay, LLMReplaySummary } from "./benchmark-llm-replay";
import {
  AcceptanceTab,
  DiagnosisCounts,
  DiagnosisTransitions,
  RunQuestionResults,
} from "./benchmark-quality";
import { SourceHintPicker } from "./benchmark-source-hint-picker";
import { SweepProgressDialog } from "./benchmark-sweep-progress";

const BenchmarkTrendsTab = lazy(async () => ({
  default: (await import("./benchmark-trends")).BenchmarkTrendsTab,
}));

// ─── Reusable widgets ─────────────────────────────────────────────────────

function BenchmarkDatasetSelect({
  value,
  onChange,
  allowAll = false,
}: {
  value: string;
  onChange: (value: string) => void;
  allowAll?: boolean;
}) {
  const datasetsQuery = useBenchmarkDatasets();
  const { data: datasets, isLoading } = datasetsQuery;
  const options = [...new Set([...(datasets ?? []), ...(value ? [value] : [])])].sort();
  return (
    <div className="space-y-1 min-w-40">
      <Select
        value={value || "all-datasets"}
        onValueChange={(next) => onChange(next === "all-datasets" ? "" : next)}
      >
        <SelectTrigger aria-label="Dataset">
          <SelectValue placeholder={isLoading ? "Loading datasets…" : "Select dataset"} />
        </SelectTrigger>
        <SelectContent>
          {allowAll && <SelectItem value="all-datasets">All datasets</SelectItem>}
          {options.map((name) => (
            <SelectItem key={name} value={name}>
              {name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {datasetsQuery.isError && (
        <QueryError query={datasetsQuery} fallback="Could not load datasets." />
      )}
    </div>
  );
}

function ScoreBadge({ score, max = 10 }: { score: number | null | undefined; max?: number }) {
  if (score === null || score === undefined) return <Badge variant="secondary">-</Badge>;
  const pct = (score / max) * 100;
  if (pct >= 70)
    return <Badge variant="success">{typeof score === "number" ? score.toFixed(1) : score}</Badge>;
  if (pct >= 40)
    return <Badge variant="warning">{typeof score === "number" ? score.toFixed(1) : score}</Badge>;
  return (
    <Badge variant="destructive">{typeof score === "number" ? score.toFixed(1) : score}</Badge>
  );
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

// ─── Tab 1: Questions ─────────────────────────────────────────────────────

function QuestionsTab() {
  const [search, setSearch] = useState("");
  const [datasetFilter, setDatasetFilter] = useState<string>("");
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const [loadedPage, setLoadedPage] = useState(1);
  const questions = useBenchmarkQuestions({
    search: search || undefined,
    dataset: datasetFilter || undefined,
    limit: pageSize,
    offset: (page - 1) * pageSize,
  });
  const { data, isLoading, isFetching, isPlaceholderData, refetch } = questions;
  const totalPages = data ? Math.max(1, Math.ceil(data.total / pageSize)) : page;
  const displayedPage = isPlaceholderData ? loadedPage : page;

  useEffect(() => {
    if (data && !isPlaceholderData) {
      setLoadedPage(page);
      if (page > totalPages) setPage(totalPages);
    }
  }, [data, isPlaceholderData, page, totalPages]);

  const changePage = (nextPage: number) => {
    setPage(nextPage);
  };

  const pagination = (
    <nav
      aria-label="Questions pagination"
      className="flex flex-wrap items-center justify-between gap-3"
    >
      <span className="text-xs text-muted-foreground" aria-live="polite">
        {data
          ? `${data.total === 0 ? 0 : (displayedPage - 1) * pageSize + 1}–${Math.min(displayedPage * pageSize, data.total)} of ${data.total} questions`
          : "Loading questions…"}
      </span>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={!data || isFetching || page <= 1}
          onClick={() => changePage(page - 1)}
        >
          Previous
        </Button>
        <span className="text-xs text-muted-foreground">
          Page {displayedPage} of {totalPages}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={!data || isFetching || page >= totalPages}
          onClick={() => changePage(page + 1)}
        >
          Next
        </Button>
      </div>
    </nav>
  );
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
    dataset: "belarus_law-main",
    notes: "",
  });

  const tagCatalog = useExportBenchmarkQuestions(undefined, { enabled: dialogOpen });
  const availableTags = useMemo(
    () =>
      [
        ...new Set(
          [
            ...(tagCatalog.data ?? []).flatMap((question) => question.tags ?? []),
            ...(form.tags ?? []),
          ]
            .map((tag) => tag.trim())
            .filter(Boolean),
        ),
      ].sort((a, b) => a.localeCompare(b)),
    [tagCatalog.data, form.tags],
  );

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
      setForm({
        question: "",
        expected_answer: "",
        source_hint: "",
        tags: [],
        dataset: "belarus_law-main",
        notes: "",
      });
      refetch();
    } catch (e) {
      toast.error(getErrorMessage(e, "Could not save the question."));
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this question?")) return;
    try {
      await deleteQuestion.mutateAsync(id);
      toast.success("Deleted");
      refetch();
    } catch (e) {
      toast.error(getErrorMessage(e, "Could not delete the question."));
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
    } catch (e) {
      toast.error(
        e instanceof SyntaxError
          ? "Invalid JSON"
          : getErrorMessage(e, "Could not import the questions."),
      );
    }
  };

  const handleExport = async () => {
    try {
      const { data } = await apiClient.get(
        `/admin/benchmark/questions/export${datasetFilter ? `?dataset=${datasetFilter}` : ""}`,
      );
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `benchmark-questions${datasetFilter ? `-${datasetFilter}` : ""}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Exported");
    } catch (error) {
      toast.error(getErrorMessage(error, "Could not export the benchmark questions."));
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
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          className="max-w-sm"
        />
        <BenchmarkDatasetSelect
          value={datasetFilter}
          onChange={(value) => {
            setDatasetFilter(value);
            setPage(1);
          }}
          allowAll
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
            setForm({
              question: "",
              expected_answer: "",
              source_hint: "",
              tags: [],
              dataset: "belarus_law-main",
              notes: "",
            });
            setDialogOpen(true);
          }}
        >
          <Plus className="h-4 w-4 mr-1" /> Add Question
        </Button>
      </div>

      {pagination}
      {questions.isError && (
        <QueryError query={questions} fallback="Could not load benchmark questions." />
      )}
      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-36 w-full rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="space-y-2 [overflow-anchor:none]" aria-busy={isFetching}>
          {data?.questions.map((q, idx) => (
            <article key={q.id} className="min-w-0 overflow-hidden rounded-lg border bg-card">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/30 px-3 py-1.5">
                <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs">
                  <span className="font-mono text-muted-foreground">
                    #{(displayedPage - 1) * pageSize + idx + 1}
                  </span>
                  <Badge variant="outline" className="max-w-full whitespace-normal break-words">
                    {q.dataset}
                  </Badge>
                  <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                    {q.is_active ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5" />
                    )}
                    {q.is_active ? "Active" : "Inactive"}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2"
                    onClick={() => openEdit(q)}
                  >
                    Edit
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                    aria-label={`Delete question ${(displayedPage - 1) * pageSize + idx + 1}`}
                    onClick={() => handleDelete(q.id)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="space-y-2 p-3">
                <div className="space-y-1">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Question
                  </p>
                  <h3 className="whitespace-pre-wrap break-words text-sm font-medium leading-snug [overflow-wrap:anywhere]">
                    {q.question}
                  </h3>
                </div>
                <div className="space-y-1 rounded-md border bg-muted/30 p-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Expected answer
                  </p>
                  <p className="whitespace-pre-wrap break-words text-sm leading-snug [overflow-wrap:anywhere]">
                    {q.expected_answer || (
                      <span className="text-muted-foreground">No expected answer</span>
                    )}
                  </p>
                </div>
                {q.source_hint && (
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <p className="font-medium">Source hint</p>
                    <p className="whitespace-pre-wrap break-words leading-relaxed [overflow-wrap:anywhere]">
                      {q.source_hint}
                    </p>
                  </div>
                )}
                {!!q.tags?.length && (
                  <div className="flex flex-wrap gap-1.5">
                    {q.tags.map((t) => (
                      <Badge
                        key={t}
                        variant="secondary"
                        className="max-w-full whitespace-normal break-words text-xs"
                      >
                        {t}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
          {data?.questions.length === 0 && (
            <div className="rounded-xl border border-dashed py-12 text-center text-sm text-muted-foreground">
              No questions yet. Add your first question above.
            </div>
          )}
        </div>
      )}
      {pagination}

      {/* Create/Edit Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="flex max-h-[90dvh] w-[95vw] max-w-4xl flex-col gap-0 overflow-hidden p-0">
          <DialogHeader className="shrink-0 border-b px-6 py-6 text-left sm:px-8">
            <DialogTitle className="pr-8 text-2xl font-semibold tracking-tight">
              {editQuestion ? "Edit question" : "Add question"}
            </DialogTitle>
            <DialogDescription className="pt-1 leading-relaxed">
              Define the question, reference answer and sources used to evaluate a benchmark
              response.
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 flex-1 space-y-7 overflow-y-auto px-6 py-6 sm:px-8">
            <section className="space-y-5" aria-label="Question and expected answer">
              <div className="flex flex-col gap-2.5">
                <Label htmlFor="benchmark-question">
                  Question <span className="text-destructive">*</span>
                </Label>
                <Textarea
                  id="benchmark-question"
                  className="min-h-28 resize-y px-3 py-3 text-sm leading-relaxed"
                  value={form.question}
                  onChange={(e) => setForm({ ...form, question: e.target.value })}
                  placeholder="Enter the question as a user would ask it…"
                />
              </div>
              <div className="flex flex-col gap-2.5">
                <Label htmlFor="benchmark-expected-answer">Expected answer</Label>
                <Textarea
                  id="benchmark-expected-answer"
                  aria-describedby="benchmark-answer-help"
                  className="min-h-40 resize-y px-3 py-3 text-sm leading-relaxed"
                  value={form.expected_answer || ""}
                  onChange={(e) => setForm({ ...form, expected_answer: e.target.value || null })}
                  placeholder="Describe the correct answer, including required facts and calculations…"
                />
                <p
                  id="benchmark-answer-help"
                  className="text-xs leading-relaxed text-muted-foreground"
                >
                  Reference answer for correctness scoring. Include the details a successful
                  response should cover.
                </p>
              </div>
            </section>
            <section
              className="space-y-3 rounded-xl border bg-muted/20 p-4 sm:p-5"
              aria-label="Expected sources"
            >
              <h3 className="text-sm font-semibold">Expected sources</h3>
              <SourceHintPicker
                value={form.source_hint || null}
                onChange={(v) => setForm({ ...form, source_hint: v })}
              />
            </section>
            <section className="space-y-5 border-t pt-6" aria-label="Question details">
              <h3 className="text-sm font-semibold">Question details</h3>
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
                <div className="flex min-w-0 flex-col gap-2.5">
                  <Label htmlFor="benchmark-dataset">Dataset</Label>
                  <Input
                    id="benchmark-dataset"
                    value={form.dataset}
                    onChange={(e) => setForm({ ...form, dataset: e.target.value })}
                  />
                </div>
                <div className="flex min-w-0 flex-col gap-2.5">
                  <Label htmlFor="benchmark-tags">Tags</Label>
                  <Input
                    id="benchmark-tags"
                    aria-describedby="benchmark-tags-help"
                    value={(form.tags || []).join(", ")}
                    onChange={(e) =>
                      setForm({
                        ...form,
                        tags: e.target.value
                          .split(",")
                          .map((tag) => tag.trim())
                          .filter(Boolean),
                      })
                    }
                    placeholder="legal, narrow, regression"
                  />
                  <p id="benchmark-tags-help" className="text-xs text-muted-foreground">
                    Separate tags with commas.
                  </p>
                </div>
              </div>
              <div className="space-y-3">
                <p className="text-sm font-medium">Available tags</p>
                <p className="text-xs text-muted-foreground">
                  Tags used across benchmark questions. Click to select or remove; you can also
                  enter new tags above.
                </p>
                {tagCatalog.isFetching && !tagCatalog.data && (
                  <p role="status" className="text-xs text-muted-foreground">
                    Loading tags…
                  </p>
                )}
                {tagCatalog.isError && (
                  <div className="flex items-center gap-2 text-xs">
                    <p role="alert" className="text-destructive">
                      Could not load available tags.
                    </p>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => tagCatalog.refetch()}
                    >
                      Retry
                    </Button>
                  </div>
                )}
                {!tagCatalog.isPending && !tagCatalog.isError && availableTags.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    No tags yet. Enter the first tag above.
                  </p>
                )}
                {availableTags.length > 0 && (
                  <div
                    className="flex max-h-40 flex-wrap gap-2 overflow-y-auto rounded-lg border bg-muted/20 p-3"
                    role="group"
                    aria-label="Available benchmark tags"
                  >
                    {availableTags.map((tag) => {
                      const selected = form.tags?.includes(tag) ?? false;
                      return (
                        <button
                          key={tag}
                          type="button"
                          aria-pressed={selected}
                          onClick={() =>
                            setForm((previous) => ({
                              ...previous,
                              tags: selected
                                ? (previous.tags ?? []).filter((value) => value !== tag)
                                : [...new Set([...(previous.tags ?? []), tag])],
                            }))
                          }
                          className={`max-w-full rounded-full border px-3 py-1.5 text-left text-xs leading-relaxed break-words transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${selected ? "border-primary bg-primary text-primary-foreground" : "border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground"}`}
                        >
                          {selected && (
                            <span aria-hidden="true" className="mr-1">
                              ✓
                            </span>
                          )}
                          {tag}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-2.5">
                <Label htmlFor="benchmark-notes">Internal notes</Label>
                <Textarea
                  id="benchmark-notes"
                  className="min-h-20 resize-y leading-relaxed"
                  value={form.notes || ""}
                  onChange={(e) => setForm({ ...form, notes: e.target.value || null })}
                  placeholder="Context for reviewers or a link to the original issue…"
                />
              </div>
            </section>
          </div>
          <DialogFooter className="shrink-0 gap-2 border-t bg-muted/20 px-6 py-4 sm:items-center sm:px-8">
            <Button variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={createQuestion.isPending || updateQuestion.isPending}
            >
              {createQuestion.isPending || updateQuestion.isPending
                ? "Saving…"
                : editQuestion
                  ? "Save changes"
                  : "Create question"}
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
            onDragOver={(e) => {
              e.preventDefault();
              e.currentTarget.classList.add("border-primary", "bg-primary/5");
            }}
            onDragLeave={(e) => {
              e.currentTarget.classList.remove("border-primary", "bg-primary/5");
            }}
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
            <Button variant="outline" onClick={() => setImportDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleImport}
              disabled={importQuestions.isPending || !importText.trim()}
            >
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
  const [strategy, setStrategy] = useState<"grid" | "random" | "optuna_tpe">("grid");
  const [topNLlm, setTopNLlm] = useState(3);
  const [evaluationMode, setEvaluationMode] = useState<"fast" | "full">("fast");
  const [judgeModel, setJudgeModel] = useState<string | null>(null);
  const [llmReplayMode, setLLMReplayMode] = useState<LLMReplayMode>("off");
  const [dataset, setDataset] = useState("belarus_law-main");
  const { data: spaces } = useSpaces();
  const [spaceId, setSpaceId] = useState<string>("");
  const selectedSpaceId = spaceId
    ? Number(spaceId)
    : spaces?.length === 1
      ? spaces[0].id
      : undefined;

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
    hit_rate: 0.2,
    mrr: 0,
    correctness: 0.3,
    faithfulness: 0.4,
    relevancy: 0.1,
    fragment_recall_at_k: 0,
    fragment_mrr: 0,
    retrieval_fact_coverage: 0,
    context_fragment_recall: 0,
    context_fact_coverage: 0,
  });

  const estimatedConfigs = useMemo(() => {
    if (strategy !== "grid") return 50;
    let total = 1;
    for (const [, cfg] of Object.entries(params)) {
      if (!cfg.enabled) continue;
      const values = cfg.values
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      total *= values.length || 1;
    }
    return total;
  }, [params, strategy]);

  const handleStart = async () => {
    if ((evaluationMode === "full" || topNLlm > 0) && !judgeModel?.trim()) {
      toast.error("Select a judge model");
      return;
    }
    const searchSpace: Record<string, any> = {};
    for (const [key, cfg] of Object.entries(params)) {
      if (!cfg.enabled) continue;
      const values = cfg.values
        .split(",")
        .map((s) => parseFloat(s.trim()))
        .filter((n) => !isNaN(n));
      if (values.length > 0) {
        searchSpace[key] = { values };
      }
    }

    if (Object.keys(searchSpace).length === 0) {
      toast.error("Enable at least one parameter");
      return;
    }

    if (!selectedSpaceId) {
      toast.error("Select a knowledge space");
      return;
    }

    try {
      const result = await createSweep.mutateAsync({
        space_id: selectedSpaceId,
        strategy,
        evaluation_mode: evaluationMode,
        search_space: searchSpace,
        objective_weights: weights,
        dataset,
        top_n_llm: topNLlm,
        judge_model: judgeModel?.trim() || null,
        llm_replay_mode: llmReplayMode,
      });
      toast.success(`Sweep #${result.id} started`);
      onSweepCreated(result.id);
    } catch (e) {
      toast.error(getErrorMessage(e, "Failed to create sweep"));
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
          <div className="flex gap-3 mb-2">
            <Button
              variant={evaluationMode === "fast" ? "default" : "outline"}
              onClick={() => setEvaluationMode("fast")}
            >
              Fast
            </Button>
            <Button
              variant={evaluationMode === "full" ? "default" : "outline"}
              onClick={() => {
                setEvaluationMode("full");
                setStrategy("grid");
              }}
            >
              Full — all combinations
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mb-3">
            {evaluationMode === "full"
              ? "Run RAG and judge for every grid combination. Top-N does not limit this mode."
              : "Evaluate a shortlist using retrieval quality, fragment completeness, MRR and parameter diversity."}
          </p>
          <div className="flex gap-3">
            {(["grid", "random", "optuna_tpe"] as const).map((s) => (
              <Button
                key={s}
                variant={strategy === s ? "default" : "outline"}
                onClick={() => setStrategy(s)}
                disabled={evaluationMode === "full" && s !== "grid"}
              >
                {s}
              </Button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {strategy === "grid" && "Cartesian product of all values. Best for 2-4 parameters."}
            {strategy === "random" &&
              "50 random points from the search space. Good for 5+ parameters."}
            {strategy === "optuna_tpe" && "50 Optuna TPE trials guided by retrieval scores."}
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
                      onChange={(e) =>
                        setParams({ ...params, [key]: { ...cfg, enabled: e.target.checked } })
                      }
                      className="rounded"
                    />
                    {key}
                  </label>
                  <Badge variant="secondary" className="text-xs">
                    cheap
                  </Badge>
                </div>
                {cfg.enabled && (
                  <Input
                    value={cfg.values}
                    onChange={(e) =>
                      setParams({ ...params, [key]: { ...cfg, values: e.target.value } })
                    }
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
            <p className="text-xs text-muted-foreground">
              Metrics are normalized to 0–1. Influence is weight / sum of weights. Final selection
              uses only fully evaluated configurations. Fast mode with Top-N = 0 uses retrieval
              only. Fragment and context metrics require dataset annotations; missing evidence is
              unavailable.
            </p>
            {Object.entries(weights).map(([key, val]) => (
              <div key={key} className="flex items-center gap-3">
                <Label className="w-44 text-xs break-words">{key}</Label>
                <Input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={val}
                  onChange={(e) =>
                    setWeights({ ...weights, [key]: parseFloat(e.target.value) || 0 })
                  }
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
              <BenchmarkDatasetSelect value={dataset} onChange={setDataset} />
            </div>
            <div className="flex items-center gap-3">
              <Label className="w-28 text-xs">Knowledge space</Label>
              <Select value={selectedSpaceId?.toString() ?? ""} onValueChange={setSpaceId}>
                <SelectTrigger className="w-64">
                  <SelectValue placeholder="Select space" />
                </SelectTrigger>
                <SelectContent>
                  {spaces?.map((space) => (
                    <SelectItem key={space.id} value={String(space.id)}>
                      {space.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {evaluationMode === "fast" && (
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
            )}
            <BenchmarkJudgeModel
              value={judgeModel}
              onChange={setJudgeModel}
              disabled={evaluationMode === "fast" && topNLlm === 0}
            />
            <BenchmarkLLMReplay
              value={llmReplayMode}
              onChange={setLLMReplayMode}
              disabled={evaluationMode === "fast" && topNLlm === 0}
            />
            <div className="mt-4 p-3 bg-muted rounded-md text-sm">
              <div className="font-medium">Estimated combinations</div>
              <div className="text-2xl font-bold font-mono mt-1">{estimatedConfigs}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {strategy === "grid"
                  ? `${evaluationMode === "full" ? estimatedConfigs : Math.min(topNLlm, estimatedConfigs)} full RAG + judge evaluations (one per dataset question; several judge calls per answer)`
                  : `${estimatedConfigs} random evaluations`}
              </div>
            </div>
            <Button
              className="w-full mt-2"
              onClick={handleStart}
              disabled={
                createSweep.isPending ||
                ((evaluationMode === "full" || topNLlm > 0) && !judgeModel?.trim())
              }
            >
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

function SweepJobError({ jobId }: { jobId: number | null }) {
  const { data: job, isLoading, isError } = useJob(jobId);
  return (
    <details
      className="mt-1 max-w-md text-xs text-destructive"
      onClick={(e) => e.stopPropagation()}
    >
      <summary className="cursor-pointer">
        {job?.error_message?.match(/Оценено \d+ из \d+, ошибок \d+/)?.[0] || "Error details"}
      </summary>
      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words">
        {jobId === null ? (
          <a href="/admin/jobs" className="underline">
            This older sweep has no linked job. View Jobs.
          </a>
        ) : (
          job?.error_message ||
          (isLoading
            ? "Loading error…"
            : isError
              ? "Could not load job error"
              : "No error details recorded")
        )}
      </pre>
    </details>
  );
}

function SweepProgressTab({
  activeSweepId,
  onSweepSelect,
}: {
  activeSweepId: number | null;
  onSweepSelect: (id: number) => void;
}) {
  const sweeps = useSweeps();
  const { data: sweepsData, isLoading, refetch } = sweeps;
  const cancelSweep = useCancelSweep();
  const resumeSweep = useResumeSweep();
  const restartSweep = useCreateSweep();
  const [detailSweep, setDetailSweep] = useState<SweepResponse | null>(null);

  function selectSweep(sweep: SweepResponse) {
    onSweepSelect(sweep.id);
    if (sweep.status === "pending" || sweep.status === "running") setDetailSweep(sweep);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Sweeps</h3>
          <p className="text-sm text-muted-foreground">
            Click an active sweep for live execution details.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-1" /> Refresh
        </Button>
      </div>

      {sweeps.isError && <QueryError query={sweeps} fallback="Could not load sweeps." />}
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
                <TableHead>Evaluation</TableHead>
                <TableHead>Progress (current phase)</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead>Judge model</TableHead>
                <TableHead>Best Run</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sweepsData?.sweeps.map((s) => (
                <TableRow
                  key={s.id}
                  className={
                    activeSweepId === s.id
                      ? "cursor-pointer bg-muted/50"
                      : "cursor-pointer hover:bg-muted/30"
                  }
                  onClick={() => selectSweep(s)}
                >
                  <TableCell className="font-mono">
                    {s.status === "pending" || s.status === "running" ? (
                      <button
                        type="button"
                        className="underline underline-offset-4"
                        aria-label={`View progress for sweep #${s.id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          selectSweep(s);
                        }}
                      >
                        #{s.id}
                      </button>
                    ) : (
                      `#${s.id}`
                    )}
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={s.status} />
                    {s.status === "failed" && <SweepJobError jobId={s.job_id ?? null} />}
                  </TableCell>
                  <TableCell className="text-xs">{s.strategy}</TableCell>
                  <TableCell className="text-xs">{s.evaluation_mode}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Progress
                        value={
                          s.total_configs > 0 ? (s.evaluated_configs / s.total_configs) * 100 : 0
                        }
                        className="w-24 h-2"
                      />
                      <span className="text-xs text-muted-foreground">
                        {s.status === "running" && s.total_configs === 0
                          ? "Preparing…"
                          : `${s.evaluated_configs}/${s.total_configs}`}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-xs">{s.dataset}</TableCell>
                  <TableCell className="text-xs">
                    {s.evaluation_mode === "fast" && s.top_n_llm === 0
                      ? "-"
                      : s.judge_model || "Default"}
                    {s.llm_replay_mode && s.llm_replay_mode !== "off" && (
                      <div className="text-muted-foreground">replay: {s.llm_replay_mode}</div>
                    )}
                  </TableCell>
                  <TableCell className="text-xs font-mono">
                    {s.best_run_id ? `#${s.best_run_id}` : "-"}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {(s.status === "failed" || s.status === "cancelled") && (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={resumeSweep.isPending}
                          onClick={async (e) => {
                            e.stopPropagation();
                            try {
                              await resumeSweep.mutateAsync(s.id);
                              toast.success("Sweep resumed; only missing scores will be evaluated");
                              refetch();
                            } catch (error) {
                              toast.error(getErrorMessage(error, "Could not resume the sweep."));
                            }
                          }}
                        >
                          Resume
                        </Button>
                      )}
                      {s.status !== "pending" && s.status !== "running" && (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={restartSweep.isPending}
                          onClick={async (e) => {
                            e.stopPropagation();
                            try {
                              const created = await restartSweep.mutateAsync({
                                strategy: s.strategy,
                                search_space: s.search_space,
                                objective_weights: s.objective_weights,
                                dataset: s.dataset,
                                space_id: s.space_id,
                                top_n_llm: s.top_n_llm,
                                judge_model: s.judge_model,
                                evaluation_mode: s.evaluation_mode,
                                llm_replay_mode: s.llm_replay_mode,
                              });
                              onSweepSelect(created.id);
                              toast.success(
                                `Sweep #${created.id} restarted with the same settings`,
                              );
                              refetch();
                            } catch (error) {
                              toast.error(getErrorMessage(error, "Could not restart the sweep."));
                            }
                          }}
                        >
                          <RefreshCw className="h-4 w-4 mr-1" />
                          Restart with same settings
                        </Button>
                      )}
                      {(s.status === "pending" || s.status === "running") && (
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={cancelSweep.isPending && cancelSweep.variables === s.id}
                          onClick={async (e) => {
                            e.stopPropagation();
                            try {
                              await cancelSweep.mutateAsync(s.id);
                              toast.success("Sweep cancelled");
                            } catch (error) {
                              toast.error(getErrorMessage(error, "Could not cancel the sweep."));
                            }
                          }}
                        >
                          {cancelSweep.isPending && cancelSweep.variables === s.id
                            ? "Cancelling…"
                            : "Cancel"}
                        </Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {sweepsData?.sweeps.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} className="text-center py-8 text-muted-foreground">
                    No sweeps yet. Create one in the Sweep Builder tab.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      )}
      {detailSweep && (
        <SweepProgressDialog
          key={detailSweep.id}
          initialSweep={detailSweep}
          onClose={() => setDetailSweep(null)}
        />
      )}
    </div>
  );
}

// ─── Tab 4: Leaderboard ───────────────────────────────────────────────────

function LeaderboardTab() {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [reviewRun, setReviewRun] = useState<BenchmarkRun | null>(null);
  const reviewTrigger = useRef<HTMLButtonElement>(null);
  const [page, setPage] = useState(1);
  const pageSize = 25;
  const runs = useBenchmarkRuns({ limit: pageSize, offset: (page - 1) * pageSize });
  const totalPages = Math.max(1, Math.ceil((runs.data?.total ?? 0) / pageSize));
  const { data: runsData, isLoading } = runs;
  const applyConfig = useApplyRunConfig();
  const comparison = useCompareRuns(selectedIds);
  const { data: compareData } = comparison;

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]));
  };

  const handleApply = async (run: BenchmarkRun) => {
    if (!confirm(`Apply config from run #${run.id}? This will update live settings.`)) return;
    try {
      const result = await applyConfig.mutateAsync(run.id);
      if (result.failed.length > 0) {
        toast(
          `Applied ${result.applied}, failed ${result.failed.length}: ${result.failed.map((f) => f.key).join(", ")}`,
          { icon: "\u26a0\ufe0f" },
        );
      } else {
        toast.success(`Applied ${result.applied} config parameters: ${result.keys.join(", ")}`);
      }
    } catch (e) {
      toast.error(getErrorMessage(e, "Failed to apply"));
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground" aria-live="polite">
          {selectedIds.length} of 5 runs selected · Select 2 or more to compare
        </p>
        <div className="flex gap-2">
          {selectedIds.length > 0 && (
            <Button variant="ghost" onClick={() => setSelectedIds([])}>
              Clear selection
            </Button>
          )}
          <Button variant="outline" onClick={() => void runs.refetch()} disabled={runs.isFetching}>
            <RefreshCw aria-hidden="true" /> Refresh runs
          </Button>
        </div>
      </div>
      {selectedIds.length >= 2 && (
        <Card className="border-primary">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <GitCompare className="h-4 w-4" /> Comparing {selectedIds.length} runs
            </CardTitle>
          </CardHeader>
          <CardContent>
            {comparison.isPending && <p role="status">Loading comparison…</p>}
            <div className="overflow-auto max-h-80">
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
            <DiagnosisTransitions transitions={compareData?.diagnosis_transitions} />
          </CardContent>
        </Card>
      )}

      {runs.isError && <QueryError query={runs} fallback="Could not load benchmark runs." />}
      {comparison.isError && <QueryError query={comparison} fallback="Could not compare runs." />}
      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Compare</TableHead>
                <TableHead>Run</TableHead>
                <TableHead>Config</TableHead>
                <TableHead>Hit Rate</TableHead>
                <TableHead>Faithfulness</TableHead>
                <TableHead>Relevancy</TableHead>
                <TableHead>Composite</TableHead>
                <TableHead>LLM</TableHead>
                <TableHead>Time</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runsData?.runs.length === 0 && (
                <TableRow>
                  <TableCell colSpan={10} className="py-8 text-center text-muted-foreground">
                    {page === 1 ? "No benchmark runs yet." : "No runs on this page."}
                  </TableCell>
                </TableRow>
              )}
              {runsData?.runs.map((run) => (
                <TableRow
                  key={run.id}
                  data-state={selectedIds.includes(run.id) ? "selected" : undefined}
                >
                  <TableCell>
                    <input
                      type="checkbox"
                      aria-label={`Compare run #${run.id}`}
                      checked={selectedIds.includes(run.id)}
                      disabled={selectedIds.length >= 5 && !selectedIds.includes(run.id)}
                      onChange={() => toggleSelect(run.id)}
                      className="size-4 align-middle accent-primary"
                    />
                  </TableCell>
                  <TableCell className="whitespace-nowrap">
                    <p className="font-semibold">#{run.id}</p>
                    {run.dataset && <p className="text-sm text-muted-foreground">{run.dataset}</p>}
                    <Button
                      size="sm"
                      className="mt-2"
                      aria-label={`Review questions for run #${run.id}`}
                      onClick={(event) => {
                        reviewTrigger.current = event.currentTarget;
                        setReviewRun(run);
                      }}
                    >
                      Review questions
                    </Button>
                  </TableCell>
                  <TableCell className="text-xs max-w-[200px]">
                    <div className="truncate font-mono">
                      {Object.entries(run.config_json)
                        .slice(0, 3)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(" ")}
                    </div>
                  </TableCell>
                  <TableCell>
                    {run.summary_metrics.hit_rate !== null &&
                    run.summary_metrics.hit_rate !== undefined ? (
                      <span
                        className={
                          run.summary_metrics.hit_rate >= 0.5 ? "text-green-500" : "text-red-500"
                        }
                      >
                        {(run.summary_metrics.hit_rate * 100).toFixed(0)}%
                      </span>
                    ) : (
                      "-"
                    )}
                  </TableCell>
                  <TableCell>
                    <ScoreBadge
                      score={
                        run.summary_metrics.faithfulness ?? run.summary_metrics.avg_faithfulness
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <ScoreBadge
                      score={run.summary_metrics.relevancy ?? run.summary_metrics.avg_relevancy}
                    />
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {run.summary_metrics.composite?.toFixed(3) ?? "-"}
                    {run.summary_metrics.evaluation_mode && (
                      <div className="text-muted-foreground">
                        {run.summary_metrics.evaluation_mode}:{" "}
                        {run.summary_metrics.evaluated_config_count}/
                        {run.summary_metrics.search_config_count} evaluated
                        {run.summary_metrics.scored_config_count != null && (
                          <span> ({run.summary_metrics.scored_config_count} scored)</span>
                        )}
                      </div>
                    )}
                  </TableCell>
                  <TableCell>
                    {run.llm_evaluated ? (
                      <Badge variant="success" className="text-xs">
                        LLM
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">
                        Retrieval
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {run.duration_sec < 60
                      ? `${run.duration_sec.toFixed(0)}s`
                      : `${(run.duration_sec / 60).toFixed(1)}m`}
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button variant="outline" size="sm" onClick={() => handleApply(run)}>
                        Apply
                      </Button>
                      {(["csv", "json"] as const).map((format) => (
                        <Button
                          key={format}
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            try {
                              const response = await apiClient.get(
                                `/benchmark/results/${run.id}/export?format=${format}`,
                                { responseType: "blob" },
                              );
                              const url = URL.createObjectURL(response.data);
                              const anchor = document.createElement("a");
                              anchor.href = url;
                              anchor.download = `benchmark-${run.id}.${format}`;
                              anchor.click();
                              URL.revokeObjectURL(url);
                            } catch (error) {
                              toast.error(
                                getErrorMessage(error, "Could not export the benchmark run."),
                              );
                            }
                          }}
                        >
                          {format.toUpperCase()}
                        </Button>
                      ))}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
      {(runsData || page > 1) && (
        <nav
          aria-label="Run pagination"
          className="flex flex-wrap items-center justify-between gap-3"
        >
          <span className="text-sm text-muted-foreground" aria-live="polite">
            {runsData
              ? `${runsData.runs.length === 0 ? 0 : (page - 1) * pageSize + 1}–${runsData.runs.length === 0 ? 0 : (page - 1) * pageSize + runsData.runs.length} of ${runsData.total} runs`
              : `Loading page ${page}…`}
          </span>
          <div className="flex gap-2">
            <Button variant="outline" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              Previous
            </Button>
            <Button
              variant="outline"
              disabled={!runsData || page >= totalPages}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
        </nav>
      )}
      <Dialog open={reviewRun !== null} onOpenChange={(open) => !open && setReviewRun(null)}>
        {reviewRun && (
          <DialogContent
            className="flex max-h-[95dvh] w-[calc(100%_-_1rem)] max-w-6xl flex-col gap-0 overflow-hidden p-0 sm:w-[calc(100%_-_3rem)]"
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              reviewTrigger.current?.focus();
            }}
          >
            <DialogHeader className="shrink-0 border-b px-4 py-4 pr-10 text-left sm:px-6 sm:pr-12">
              <DialogTitle className="text-xl">Review questions · Run #{reviewRun.id}</DialogTitle>
              <DialogDescription>
                {reviewRun.dataset || "Benchmark run"} ·{" "}
                {reviewRun.llm_evaluated ? "Answers evaluated" : "Retrieval only"}
              </DialogDescription>
            </DialogHeader>
            <div className="min-h-0 overflow-y-auto space-y-5 px-4 py-5 sm:px-6">
              <RunQuestionResults key={reviewRun.id} runId={reviewRun.id} />
              <RunReviewDiagnostics key={reviewRun.id} runId={reviewRun.id} />
            </div>
          </DialogContent>
        )}
      </Dialog>
    </div>
  );
}

function RunReviewDiagnostics({ runId }: { runId: number }) {
  const [open, setOpen] = useState(false);
  const query = useBenchmarkReviewSummary(open ? runId : null);
  const metrics = query.data?.summary_metrics;
  return (
    <details
      className="rounded-lg border p-4"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer font-semibold">Run diagnostics and token usage</summary>
      {open && query.isError && (
        <QueryError query={query} fallback="Could not load the full run summary." />
      )}
      {open && query.isPending && <Skeleton className="mt-3 h-12 w-full" />}
      {open && metrics && (
        <div className="mt-3 space-y-2">
          {metrics.judge_evaluated_count != null && (
            <p className="text-sm text-muted-foreground">
              Judge evaluated {metrics.judge_evaluated_count} of {metrics.total_questions}, errors{" "}
              {metrics.judge_error_count ?? 0}
            </p>
          )}
          {metrics.total_judge_input_tokens != null && (
            <p className="text-sm text-muted-foreground">
              This configuration: generation{" "}
              {Number(metrics.total_input_tokens || 0).toLocaleString()} input /{" "}
              {Number(metrics.total_output_tokens || 0).toLocaleString()} output tokens; judge{" "}
              {Number(metrics.total_judge_input_tokens || 0).toLocaleString()} input /{" "}
              {Number(metrics.total_judge_output_tokens || 0).toLocaleString()} output tokens (
              {metrics.total_judge_calls} calls). RAG auxiliary calls are excluded. Cost estimate
              covers generation only.
            </p>
          )}
          <LLMReplaySummary stats={metrics.llm_replay} />
          {metrics.retrieval_labelled_count != null && (
            <p className="text-sm text-muted-foreground">
              Annotated cases: {metrics.retrieval_annotated_count}; source-only cases:{" "}
              {metrics.retrieval_source_only_count}; unavailable retrieval evidence:{" "}
              {metrics.retrieval_unavailable_count}.
            </p>
          )}
          <div className="mb-2">
            <DiagnosisCounts counts={metrics.diagnosis_counts} />
          </div>
        </div>
      )}
    </details>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────

export function AdminBenchmarkPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTab = searchParams.get("tab");
  const activeTab =
    requestedTab &&
    ["questions", "sweep-builder", "progress", "leaderboard", "trends", "acceptance"].includes(
      requestedTab,
    )
      ? requestedTab
      : "questions";
  const sweepId = Number(searchParams.get("sweep"));
  const activeSweepId = Number.isSafeInteger(sweepId) && sweepId > 0 ? sweepId : null;

  function setActiveTab(tab: string) {
    setSearchParams((current) => {
      current.set("tab", tab);
      return current;
    });
  }

  function setActiveSweepId(id: number) {
    setSearchParams((current) => {
      current.set("sweep", String(id));
      return current;
    });
  }

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
          <TabsTrigger value="acceptance">
            <Target className="h-4 w-4 mr-1" /> Acceptance
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
              setSearchParams((current) => {
                current.set("sweep", String(id));
                current.set("tab", "progress");
                return current;
              });
            }}
          />
        </TabsContent>

        <TabsContent value="progress" className="mt-4">
          <Card>
            <CardContent className="pt-4">
              <SweepProgressTab activeSweepId={activeSweepId} onSweepSelect={setActiveSweepId} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="leaderboard" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Leaderboard</CardTitle>
              <CardDescription>
                Review answers and evidence from a run, compare configurations, or apply a
                configuration.
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
                See where evidence is lost between search, context and answer. Compare runs on the
                same dataset.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Suspense fallback={<Skeleton className="h-56 w-full" />}>
                <BenchmarkTrendsTab />
              </Suspense>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="acceptance" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Acceptance</CardTitle>
              <CardDescription>
                Per-dataset thresholds, question composition, run-to-run noise and judge calibration
                used by the regression check.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AcceptanceTab />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
