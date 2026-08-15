"use client";
import {
  BarChart3,
  CheckCircle2,
  Clock,
  FileText,
  Play,
  RefreshCw,
  TrendingUp,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import { useBenchmark, useBenchmarkResult, useBenchmarkResults } from "@/shared/api/hooks";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
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
        <div className={`text-2xl font-bold ${color}`}>
          {value !== null && value !== undefined ? value : "-"}
        </div>
      </CardContent>
    </Card>
  );
}

function ScoreBadge({ score, max = 10 }: { score: number | null; max?: number }) {
  if (score === null) return <Badge variant="secondary">-</Badge>;
  const pct = (score / max) * 100;
  if (pct >= 70) return <Badge variant="success">{score.toFixed(1)}</Badge>;
  if (pct >= 40) return <Badge variant="warning">{score.toFixed(1)}</Badge>;
  return <Badge variant="destructive">{score.toFixed(1)}</Badge>;
}

function RunBenchmarkForm({ onSuccess }: { onSuccess: () => void }) {
  const benchmark = useBenchmark();
  const [questionsPath, setQuestionsPath] = useState("");
  const [topK, setTopK] = useState("");
  const [judgeModel, setJudgeModel] = useState("");

  const handleSubmit = () => {
    const payload: Record<string, unknown> = {};
    if (questionsPath.trim()) payload.questions_path = questionsPath.trim();
    if (topK.trim()) payload.top_k = parseInt(topK, 10);
    if (judgeModel.trim()) payload.judge_model = judgeModel.trim();

    benchmark.mutate(payload as any, {
      onSuccess: () => {
        toast.success("Benchmark started! Track progress in the Jobs page.");
        setQuestionsPath("");
        setTopK("");
        setJudgeModel("");
        onSuccess();
      },
      onError: (err: any) => {
        toast.error(err.response?.data?.detail || "Failed to start benchmark");
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" /> Run Benchmark
        </CardTitle>
        <CardDescription>
          Start a new RAG quality benchmark. Runs as a background job — monitor progress in the
          Jobs page.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="questions-path">Questions File (optional)</Label>
            <Input
              id="questions-path"
              placeholder="Default: test_questions.json"
              value={questionsPath}
              onChange={(e) => setQuestionsPath(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="top-k">Top K (optional)</Label>
            <Input
              id="top-k"
              type="number"
              placeholder="Default: from settings"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="judge-model">Judge Model (optional)</Label>
            <Input
              id="judge-model"
              placeholder="Default: RAG model"
              value={judgeModel}
              onChange={(e) => setJudgeModel(e.target.value)}
            />
          </div>
        </div>
        <Button
          className="mt-4"
          onClick={handleSubmit}
          disabled={benchmark.isPending}
        >
          {benchmark.isPending ? (
            <>
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> Starting...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" /> Start Benchmark
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

function ResultsList({
  results,
  isLoading,
  onSelect,
  selectedFile,
}: {
  results: any[] | undefined;
  isLoading: boolean;
  onSelect: (filename: string) => void;
  selectedFile: string | null;
}) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  if (!results?.length) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-30" />
        <p>No benchmark results yet. Run your first benchmark above.</p>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Questions</TableHead>
          <TableHead>Hit Rate</TableHead>
          <TableHead>Faithfulness</TableHead>
          <TableHead>Relevancy</TableHead>
          <TableHead>Correctness</TableHead>
          <TableHead>Time</TableHead>
          <TableHead></TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {results.map((r) => (
          <TableRow
            key={r.filename}
            className={selectedFile === r.filename ? "bg-muted/50" : "cursor-pointer hover:bg-muted/30"}
            onClick={() => onSelect(r.filename)}
          >
            <TableCell className="font-mono text-xs">{r.model ?? "-"}</TableCell>
            <TableCell>{r.total_questions}</TableCell>
            <TableCell>
              {r.hit_rate !== null ? (
                <span className="flex items-center gap-1">
                  {r.hit_rate >= 0.5 ? (
                    <CheckCircle2 className="h-3 w-3 text-green-500" />
                  ) : r.hit_rate > 0 ? (
                    <TrendingUp className="h-3 w-3 text-yellow-500" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-500" />
                  )}
                  {(r.hit_rate * 100).toFixed(0)}%
                </span>
              ) : (
                "-"
              )}
            </TableCell>
            <TableCell>
              <ScoreBadge score={r.avg_faithfulness} />
            </TableCell>
            <TableCell>
              <ScoreBadge score={r.avg_relevancy} />
            </TableCell>
            <TableCell>
              <ScoreBadge score={r.avg_correctness} />
            </TableCell>
            <TableCell className="text-xs text-muted-foreground">
              {r.total_time_sec < 60
                ? `${r.total_time_sec.toFixed(0)}s`
                : `${(r.total_time_sec / 60).toFixed(1)}m`}
            </TableCell>
            <TableCell>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(r.filename);
                }}
              >
                <FileText className="h-4 w-4" />
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function ResultDetail({ filename }: { filename: string }) {
  const { data, isLoading } = useBenchmarkResult(filename);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-8 text-muted-foreground">Failed to load result details.</div>
    );
  }

  const { summary, results } = data;

  return (
    <div className="space-y-6">
      {/* Summary metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MetricCard
          label="Hit Rate"
          value={summary.hit_rate !== null ? `${(summary.hit_rate * 100).toFixed(0)}%` : null}
          icon={TrendingUp}
          color={summary.hit_rate !== null && summary.hit_rate >= 0.5 ? "text-green-500" : "text-red-500"}
        />
        <MetricCard
          label="MRR"
          value={summary.avg_mrr !== null ? summary.avg_mrr.toFixed(3) : null}
          icon={BarChart3}
        />
        <MetricCard
          label="Faithfulness"
          value={summary.avg_faithfulness !== null ? `${summary.avg_faithfulness.toFixed(1)}/10` : null}
          icon={CheckCircle2}
          color="text-blue-500"
        />
        <MetricCard
          label="Relevancy"
          value={summary.avg_relevancy !== null ? `${summary.avg_relevancy.toFixed(1)}/10` : null}
          icon={TrendingUp}
          color={summary.avg_relevancy !== null && summary.avg_relevancy >= 5 ? "text-green-500" : "text-orange-500"}
        />
        <MetricCard
          label="Correctness"
          value={summary.avg_correctness !== null ? `${summary.avg_correctness.toFixed(1)}/10` : null}
          icon={CheckCircle2}
          color="text-violet-500"
        />
        <MetricCard
          label="Avg Similarity"
          value={summary.avg_similarity !== null ? summary.avg_similarity.toFixed(3) : null}
          icon={TrendingUp}
        />
      </div>

      {/* Per-question results */}
      <Card>
        <CardHeader>
          <CardTitle>Per-Question Results ({results.length})</CardTitle>
          <CardDescription>
            Model: {summary.model ?? "unknown"} | File: {summary.filename}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-[600px] overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[40px]">#</TableHead>
                  <TableHead>Question</TableHead>
                  <TableHead>Answer</TableHead>
                  <TableHead className="w-[60px]">Faith</TableHead>
                  <TableHead className="w-[60px]">Rel</TableHead>
                  <TableHead className="w-[60px]">Corr</TableHead>
                  <TableHead className="w-[60px]">Hit</TableHead>
                  <TableHead>Sources</TableHead>
                  <TableHead className="w-[50px]">Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((r, idx) => (
                  <TableRow key={r.id}>
                    <TableCell className="text-xs text-muted-foreground">{idx + 1}</TableCell>
                    <TableCell className="text-sm max-w-[300px]">
                      <div className="truncate" title={r.question}>
                        {r.question}
                      </div>
                      {r.expected_answer && (
                        <div className="text-xs text-muted-foreground mt-0.5" title={r.expected_answer}>
                          Expected: {r.expected_answer.length > 60 ? r.expected_answer.slice(0, 60) + "..." : r.expected_answer}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-sm max-w-[300px]">
                      <div
                        className="truncate"
                        title={r.answer}
                      >
                        {r.answer.length > 80 ? r.answer.slice(0, 80) + "..." : r.answer}
                      </div>
                    </TableCell>
                    <TableCell>
                      <ScoreBadge score={r.generator_metrics.faithfulness} />
                    </TableCell>
                    <TableCell>
                      <ScoreBadge score={r.generator_metrics.relevancy} />
                    </TableCell>
                    <TableCell>
                      <ScoreBadge score={r.generator_metrics.correctness} />
                    </TableCell>
                    <TableCell>
                      {r.retriever_metrics.hit_rate !== null ? (
                        r.retriever_metrics.hit_rate > 0 ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                        ) : (
                          <XCircle className="h-4 w-4 text-red-500" />
                        )
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-xs max-w-[150px]">
                      <div className="truncate text-muted-foreground">
                        {r.retriever_metrics.retrieved_sources.join(", ") || "-"}
                      </div>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {r.latency_sec.toFixed(0)}s
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

export function AdminBenchmarkPage() {
  const { data, isLoading, refetch } = useBenchmarkResults();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Benchmark</h1>
          <p className="text-muted-foreground">
            RAG quality evaluation — retrieval metrics + LLM-judge scoring
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" /> Refresh
        </Button>
      </div>

      {/* Run benchmark form */}
      <RunBenchmarkForm onSuccess={() => refetch()} />

      {/* Results */}
      <Tabs defaultValue="list">
        <TabsList>
          <TabsTrigger value="list">Results History</TabsTrigger>
          {selectedFile && <TabsTrigger value="detail">Detail: {selectedFile}</TabsTrigger>}
        </TabsList>

        <TabsContent value="list" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" /> Past Results
              </CardTitle>
              <CardDescription>
                {data?.total ?? 0} benchmark runs found. Click a row to view details.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResultsList
                results={data?.results}
                isLoading={isLoading}
                onSelect={setSelectedFile}
                selectedFile={selectedFile}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {selectedFile && (
          <TabsContent value="detail" className="mt-4">
            <ResultDetail filename={selectedFile} />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
