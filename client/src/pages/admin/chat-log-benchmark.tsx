import { useId, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useCurrentUser } from "@/shared/api/hooks/use-auth";
import {
  useBenchmarkDraft,
  useChatLog,
  useTransferToBenchmark,
} from "@/shared/api/hooks/use-chat-logs";
import type { BenchmarkDraft } from "@/shared/api/types";
import { getErrorStatus } from "@/shared/lib/errors";
import { Button } from "@/shared/ui/button";
import { DateInput } from "@/shared/ui/date-input";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { QueryError } from "@/shared/ui/query-error";
import { Textarea } from "@/shared/ui/textarea";
import { SourceHintPicker } from "./benchmark-source-hint-picker";
import { ChatLogContext } from "./chat-log-context";
import { chatLogReviewError } from "./chat-log-review";

const answerTypes = [
  ["", "Default"],
  ["factual", "Factual"],
  ["negative", "Negative answer"],
  ["false_premise", "Correct a false premise"],
  ["not_in_corpus", "Not in corpus"],
  ["out_of_scope", "Out of scope"],
  ["ambiguous", "Clarify ambiguity"],
] as const;

function nonemptyLines(value: string) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function ChatLogBenchmarkPage() {
  const { logId } = useParams();
  const id = Number(logId);
  const valid = Number.isSafeInteger(id) && id > 0;
  const user = useCurrentUser();
  const canTransfer = user.data?.role === "admin";
  const log = useChatLog(id, valid && canTransfer);
  const review = log.data?.entry.review;
  const candidate =
    review?.status === "benchmark_candidate" && review.benchmark_question_id == null;
  const draft = useBenchmarkDraft(id, valid && canTransfer && candidate);
  const questionId = review?.benchmark_question_id ?? draft.data?.benchmark_question_id;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-6 p-4 sm:p-8">
      <h1 className="text-3xl font-semibold tracking-tight">Create a benchmark question</h1>
      {user.isPending ? (
        <p role="status">Loading…</p>
      ) : !canTransfer ? (
        <p role="alert">Only administrators can create benchmark questions.</p>
      ) : !valid || getErrorStatus(log.error) === 404 ? (
        <p role="alert">Could not load this answer. Check the link and your access rights.</p>
      ) : log.isError ? (
        <QueryError query={log} fallback="Could not load this answer." />
      ) : log.isPending ? (
        <p role="status">Loading answer…</p>
      ) : (
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <ChatLogContext entry={log.data.entry} />
          {questionId != null ? (
            <TransferredQuestion id={questionId} />
          ) : !candidate ? (
            <p role="alert">
              Save this entry as a benchmark candidate before creating the question.
            </p>
          ) : draft.isError ? (
            <QueryError query={draft} fallback="Could not load the benchmark draft." />
          ) : draft.isPending ? (
            <p role="status">Loading question…</p>
          ) : (
            <BenchmarkTransferForm
              key={id}
              id={id}
              draft={draft.data}
              originalAnswer={log.data.entry.answer}
            />
          )}
        </div>
      )}
    </div>
  );
}

function TransferredQuestion({ id }: { id: number }) {
  return (
    <section className="space-y-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-5">
      <p role="status" className="font-medium">
        Benchmark question #{id} created
      </p>
      <Button asChild variant="outline">
        <Link to="/admin/benchmark?tab=questions">Open benchmark questions</Link>
      </Button>
    </section>
  );
}

function BenchmarkTransferForm({
  id,
  draft,
  originalAnswer,
}: {
  id: number;
  draft: BenchmarkDraft;
  originalAnswer?: string | null;
}) {
  const fieldsId = useId();
  const transfer = useTransferToBenchmark(id);
  const [form, setForm] = useState(draft);
  const [dataset, setDataset] = useState(draft.datasets[0] ?? "");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [active, setActive] = useState(true);
  const [tags, setTags] = useState("");
  const [notes, setNotes] = useState("");
  const [answerType, setAnswerType] = useState("");
  const [requiredFacts, setRequiredFacts] = useState("");
  const [requiredConditions, setRequiredConditions] = useState("");
  const [annotations, setAnnotations] = useState("");
  const [annotationsError, setAnnotationsError] = useState(false);
  const canSubmit =
    !transfer.isPending &&
    Boolean(dataset && form.question.trim() && (!active || expectedAnswer.trim()));

  if (transfer.isSuccess) return <TransferredQuestion id={transfer.data.question_id} />;

  return (
    <form
      className="min-w-0 space-y-5 rounded-xl border bg-card p-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSubmit) return;

        let parsed: Record<string, unknown> = {};
        if (annotations.trim()) {
          try {
            const value: unknown = JSON.parse(annotations);
            if (value === null || typeof value !== "object" || Array.isArray(value)) {
              setAnnotationsError(true);
              return;
            }
            parsed = value as Record<string, unknown>;
          } catch {
            setAnnotationsError(true);
            return;
          }
        }
        const facts = nonemptyLines(requiredFacts);
        const conditions = nonemptyLines(requiredConditions);
        if (answerType) parsed.answer_type = answerType;
        if (facts.length) parsed.required_facts = facts;
        if (conditions.length) parsed.required_conditions = conditions;
        setAnnotationsError(false);
        transfer.mutate({
          dataset,
          question: form.question,
          history: form.history,
          expected_answer: expectedAnswer.trim() || null,
          source_hint: form.source_hint ?? "",
          as_of_date: form.as_of_date ?? "",
          annotations: Object.keys(parsed).length ? parsed : null,
          tags: tags
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean),
          notes: notes.trim() || null,
          is_active: active,
        });
      }}
    >
      <p role="note" className="text-sm text-amber-600 dark:text-amber-400">
        The transferred text is screened for personal data. Review the content before adding it to a
        benchmark dataset.
      </p>
      {form.datasets.length === 0 ? (
        <p role="alert">This entry has no pack and cannot be transferred.</p>
      ) : (
        <div className="flex flex-col gap-2">
          <Label htmlFor={`${fieldsId}-dataset`}>Dataset</Label>
          <select
            id={`${fieldsId}-dataset`}
            className="block h-10 rounded-md border border-input bg-background px-3 text-sm"
            value={dataset}
            onChange={(event) => setDataset(event.target.value)}
          >
            {form.datasets.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </div>
      )}
      <Label className="flex flex-col gap-2">
        <span>Question</span>
        <Textarea
          required
          maxLength={16000}
          value={form.question}
          onChange={(event) => setForm({ ...form, question: event.target.value })}
        />
      </Label>
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Label htmlFor={`${fieldsId}-expected`}>Expected answer</Label>
          {originalAnswer && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setExpectedAnswer(originalAnswer)}
            >
              Copy original answer
            </Button>
          )}
        </div>
        <Textarea
          id={`${fieldsId}-expected`}
          required={active}
          maxLength={16000}
          className="min-h-32"
          value={expectedAnswer}
          onChange={(event) => setExpectedAnswer(event.target.value)}
          placeholder="Describe what a correct answer should include."
        />
      </div>
      <section
        className="space-y-3 rounded-lg border bg-muted/20 p-4"
        aria-label="Expected sources"
      >
        <h2 className="text-sm font-semibold">Expected sources</h2>
        <SourceHintPicker
          value={form.source_hint ?? null}
          onChange={(source_hint) => setForm({ ...form, source_hint })}
        />
      </section>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <p className="text-sm font-medium">Question date</p>
          <DateInput
            title="Question date"
            value={form.as_of_date ?? ""}
            onValueChange={(as_of_date) => setForm({ ...form, as_of_date })}
          />
        </div>
        <Label className="flex flex-col gap-2">
          <span>Tags</span>
          <Input
            value={tags}
            onChange={(event) => setTags(event.target.value)}
            placeholder="Separate tags with commas"
          />
        </Label>
      </div>
      <Label className="flex flex-col gap-2">
        <span>Internal notes</span>
        <Textarea
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          maxLength={2000}
        />
      </Label>
      <details className="space-y-3 rounded-lg border p-4">
        <summary className="cursor-pointer text-sm font-medium">Evaluation annotations</summary>
        <div className="flex flex-col gap-2">
          <Label htmlFor={`${fieldsId}-answer-type`}>Answer type</Label>
          <select
            id={`${fieldsId}-answer-type`}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm"
            value={answerType}
            onChange={(event) => setAnswerType(event.target.value)}
          >
            {answerTypes.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <Label className="flex flex-col gap-2">
            <span>Required facts</span>
            <Textarea
              value={requiredFacts}
              onChange={(event) => setRequiredFacts(event.target.value)}
              placeholder="One fact per line"
            />
          </Label>
          <Label className="flex flex-col gap-2">
            <span>Required conditions</span>
            <Textarea
              value={requiredConditions}
              onChange={(event) => setRequiredConditions(event.target.value)}
              placeholder="One condition per line"
            />
          </Label>
        </div>
        <Label className="flex flex-col gap-2">
          <span>Additional annotations (JSON)</span>
          <Textarea
            className="min-h-24 font-mono"
            aria-invalid={annotationsError}
            value={annotations}
            onChange={(event) => {
              setAnnotations(event.target.value);
              setAnnotationsError(false);
            }}
            placeholder={JSON.stringify(
              { expected_fragments: [{ source: "document.pdf", text: "Reference passage" }] },
              null,
              2,
            )}
          />
        </Label>
      </details>
      {form.history.map((turn, index) => (
        <div key={index} className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <Label htmlFor={`${fieldsId}-history-${index}`}>
              History {index + 1}: {turn.role === "user" ? "question" : "answer"}
            </Label>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() =>
                setForm({ ...form, history: form.history.filter((_, item) => item !== index) })
              }
            >
              Remove
            </Button>
          </div>
          <Textarea
            id={`${fieldsId}-history-${index}`}
            value={turn.content}
            onChange={(event) =>
              setForm({
                ...form,
                history: form.history.map((item, position) =>
                  position === index ? { ...item, content: event.target.value } : item,
                ),
              })
            }
          />
        </div>
      ))}
      <Label className="flex items-center gap-2">
        <input
          type="checkbox"
          className="size-4"
          checked={active}
          onChange={(event) => setActive(event.target.checked)}
        />
        <span>Include in benchmark runs</span>
      </Label>
      <div className="flex gap-2">
        <Button type="submit" disabled={!canSubmit}>
          {transfer.isPending ? "Creating question…" : "Create question"}
        </Button>
        <Button variant="ghost" asChild>
          <Link to="/admin/chat-logs">Cancel</Link>
        </Button>
      </div>
      {annotationsError ? (
        <p role="alert">Enter a valid JSON object for the evaluation annotations.</p>
      ) : (
        transfer.isError && <p role="alert">{chatLogReviewError(transfer.error)}</p>
      )}
    </form>
  );
}
