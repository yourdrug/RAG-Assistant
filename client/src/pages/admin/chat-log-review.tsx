"use client";
import { useEffect, useState } from "react";
import {
  useBenchmarkDraft,
  useReviewChatLog,
  useTransferToBenchmark,
} from "@/shared/api/hooks/use-chat-logs";
import type { BenchmarkDraft, ChatLogEntry, ReviewStatus } from "@/shared/api/types";
import { Button } from "@/shared/ui/button";
import { Label } from "@/shared/ui/label";
import { Textarea } from "@/shared/ui/textarea";

export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  new: "Новая",
  benchmark_candidate: "Кандидат в benchmark",
  in_benchmark: "В benchmark",
  content_gap: "Нет документа",
  pipeline_issue: "Проблема конвейера",
  wont_fix: "Не исправлять",
};

const MANUAL_STATUSES = (Object.keys(REVIEW_STATUS_LABELS) as ReviewStatus[]).filter(
  (status) => status !== "in_benchmark",
);

function apiError(error: unknown): string {
  const status = (error as { response?: { status?: number } })?.response?.status;
  if (status === 409) return "Запись уже перенесена в benchmark.";
  if (status === 422) return "Проверьте поля формы.";
  if (status === 403) return "Недостаточно прав.";
  return "Не удалось сохранить.";
}

export function ReviewForm({ entry }: { entry: ChatLogEntry }) {
  const mutation = useReviewChatLog(entry.id);
  const [status, setStatus] = useState<ReviewStatus>(entry.review?.status ?? "new");
  const [note, setNote] = useState(entry.review?.note ?? "");
  const transferred = entry.review?.status === "in_benchmark";
  return (
    <div className="space-y-2 rounded border p-3">
      <p className="font-medium">Разбор</p>
      <div className="flex flex-wrap items-end gap-3">
        <Label className="space-y-1">
          <span>Статус</span>
          <select
            className="block rounded border p-2"
            value={status}
            onChange={(event) => setStatus(event.target.value as ReviewStatus)}
          >
            {transferred && (
              <option value="in_benchmark">{REVIEW_STATUS_LABELS.in_benchmark}</option>
            )}
            {MANUAL_STATUSES.map((value) => (
              <option key={value} value={value}>
                {REVIEW_STATUS_LABELS[value]}
              </option>
            ))}
          </select>
        </Label>
        <Button
          variant="outline"
          disabled={mutation.isPending || status === "in_benchmark"}
          onClick={() => mutation.mutate({ status, note: note.trim() || null })}
        >
          Сохранить разбор
        </Button>
      </div>
      <Textarea
        aria-label="Заметка разбора"
        placeholder="Заметка"
        maxLength={2000}
        value={note}
        onChange={(event) => setNote(event.target.value)}
      />
      {entry.review?.benchmark_question_id != null && (
        <p className="text-sm">Вопрос benchmark: #{entry.review.benchmark_question_id}</p>
      )}
      {mutation.isError && <p role="alert">{apiError(mutation.error)}</p>}
      {mutation.isSuccess && <p className="text-sm text-muted-foreground">Сохранено.</p>}
    </div>
  );
}

/** ADMIN only: the server also requires the benchmark capability. */
export function BenchmarkTransferForm({ entry }: { entry: ChatLogEntry }) {
  const [open, setOpen] = useState(false);
  const draft = useBenchmarkDraft(entry.id, open);
  const transfer = useTransferToBenchmark(entry.id);
  const [form, setForm] = useState<BenchmarkDraft | null>(null);
  const [dataset, setDataset] = useState("");

  useEffect(() => {
    if (draft.data) {
      setForm(draft.data);
      setDataset(draft.data.datasets[0] ?? "");
    }
  }, [draft.data]);

  if (entry.review?.benchmark_question_id != null) return null;
  if (!open) {
    return (
      <Button variant="outline" onClick={() => setOpen(true)}>
        В benchmark
      </Button>
    );
  }
  return (
    <div className="space-y-3 rounded border p-3">
      <p className="font-medium">Перенос в benchmark</p>
      <p role="note" className="text-sm text-amber-600 dark:text-amber-400">
        Текст прошёл автоматическое удаление персональных данных, но проверьте его вручную: вопрос
        попадёт в benchmark-набор и будет виден всем, кто управляет benchmark.
      </p>
      {draft.isLoading && <p>Загрузка…</p>}
      {draft.isError && <p role="alert">{apiError(draft.error)}</p>}
      {form && (
        <>
          {form.datasets.length === 0 ? (
            <p role="alert">У записи нет пакета: перенос невозможен.</p>
          ) : (
            <Label className="block space-y-1">
              <span>Набор</span>
              <select
                className="block rounded border p-2"
                value={dataset}
                onChange={(event) => setDataset(event.target.value)}
              >
                {form.datasets.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </Label>
          )}
          <Label className="block space-y-1">
            <span>Вопрос</span>
            <Textarea
              value={form.question}
              onChange={(event) => setForm({ ...form, question: event.target.value })}
            />
          </Label>
          {form.history.map((turn, index) => (
            <Label key={index} className="block space-y-1">
              <span className="flex items-center justify-between">
                <span>
                  История {index + 1}: {turn.role === "user" ? "вопрос" : "ответ"}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setForm({ ...form, history: form.history.filter((_, item) => item !== index) })
                  }
                >
                  Убрать
                </Button>
              </span>
              <Textarea
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
            </Label>
          ))}
          <p className="text-sm text-muted-foreground">
            Дата: {form.as_of_date ?? "—"} · Источники: {form.source_hint ?? "—"}. Вопрос создаётся
            неактивным до заполнения ожидаемого ответа и аннотаций.
          </p>
          <div className="flex gap-2">
            <Button
              disabled={transfer.isPending || !dataset || !form.question.trim()}
              onClick={() =>
                transfer.mutate({ dataset, question: form.question, history: form.history })
              }
            >
              Создать вопрос
            </Button>
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Отмена
            </Button>
          </div>
          {transfer.isError && <p role="alert">{apiError(transfer.error)}</p>}
          {transfer.isSuccess && (
            <p className="text-sm">
              Создан вопрос #{transfer.data.question_id} в наборе {transfer.data.dataset}.
            </p>
          )}
        </>
      )}
    </div>
  );
}
