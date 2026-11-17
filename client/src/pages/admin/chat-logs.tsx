"use client";
import { useState } from "react";
import {
  type ChatLogsParams,
  downloadChatLogs,
  useChatLog,
  useChatLogs,
} from "@/shared/api/hooks/use-chat-logs";
import type { AnswerOutcome, ChatLogEntry, FeedbackReason, ReviewStatus } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { useAuthStore } from "@/stores/auth-store";
import { FEEDBACK_REASON_LABELS } from "@/widgets/chat/message-feedback";
import { BenchmarkTransferForm, REVIEW_STATUS_LABELS, ReviewForm } from "./chat-log-review";
import { ChatLogStatsPanel } from "./chat-log-stats";
import { ChatLogTrace } from "./chat-log-trace";

const outcomes: AnswerOutcome[] = [
  "generated",
  "grounded",
  "cache_hit",
  "not_found",
  "out_of_domain",
  "input_blocked",
  "output_blocked",
  "error",
  "cancelled",
];

function LogCard({ entry, canTransfer }: { entry: ChatLogEntry; canTransfer: boolean }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"answer" | "trace">("answer");
  const { data, isLoading, isError } = useChatLog(entry.id, open);
  const log = data?.entry ?? entry;
  return (
    <details className="rounded border p-4" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer space-x-3">
        <Badge variant={entry.outcome === "error" ? "destructive" : "secondary"}>
          {entry.outcome}
        </Badge>
        {entry.feedback && (
          <Badge variant={entry.feedback.rating === "down" ? "destructive" : "outline"}>
            {entry.feedback.rating === "down" ? "👎" : "👍"}
            {entry.feedback.comment ? " 💬" : ""}
          </Badge>
        )}
        {entry.review && entry.review.status !== "new" && (
          <Badge variant="outline">{REVIEW_STATUS_LABELS[entry.review.status]}</Badge>
        )}
        <span>{entry.question}</span>
        <span className="text-xs text-muted-foreground">
          {new Date(entry.creation_date).toLocaleString()} ·{" "}
          {entry.user_email ?? `#${entry.user_id ?? "—"}`} · {entry.latency_ms ?? "—"} ms
        </span>
      </summary>
      {open && (
        <div className="mt-4 space-y-4">
          {isLoading && <p>Загрузка карточки…</p>}
          {isError && <p role="alert">Не удалось загрузить карточку.</p>}
          <div className="flex gap-2" role="tablist" aria-label="Карточка ответа">
            <Button
              role="tab"
              aria-selected={tab === "answer"}
              variant="outline"
              onClick={() => setTab("answer")}
            >
              Ответ
            </Button>
            <Button
              role="tab"
              aria-selected={tab === "trace"}
              variant="outline"
              onClick={() => setTab("trace")}
            >
              Трасса
            </Button>
          </div>
          {tab === "trace" ? (
            <ChatLogTrace key={entry.id} entry={log} />
          ) : (
            <>
              {data && data.context.length > 0 && (
                <details>
                  <summary className="cursor-pointer text-sm">
                    Предыдущие ходы беседы ({data.context.length})
                  </summary>
                  <ol className="mt-2 space-y-2 text-sm">
                    {data.context.map((turn, index) => (
                      <li key={turn.id ?? index} className="rounded bg-muted/50 p-2">
                        <span className="text-xs text-muted-foreground">
                          {turn.role === "user" ? "Вопрос" : "Ответ"}
                        </span>
                        <p className="whitespace-pre-wrap">{turn.content}</p>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
              <p className="whitespace-pre-wrap">{log.answer || "Ответ не доставлен"}</p>
              {log.feedback && (
                <div className="rounded border p-3 text-sm">
                  <p className="font-medium">
                    Реакция пользователя: {log.feedback.rating === "down" ? "👎" : "👍"}
                  </p>
                  {log.feedback.reasons.length > 0 && (
                    <p>
                      Причины:{" "}
                      {log.feedback.reasons
                        .map((reason) => FEEDBACK_REASON_LABELS[reason])
                        .join(", ")}
                    </p>
                  )}
                  {log.feedback.comment && (
                    <p className="whitespace-pre-wrap">Комментарий: {log.feedback.comment}</p>
                  )}
                </div>
              )}
              <ReviewForm key={`${log.id}-${log.review?.status}`} entry={log} />
              {canTransfer && <BenchmarkTransferForm entry={log} />}
              <dl className="grid grid-cols-2 gap-2 text-sm">
                <dt>Пространство / пакет</dt>
                <dd>
                  {log.space_id} / {log.pack_key ?? "—"}
                </dd>
                <dt>Домен / ширина</dt>
                <dd>
                  {log.domain ?? "—"} / {log.breadth ?? "—"} (запрошено:{" "}
                  {log.depth_requested ?? "авто"})
                </dd>
                <dt>Модель / профиль</dt>
                <dd>
                  {log.model_used ?? "—"} / {log.evidence_profile ?? "—"}
                </dd>
                <dt>Уверенность / код ошибки</dt>
                <dd>
                  {log.confidence ?? "—"} / {log.error_code ?? "—"}
                </dd>
                <dt>Источники / кандидаты / top rerank</dt>
                <dd>
                  {log.sources_count ?? "—"} / {log.candidates_count ?? "—"} /{" "}
                  {log.top_rerank_score ?? "—"}
                </dd>
                <dt>Токены: вход / выход / вспомогательные</dt>
                <dd>
                  {log.input_tokens ?? "—"} / {log.output_tokens ?? "—"} / {log.aux_tokens ?? "—"}
                </dd>
                <dt>TTFT / поиск / реранк / генерация, мс</dt>
                <dd>
                  {log.ttft_ms ?? "—"} / {log.retrieval_ms ?? "—"} / {log.rerank_ms ?? "—"} /{" "}
                  {log.generation_ms ?? "—"}
                </dd>
                <dt>Сообщение / запрос</dt>
                <dd>
                  {log.message_id ?? "—"} / {log.request_id ?? "—"}
                </dd>
                <dt>Дата вопроса</dt>
                <dd>{log.as_of_date ?? "—"}</dd>
                <dt>Сокращённый / уточнённый запрос</dt>
                <dd>
                  {log.condensed_query ?? "—"} / {log.refined_query ?? "—"}
                </dd>
                <dt>Подзапросы</dt>
                <dd>{log.sub_queries?.join("; ") || "—"}</dd>
              </dl>
              <div className="space-x-2">
                {log.sources?.map((source, index) =>
                  source.document_id ? (
                    <a
                      className="underline"
                      key={`${source.document_id}-${index}`}
                      href={`/documents/${source.document_id}`}
                    >
                      {source.source}
                    </a>
                  ) : (
                    <span key={`${source.source}-${index}`}>{source.source}</span>
                  ),
                )}
              </div>
              <details>
                <summary className="cursor-pointer">
                  Настройки: {log.config_hash ?? "нет снимка"}
                </summary>
                <p className="text-sm">Версия процесса: {log.config_version ?? "—"}</p>
                <p>Отличия от текущих настроек</p>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(data?.snapshot.diff ?? "Сравнение недоступно", null, 2)}
                </pre>
                <p>Снимок на момент ответа</p>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(data?.snapshot.data, null, 2)}
                </pre>
              </details>
            </>
          )}
        </div>
      )}
    </details>
  );
}

export function AdminChatLogsPage() {
  const [filters, setFilters] = useState({
    search: "",
    domain: "",
    space_id: "",
    outcome: "",
    model: "",
    breadth: "",
    config_hash: "",
    rating: "",
    reason: "",
    has_comment: "",
    review_status: "",
    user_id: "",
    date_from: "",
    date_to: "",
  });
  const [sortBy, setSortBy] = useState<"date" | "latency" | "confidence">("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const [needsReview, setNeedsReview] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const isAdmin = useAuthStore((state) => state.user?.role === "admin");
  const pageSize = 20;
  const params: ChatLogsParams = {
    ...filters,
    needs_review: needsReview,
    space_id: filters.space_id ? Number(filters.space_id) : undefined,
    user_id: filters.user_id ? Number(filters.user_id) : undefined,
    date_from: filters.date_from ? new Date(filters.date_from).toISOString() : undefined,
    date_to: filters.date_to ? new Date(filters.date_to).toISOString() : undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit: pageSize,
    offset: page * pageSize,
  };
  const { data, refetch, isFetching, isError } = useChatLogs(params);
  const update = (key: keyof typeof filters, value: string) => {
    setFilters((previous) => ({ ...previous, [key]: value }));
    setPage(0);
  };
  const showQueue = (queue: "triage" | "candidates" | "all") => {
    setNeedsReview(queue === "triage");
    setFilters((previous) => ({
      ...previous,
      review_status: queue === "candidates" ? "benchmark_candidate" : "",
    }));
    setSortBy("date");
    setSortOrder("desc");
    setPage(0);
  };
  const exportLogs = (format: "csv" | "json") => {
    setExportError(null);
    downloadChatLogs(params, format).catch(() => setExportError("Не удалось выгрузить журнал."));
  };
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Журнал ответов</h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => exportLogs("csv")}>
            CSV
          </Button>
          <Button variant="outline" onClick={() => exportLogs("json")}>
            JSON
          </Button>
          <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
            Обновить
          </Button>
        </div>
      </div>
      {exportError && <p role="alert">{exportError}</p>}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Очереди">
        <Button
          variant={needsReview ? "default" : "outline"}
          aria-pressed={needsReview}
          onClick={() => showQueue("triage")}
          title="👎 или not_found/error, ещё не разобранные"
        >
          Очередь разбора
        </Button>
        {isAdmin && (
          <Button
            variant={filters.review_status === "benchmark_candidate" ? "default" : "outline"}
            aria-pressed={filters.review_status === "benchmark_candidate"}
            onClick={() => showQueue("candidates")}
          >
            Кандидаты в benchmark
          </Button>
        )}
        <Button variant="ghost" onClick={() => showQueue("all")}>
          Все записи
        </Button>
      </div>
      <ChatLogStatsPanel params={params} />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {(
          [
            ["search", "Поиск"],
            ["domain", "Домен"],
            ["space_id", "ID пространства"],
            ["model", "Модель"],
            ["config_hash", "Хеш настроек"],
            ["user_id", "ID пользователя"],
            ["date_from", "С даты"],
            ["date_to", "По дату"],
          ] as const
        ).map(([key, label]) => (
          <Label key={key} className="block space-y-2">
            <span>{label}</span>
            <Input
              type={
                key.startsWith("date_") ? "datetime-local" : key.endsWith("_id") ? "number" : "text"
              }
              value={filters[key]}
              onChange={(event) => update(key, event.target.value)}
            />
          </Label>
        ))}
        <Label className="block space-y-2">
          <span>Исход</span>
          <select
            className="w-full border rounded p-2"
            value={filters.outcome}
            onChange={(event) => update("outcome", event.target.value)}
          >
            <option value="">Все</option>
            {outcomes.map((outcome) => (
              <option key={outcome}>{outcome}</option>
            ))}
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Оценка</span>
          <select
            className="w-full border rounded p-2"
            value={filters.rating}
            onChange={(event) => update("rating", event.target.value)}
          >
            <option value="">Все</option>
            <option value="down">👎</option>
            <option value="up">👍</option>
            <option value="none">Без оценки</option>
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Причина 👎</span>
          <select
            className="w-full border rounded p-2"
            value={filters.reason}
            onChange={(event) => update("reason", event.target.value)}
          >
            <option value="">Все</option>
            {(Object.keys(FEEDBACK_REASON_LABELS) as FeedbackReason[]).map((reason) => (
              <option key={reason} value={reason}>
                {FEEDBACK_REASON_LABELS[reason]}
              </option>
            ))}
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Комментарий</span>
          <select
            className="w-full border rounded p-2"
            value={filters.has_comment}
            onChange={(event) => update("has_comment", event.target.value)}
          >
            <option value="">Все</option>
            <option value="true">Есть</option>
            <option value="false">Нет</option>
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Статус разбора</span>
          <select
            className="w-full border rounded p-2"
            value={filters.review_status}
            onChange={(event) => update("review_status", event.target.value)}
          >
            <option value="">Все</option>
            {(Object.keys(REVIEW_STATUS_LABELS) as ReviewStatus[]).map((status) => (
              <option key={status} value={status}>
                {REVIEW_STATUS_LABELS[status]}
              </option>
            ))}
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Ширина</span>
          <select
            className="w-full border rounded p-2"
            value={filters.breadth}
            onChange={(event) => update("breadth", event.target.value)}
          >
            <option value="">Все</option>
            <option>narrow</option>
            <option>broad</option>
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Сортировка</span>
          <select
            className="w-full border rounded p-2"
            value={sortBy}
            onChange={(event) => {
              setSortBy(event.target.value as typeof sortBy);
              setPage(0);
            }}
          >
            <option value="date">Дата</option>
            <option value="latency">Время ответа</option>
            <option value="confidence">Уверенность</option>
          </select>
        </Label>
        <Label className="block space-y-2">
          <span>Порядок</span>
          <select
            className="w-full border rounded p-2"
            value={sortOrder}
            onChange={(event) => {
              setSortOrder(event.target.value as typeof sortOrder);
              setPage(0);
            }}
          >
            <option value="desc">По убыванию</option>
            <option value="asc">По возрастанию</option>
          </select>
        </Label>
      </div>
      <p>{data?.total ?? 0} записей</p>
      {isError && (
        <p role="alert">Не удалось загрузить журнал. Проверьте фильтры и повторите запрос.</p>
      )}
      <div className="space-y-3">
        {data?.logs.map((entry) => (
          <LogCard key={entry.id} entry={entry} canTransfer={isAdmin} />
        ))}
      </div>
      <div className="flex gap-4 items-center">
        <Button variant="outline" disabled={page === 0} onClick={() => setPage(page - 1)}>
          Назад
        </Button>
        <span>{page + 1}</span>
        <Button
          variant="outline"
          disabled={!data || (page + 1) * pageSize >= data.total}
          onClick={() => setPage(page + 1)}
        >
          Далее
        </Button>
      </div>
    </div>
  );
}
