"use client";
import { useState } from "react";
import { useChatLog, useChatLogs } from "@/shared/api/hooks/use-chat-logs";
import type { AnswerOutcome, ChatLogEntry } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";

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

function LogCard({ entry }: { entry: ChatLogEntry }) {
  const [open, setOpen] = useState(false);
  const { data, isLoading, isError } = useChatLog(entry.id, open);
  const log = data?.entry ?? entry;
  return (
    <details className="rounded border p-4" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer space-x-3">
        <Badge variant={entry.outcome === "error" ? "destructive" : "secondary"}>
          {entry.outcome}
        </Badge>
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
          <p className="whitespace-pre-wrap">{log.answer || "Ответ не доставлен"}</p>
          <dl className="grid grid-cols-2 gap-2 text-sm">
            <dt>Пространство / пакет</dt>
            <dd>
              {log.space_id} / {log.pack_key ?? "—"}
            </dd>
            <dt>Домен / ширина</dt>
            <dd>
              {log.domain ?? "—"} / {log.breadth ?? "—"} (запрошено: {log.depth_requested ?? "авто"}
              )
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
    user_id: "",
    date_from: "",
    date_to: "",
  });
  const [sortBy, setSortBy] = useState<"date" | "latency" | "confidence">("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const pageSize = 20;
  const { data, refetch, isFetching, isError } = useChatLogs({
    ...filters,
    space_id: filters.space_id ? Number(filters.space_id) : undefined,
    user_id: filters.user_id ? Number(filters.user_id) : undefined,
    date_from: filters.date_from ? new Date(filters.date_from).toISOString() : undefined,
    date_to: filters.date_to ? new Date(filters.date_to).toISOString() : undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit: pageSize,
    offset: page * pageSize,
  });
  const update = (key: keyof typeof filters, value: string) => {
    setFilters((previous) => ({ ...previous, [key]: value }));
    setPage(0);
  };
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Журнал ответов</h1>
        <Button variant="outline" onClick={() => refetch()} disabled={isFetching}>
          Обновить
        </Button>
      </div>
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
          <LogCard key={entry.id} entry={entry} />
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
