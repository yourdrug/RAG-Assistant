import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { apiClient } from "@/shared/api/client";
import type { ChatLogEntry } from "@/shared/api/types";
import { Button } from "@/shared/ui/button";

interface TraceDocument {
  metadata: { chunk_id?: number; document_id?: number; citation_id?: number };
  content_hash?: string;
  position?: number;
  score?: number;
  score_kind?: string;
  dense_score?: number;
  sparse_score?: number;
  rrf_score?: number;
  reason?: string;
  stage?: string;
}
interface Check {
  stage: string;
  result: Record<string, string | boolean | number | null>;
}
interface Attempt {
  query: string;
  sub_queries: string[];
  search: Record<string, unknown>;
  stages: Record<
    string,
    { status: string; documents: TraceDocument[]; budget?: Record<string, unknown> }
  >;
  exclusions: TraceDocument[];
  checks: Check[];
}
interface Trace {
  attempts: Attempt[];
  generation: { generator_called: boolean; finish_reason: string | null };
  truncated?: boolean;
}
interface ChunkResult {
  chunk_id: number;
  hidden?: boolean;
  missing?: boolean;
  changed?: boolean;
  content?: string;
}
function Fragment({ logId, chunkId }: { logId: number; chunkId: number }) {
  const chunk = useMutation({
    mutationFn: async () =>
      (
        await apiClient.post<{ chunks: ChunkResult[] }>(`/admin/chat-logs/${logId}/trace/chunks`, {
          chunk_ids: [chunkId],
        })
      ).data.chunks[0],
  });
  return (
    <details
      onToggle={(event) => {
        if (event.currentTarget.open && !chunk.data && !chunk.isPending) chunk.mutate();
      }}
    >
      <summary className="cursor-pointer">Фрагмент #{chunkId}</summary>
      {chunk.isPending && <p>Загрузка…</p>}
      {chunk.isError && <p role="alert">Не удалось загрузить фрагмент.</p>}
      {chunk.data?.hidden && <p>Нет доступа</p>}
      {chunk.data?.missing && <p>Фрагмент удалён или переиндексирован</p>}
      {chunk.data?.changed && <p>Фрагмент изменился</p>}
      <p className="whitespace-pre-wrap">{chunk.data?.content}</p>
    </details>
  );
}
export function ChatLogTrace({ entry }: { entry: ChatLogEntry }) {
  const trace = useQuery({
    queryKey: ["chatLogs", "trace", entry.id],
    queryFn: async () => (await apiClient.get<Trace>(`/admin/chat-logs/${entry.id}/trace`)).data,
    retry: false,
  });
  const [reasoning, setReasoning] = useState<Check[] | null>(null);
  const reasoningRequest = useMutation({
    mutationFn: async () => {
      const ids = [
        ...new Set(
          trace.data?.attempts.flatMap((attempt) =>
            [
              ...Object.values(attempt.stages).flatMap((stage) => stage.documents),
              ...attempt.exclusions,
            ].flatMap((doc) => (doc.metadata.chunk_id ? [doc.metadata.chunk_id] : [])),
          ) ?? [],
        ),
      ];
      const response = await apiClient.post<{ checks: Check[] }>(
        `/admin/chat-logs/${entry.id}/trace/chunks`,
        { chunk_ids: ids },
      );
      setReasoning(response.data.checks);
    },
  });
  if (trace.isPending) return <p>Загрузка трассы…</p>;
  if (trace.isError)
    return <p role="alert">Трасса недоступна: она отключена, истекла или отсутствует.</p>;
  return (
    <div className="space-y-4">
      <ol className="flex flex-wrap gap-4 text-sm">
        <li>Поиск: {entry.retrieval_ms ?? "—"} мс →</li>
        <li>Реранкер: {entry.rerank_ms ?? "—"} мс →</li>
        <li>Генерация: {entry.generation_ms ?? "—"} мс</li>
      </ol>
      {trace.data.truncated && <p>Трасса усечена по лимиту размера.</p>}
      {trace.data.attempts.map((attempt, index) => {
        const docs = new Map<number, TraceDocument>();
        for (const doc of [
          ...Object.values(attempt.stages).flatMap((stage) => stage.documents),
          ...attempt.exclusions,
        ]) {
          if (doc.metadata.chunk_id) {
            const previous = docs.get(doc.metadata.chunk_id);
            docs.set(doc.metadata.chunk_id, {
              ...doc,
              metadata: {
                ...doc.metadata,
                citation_id: doc.metadata.citation_id ?? previous?.metadata.citation_id,
              },
            });
          }
        }
        return (
          <section key={`${index}-${attempt.query}`} className="space-y-2">
            <h3 className="font-semibold">
              Попытка {index + 1}: {attempt.query}
            </h3>
            <p>Подзапросы: {attempt.sub_queries.join("; ") || "—"}</p>
            <pre className="text-xs overflow-auto">{JSON.stringify(attempt.search, null, 2)}</pre>
            <div className="overflow-auto">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr>
                    <th>Кандидат</th>
                    {Object.keys(attempt.stages).map((stage) => (
                      <th key={stage}>
                        {stage} ({attempt.stages[stage].status}): позиция / скор
                      </th>
                    ))}
                    <th>Исключение</th>
                    <th>Цитата</th>
                  </tr>
                </thead>
                <tbody>
                  {[...docs].map(([id, doc]) => (
                    <tr key={id} className="border-t align-top">
                      <td>
                        <Fragment logId={entry.id} chunkId={id} />
                      </td>
                      {Object.entries(attempt.stages).map(([stage, value]) => {
                        const item = value.documents.find(
                          (candidate) => candidate.metadata.chunk_id === id,
                        );
                        return (
                          <td key={stage}>
                            {item ? `${item.position ?? "✓"} / ${item.score ?? "—"}` : "—"}
                            {item && (
                              <div className="text-xs">
                                dense {item.dense_score ?? "—"} · sparse {item.sparse_score ?? "—"}{" "}
                                · RRF {item.rrf_score ?? "—"}
                              </div>
                            )}
                          </td>
                        );
                      })}
                      <td>
                        {attempt.exclusions
                          .filter((item) => item.metadata.chunk_id === id)
                          .map((item) => `${item.stage}: ${item.reason}`)
                          .join(", ") || "—"}
                      </td>
                      <td>
                        {doc.metadata.citation_id &&
                        entry.sources?.some(
                          (source) => source.citation_id === doc.metadata.citation_id,
                        )
                          ? `#${doc.metadata.citation_id}`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>Бюджет промпта</p>
            <pre className="text-xs overflow-auto">
              {JSON.stringify(attempt.stages.prompt?.budget ?? null, null, 2)}
            </pre>
            <p>Проверки gates</p>
            <pre className="text-xs overflow-auto">{JSON.stringify(attempt.checks, null, 2)}</pre>
          </section>
        );
      })}
      <p>
        Промпт отправлен: {trace.data.generation.generator_called ? "да" : "нет"} · Завершение:{" "}
        {trace.data.generation.finish_reason ?? "—"}
      </p>
      <Button
        variant="outline"
        disabled={reasoningRequest.isPending || !trace.data.attempts.length}
        onClick={() => reasoningRequest.mutate()}
      >
        Показать пояснения проверок
      </Button>
      {reasoningRequest.isError && <p role="alert">Не удалось загрузить пояснения.</p>}
      {reasoning && (
        <pre className="text-xs whitespace-pre-wrap">
          {reasoning.length
            ? JSON.stringify(reasoning, null, 2)
            : "Пояснения недоступны: нет доступа ко всем исходным фрагментам либо фрагменты изменились."}
        </pre>
      )}
    </div>
  );
}
