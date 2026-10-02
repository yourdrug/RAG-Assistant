import { useInfiniteQuery } from "@tanstack/react-query";
import { ArrowLeft, FileText } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { Link, useParams } from "react-router-dom";
import remarkGfm from "remark-gfm";
import { apiClient } from "@/shared/api/client";
import { useDocument } from "@/shared/api/hooks/use-documents";
import type { ChunkCursorListResponse } from "@/shared/api/types";
import { Button } from "@/shared/ui/button";

export function DocumentReaderPage() {
  const { documentId } = useParams();
  const parsedId = Number(documentId);
  const id = Number.isSafeInteger(parsedId) && parsedId > 0 ? parsedId : 0;
  const document = useDocument(id);
  const content = useInfiniteQuery({
    queryKey: ["documents", "reader", id],
    initialPageParam: "",
    queryFn: async ({ pageParam }) => {
      const params = new URLSearchParams({ limit: "50" });
      if (pageParam) params.set("cursor", pageParam);
      return (
        await apiClient.get<ChunkCursorListResponse>(`/documents/${id}/chunks/cursor?${params}`)
      ).data;
    },
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    enabled: id > 0 && document.isSuccess,
  });
  const fragments = content.data?.pages.flatMap((page) => page.chunks) ?? [];

  return (
    <main className="mx-auto w-full max-w-4xl space-y-6 p-4 sm:p-8">
      <Link
        to="/documents"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Документы
      </Link>
      {!id || document.isError ? (
        <p role="alert">Документ недоступен: он удалён или у вас нет доступа.</p>
      ) : document.isPending ? (
        <p role="status">Загрузка документа…</p>
      ) : (
        <>
          <header className="space-y-2">
            <h1 className="flex items-start gap-3 text-xl font-semibold break-words">
              <FileText className="mt-1 h-6 w-6 shrink-0" /> {document.data.filename}
            </h1>
            <p className="text-sm text-muted-foreground">
              Текст документа, используемый для поиска и ответов.
            </p>
          </header>
          {content.isError ? (
            <div role="alert" className="space-y-2">
              <p>Не удалось загрузить текст документа.</p>
              <Button variant="outline" onClick={() => content.refetch()}>
                Повторить
              </Button>
            </div>
          ) : content.isPending ? (
            <p role="status">Загрузка текста…</p>
          ) : fragments.length === 0 ? (
            <p>Текст документа пока не готов.</p>
          ) : (
            <article className="space-y-6 rounded-lg border bg-background p-4 sm:p-6">
              {fragments.map((fragment) => (
                <section key={fragment.id} className="space-y-2">
                  {(fragment.section || fragment.heading) && (
                    <h2 className="font-medium">{fragment.heading || fragment.section}</h2>
                  )}
                  <div className="prose prose-sm dark:prose-invert max-w-none break-words">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{fragment.content}</ReactMarkdown>
                  </div>
                </section>
              ))}
            </article>
          )}
          {content.hasNextPage && (
            <Button
              variant="outline"
              disabled={content.isFetchingNextPage}
              onClick={() => content.fetchNextPage()}
            >
              {content.isFetchingNextPage ? "Загрузка…" : "Показать продолжение"}
            </Button>
          )}
        </>
      )}
    </main>
  );
}
