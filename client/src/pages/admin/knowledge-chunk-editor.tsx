import { Loader2, Pencil } from "lucide-react";
import { useId, useRef, useState } from "react";
import { useChunks, useUpdateChunk } from "@/shared/api/hooks/use-chunks";
import type { ChunkResponse } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Label } from "@/shared/ui/label";
import { Textarea } from "@/shared/ui/textarea";

export function KnowledgeChunkEditor({
  documentId,
  sourceHashes,
  onEditingChange,
}: {
  documentId: number;
  sourceHashes: string[];
  onEditingChange: (editing: boolean) => void;
}) {
  const [page, setPage] = useState(0);
  const [sourcesOnly, setSourcesOnly] = useState(sourceHashes.length > 0);
  const [editing, setEditing] = useState<ChunkResponse | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const chunks = useChunks(
    documentId,
    20,
    page * 20,
    sourcesOnly ? sourceHashes.join(",") : undefined,
  );
  if (editing)
    return (
      <ChunkEditor
        chunk={editing}
        onCancel={() => {
          setEditing(null);
          onEditingChange(false);
        }}
        onSaved={(warning) => {
          setEditing(null);
          onEditingChange(false);
          setNotice(warning || "Chunk saved. Changes will be available after indexing.");
          setSourcesOnly(false);
          setPage(0);
        }}
      />
    );
  return (
    <section className="space-y-4 border-t pt-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-semibold">Document chunks</h3>
        {sourceHashes.length > 0 && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            aria-pressed={sourcesOnly}
            onClick={() => {
              setSourcesOnly(!sourcesOnly);
              setPage(0);
            }}
          >
            {sourcesOnly ? "Show all chunks" : "Show answer sources"}
          </Button>
        )}
      </div>
      {notice && (
        <p role="status" className="rounded-lg border bg-muted/40 p-3 text-sm">
          {notice}
        </p>
      )}
      {chunks.isPending && <p role="status">Loading chunks…</p>}
      {chunks.isError && (
        <p role="alert">
          Could not load chunks.{" "}
          <button type="button" className="underline" onClick={() => void chunks.refetch()}>
            Retry
          </button>
        </p>
      )}
      {chunks.data?.chunks.length === 0 && (
        <p className="text-sm text-muted-foreground">
          {sourcesOnly
            ? "Source chunks have changed or are no longer available. Show all chunks to find the current text."
            : "No chunks yet. Choose Add information to write the first passage."}
        </p>
      )}
      {chunks.data?.chunks.map((chunk) => (
        <article key={chunk.id} className="space-y-3 rounded-lg border p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="text-sm font-medium">Chunk #{chunk.id}</h4>
              {sourceHashes.includes(chunk.content_hash ?? "") && (
                <Badge variant="secondary">Answer source</Badge>
              )}
              {chunk.manual && <Badge variant="outline">Edited</Badge>}
            </div>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => {
                setNotice(null);
                setEditing(chunk);
                onEditingChange(true);
              }}
            >
              <Pencil className="mr-2 h-3.5 w-3.5" />
              Edit chunk
            </Button>
          </div>
          {(chunk.heading || chunk.section) && (
            <p className="text-xs text-muted-foreground">{chunk.heading || chunk.section}</p>
          )}
          <p className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-sm leading-relaxed">
            {chunk.content}
          </p>
        </article>
      ))}
      {chunks.data && chunks.data.total > 20 && (
        <div className="flex items-center justify-between gap-2">
          <Button
            type="button"
            variant="outline"
            disabled={page === 0 || chunks.isFetching}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page + 1} · {chunks.data.total} chunks
          </span>
          <Button
            type="button"
            variant="outline"
            disabled={(page + 1) * 20 >= chunks.data.total || chunks.isFetching}
            onClick={() => setPage(page + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </section>
  );
}

function ChunkEditor({
  chunk,
  onCancel,
  onSaved,
}: {
  chunk: ChunkResponse;
  onCancel: () => void;
  onSaved: (warning?: string | null) => void;
}) {
  const id = useId();
  const [content, setContent] = useState(chunk.content);
  const [error, setError] = useState<string | null>(null);
  const saving = useRef(false);
  const update = useUpdateChunk();
  async function save() {
    if (saving.current || !content.trim() || content === chunk.content || content.length > 10000)
      return;
    saving.current = true;
    setError(null);
    try {
      const result = await update.mutateAsync({
        documentId: chunk.document_id,
        chunkId: chunk.id,
        data: { content: content.trim() },
      });
      onSaved(result.warning);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not save this chunk. Your text is preserved; please retry.",
      );
    } finally {
      saving.current = false;
    }
  }
  return (
    <section className="space-y-4 border-t pt-5">
      <div>
        <h3 className="font-semibold">Editing chunk #{chunk.id}</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          You are replacing this passage in the document. Verify the facts before saving.
        </p>
      </div>
      <details className="rounded-lg border p-3">
        <summary className="cursor-pointer text-sm">Original text</summary>
        <p className="mt-3 whitespace-pre-wrap break-words text-sm text-muted-foreground">
          {chunk.content}
        </p>
      </details>
      <Label htmlFor={id}>Chunk content</Label>
      <Textarea
        id={id}
        className="min-h-80"
        value={content}
        onChange={(event) => setContent(event.target.value)}
        disabled={update.isPending}
        maxLength={10000}
      />
      <p className="text-xs text-muted-foreground">
        {content.length.toLocaleString()} / 10,000 characters
      </p>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" disabled={update.isPending} onClick={onCancel}>
          Cancel editing
        </Button>
        <Button
          type="button"
          disabled={
            update.isPending ||
            !content.trim() ||
            content === chunk.content ||
            content.length > 10000
          }
          onClick={() => void save()}
        >
          {update.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Save chunk
        </Button>
      </div>
    </section>
  );
}
