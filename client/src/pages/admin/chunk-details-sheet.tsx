import { Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { useChunkIndexState } from "@/shared/api/hooks";
import type { ChunkIndexStateResponse, ChunkResponse } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/shared/ui/sheet";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      {children}
    </h3>
  );
}

function MetadataGrid({ chunk }: { chunk: ChunkResponse }) {
  const rows: Array<[string, unknown]> = [
    ["section", chunk.section],
    ["heading", chunk.heading],
    ["heading_level", chunk.heading_level],
    ["content_type", chunk.content_type],
    ["doc_title", chunk.doc_title],
    ["doc_type", chunk.doc_type],
    ["doc_domain", chunk.doc_domain],
    ["visibility", chunk.visibility],
    ["owner_id", chunk.owner_id],
    ["group_id", chunk.group_id],
    ["edited_by", chunk.edited_by],
    ["edited_at", formatDate(chunk.edited_at)],
    ["created", formatDate(chunk.creation_date)],
    ["content_hash", chunk.content_hash],
  ];
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-sm sm:grid-cols-[max-content_1fr_max-content_1fr]">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="text-muted-foreground">{key}</dt>
          <dd className="break-all font-mono text-xs leading-5">{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function IndexState({ state }: { state: ChunkIndexStateResponse }) {
  if (!state.indexed) {
    return (
      <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm">
        Point not found in Qdrant.
        {state.document_status !== "done" && (
          <span className="text-muted-foreground">
            {" "}
            Document status is “{state.document_status}”; indexing may still be in progress.
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        <Badge variant={state.in_sync ? "success" : "destructive"}>
          {state.in_sync ? "In sync" : "Out of sync"}
        </Badge>
        <Badge variant={state.dense_dim ? "success" : "destructive"}>
          dense {state.dense_dim ? `✓ ${state.dense_dim}` : "missing"}
        </Badge>
        <Badge variant={state.sparse_terms ? "success" : "destructive"}>
          sparse bm25 {state.sparse_terms ? `✓ ${state.sparse_terms} terms` : "missing"}
        </Badge>
        <Badge variant={state.content_matches ? "success" : "destructive"}>
          text {state.content_matches ? "matches" : "differs"}
        </Badge>
        {state.embedding_model && <Badge variant="outline">{state.embedding_model}</Badge>}
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Field</TableHead>
              <TableHead>Postgres</TableHead>
              <TableHead>Qdrant payload</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {state.fields.map((field) => (
              <TableRow
                key={field.name}
                className={field.matches ? undefined : "bg-destructive/10"}
              >
                <TableCell className="font-mono text-xs">{field.name}</TableCell>
                <TableCell className="font-mono text-xs">{formatValue(field.expected)}</TableCell>
                <TableCell className="font-mono text-xs">{formatValue(field.actual)}</TableCell>
                <TableCell>
                  <Badge variant={field.matches ? "success" : "destructive"}>
                    {field.matches ? "OK" : "Mismatch"}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <details className="rounded-md border p-3 text-sm">
        <summary className="cursor-pointer text-muted-foreground">Full payload.metadata</summary>
        <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-all font-mono text-xs">
          {JSON.stringify(state.payload_metadata, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function IndexStateSection({ documentId, chunkId }: { documentId: number; chunkId: number }) {
  const { data, isLoading, isError } = useChunkIndexState(documentId, chunkId, true);
  return (
    <section className="space-y-2">
      <SectionTitle>Qdrant index state</SectionTitle>
      {isLoading ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
        </div>
      ) : isError || !data ? (
        <p className="text-sm text-destructive">Failed to load index state.</p>
      ) : (
        <IndexState state={data} />
      )}
    </section>
  );
}

export function ChunkDetailsSheet({
  documentId,
  chunk,
  showIndexState,
  onClose,
}: {
  documentId: number;
  chunk: ChunkResponse | null;
  showIndexState: boolean;
  onClose: () => void;
}) {
  return (
    <Sheet open={chunk !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-2xl">
        {chunk && (
          <div className="space-y-5">
            <SheetHeader>
              <SheetTitle>
                Chunk #{chunk.chunk_index}{" "}
                <span className="font-mono text-sm font-normal text-muted-foreground">
                  (id {chunk.id})
                </span>
              </SheetTitle>
              <SheetDescription className="flex flex-wrap items-center gap-1.5">
                <span>{chunk.filename || chunk.doc_title || `document ${documentId}`}</span>
                <span>· {chunk.content.length} chars</span>
                {chunk.manual && <Badge variant="outline">Manual</Badge>}
                {chunk.edited_at && <Badge variant="warning">Edited</Badge>}
              </SheetDescription>
            </SheetHeader>

            <section className="space-y-2">
              <SectionTitle>Full text</SectionTitle>
              <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md border bg-muted/40 p-3 font-sans text-sm leading-relaxed">
                {chunk.content}
              </pre>
            </section>

            <section className="space-y-2">
              <SectionTitle>Metadata (Postgres)</SectionTitle>
              <MetadataGrid chunk={chunk} />
            </section>

            {showIndexState && <IndexStateSection documentId={documentId} chunkId={chunk.id} />}
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
