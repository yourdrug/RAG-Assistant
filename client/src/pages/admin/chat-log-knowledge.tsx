import { BookOpen, CheckCircle2, FilePlus2, Loader2, Pencil } from "lucide-react";
import { useId, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useCurrentUser } from "@/shared/api/hooks/use-auth";
import { useChatLog } from "@/shared/api/hooks/use-chat-logs";
import { useAddChunk, useCreateManualDocument } from "@/shared/api/hooks/use-chunks";
import { useDocuments, useUploadableClients } from "@/shared/api/hooks/use-documents";
import { useGroups } from "@/shared/api/hooks/use-groups";
import type { ChatLogEntry, DocumentResponse, DocumentVisibility } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select";
import { Textarea } from "@/shared/ui/textarea";
import { ChatLogContext } from "./chat-log-context";
import { KnowledgeChunkEditor } from "./knowledge-chunk-editor";

const visibilityLabels: Record<DocumentVisibility, string> = {
  internal_private: "Private",
  internal_public: "Public (Internal)",
  internal_group: "Group",
  client_private: "Client Private",
  client_public: "Client Public",
};

export function ChatLogKnowledgePage() {
  const { logId } = useParams();
  const logNumber = Number(logId);
  const valid = Number.isSafeInteger(logNumber) && logNumber > 0;
  const log = useChatLog(logNumber, valid);
  return (
    <div className="mx-auto max-w-screen-2xl space-y-6 p-4 sm:p-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Improve knowledge</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Turn a reviewed answer into better source material.
        </p>
      </div>
      {!valid || log.isError ? (
        <p role="alert">Could not load this answer. Check the link and your access rights.</p>
      ) : log.isPending ? (
        <p role="status">Loading answer…</p>
      ) : (
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <ChatLogContext entry={log.data.entry} />
          <KnowledgeWorkspace key={log.data.entry.id} entry={log.data.entry} />
        </div>
      )}
    </div>
  );
}

function KnowledgeWorkspace({ entry }: { entry: ChatLogEntry }) {
  const id = useId();
  const [params] = useSearchParams();
  const { data: documents, isPending, isError, refetch } = useDocuments();
  const create = useCreateManualDocument();
  const add = useAddChunk();
  const saving = useRef(false);
  const [mode, setMode] = useState<"edit" | "existing" | "new">("edit");
  const [documentId, setDocumentId] = useState(
    params.get("docId") ??
      String(entry.sources?.find((source) => source.document_id)?.document_id ?? ""),
  );
  const [search, setSearch] = useState("");
  const [title, setTitle] = useState(entry.question.slice(0, 200));
  const [visibility, setVisibility] = useState("internal_private");
  const [groupId, setGroupId] = useState("");
  const [clientId, setClientId] = useState("");
  const [sourceId, setSourceId] = useState("");
  const { data: currentUser } = useCurrentUser();
  const { data: groups } = useGroups(mode === "new");
  const { data: clients } = useUploadableClients(mode === "new");
  const [content, setContent] = useState("");
  const [section, setSection] = useState("");
  const [created, setCreated] = useState<DocumentResponse | null>(null);
  const [saved, setSaved] = useState<{ id: number; title: string; warning?: string | null } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [editingChunk, setEditingChunk] = useState(false);
  const busy = create.isPending || add.isPending;
  const sourceIds = new Set(entry.sources?.map((source) => source.document_id));
  const eligible = (documents ?? []).filter(
    (doc) =>
      doc.space_id === entry.space_id && (doc.status === "done" || doc.status === "indexing"),
  );
  const matches = eligible
    .filter((doc) => doc.filename.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => Number(sourceIds.has(b.id)) - Number(sourceIds.has(a.id)));
  const selected = eligible.find((doc) => String(doc.id) === documentId);

  const isAdmin = currentUser?.role === "admin";
  const canUploadForClients = isAdmin || currentUser?.role === "curator";
  const sources = (documents ?? []).filter(
    (doc) => doc.space_id === entry.space_id && sourceIds.has(doc.id),
  );
  const source = sources.find((doc) => String(doc.id) === (sourceId || String(sources[0]?.id)));
  const inherited = visibility === "source";
  const effectiveVisibility = inherited ? source?.visibility : visibility;
  const effectiveGroupId = inherited ? source?.group_id : Number(groupId) || null;
  const effectiveClientId = inherited ? source?.owner_id : Number(clientId) || null;
  const group = groups?.find((group) => group.id === effectiveGroupId);
  const client = clients?.find((client) => client.id === effectiveClientId);
  const canCreate =
    !!effectiveVisibility &&
    (effectiveVisibility !== "internal_private" ||
      !inherited ||
      (!!currentUser && source?.owner_id === currentUser.id)) &&
    (effectiveVisibility !== "internal_public" || isAdmin) &&
    (effectiveVisibility !== "internal_group" || !!group) &&
    (effectiveVisibility !== "client_public" || canUploadForClients) &&
    (effectiveVisibility !== "client_private" || !!client);

  async function save() {
    if (
      mode === "edit" ||
      saving.current ||
      !content.trim() ||
      (mode !== "new" ? !selected : !title.trim() || (!created && !canCreate))
    )
      return;
    saving.current = true;
    setError(null);
    try {
      let target = mode === "existing" ? selected : created;
      if (!target && effectiveVisibility) {
        target = await create.mutateAsync({
          title: title.trim(),
          visibility: effectiveVisibility,
          group_id: effectiveVisibility === "internal_group" ? effectiveGroupId : undefined,
          client_id: effectiveVisibility === "client_private" ? effectiveClientId : undefined,
          space_id: entry.space_id,
        });
        setCreated(target);
      }
      if (!target) return;
      const result = await add.mutateAsync({
        documentId: target.id,
        data: { content: content.trim(), section: section.trim() || undefined },
      });
      setSaved({ id: target.id, title: target.filename, warning: result.warning });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not save the information. Please retry.",
      );
    } finally {
      saving.current = false;
    }
  }

  return (
    <section className="min-w-0 space-y-5 rounded-xl border bg-card p-5 sm:p-6">
      <div>
        <h2 className="text-lg font-semibold">
          {saved ? "Knowledge added" : "Knowledge workspace"}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Changes affect future answers after indexing. Existing access rights still apply.
        </p>
      </div>
      {saved ? (
        <div className="space-y-4">
          <div className="flex items-start gap-3 rounded-lg border bg-muted/40 p-4">
            <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600" />
            <div>
              <p className="font-medium">{saved.title}</p>
              <p className="mt-1 text-sm text-muted-foreground">
                One knowledge passage added. The original answer and review status are preserved.
              </p>
            </div>
          </div>
          {saved.warning && (
            <p role="status" className="text-sm text-amber-700 dark:text-amber-300">
              {saved.warning}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => {
                setSaved(null);
                setCreated(null);
                setContent("");
                setMode("edit");
                setDocumentId(String(saved.id));
              }}
            >
              Continue editing
            </Button>
            <Button asChild>
              <Link to={`/admin/documents?docId=${saved.id}`}>Open document</Link>
            </Button>
          </div>
        </div>
      ) : (
        <form
          className="space-y-5"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <fieldset
            disabled={busy || !!created || editingChunk}
            className="grid grid-cols-1 gap-2 xl:grid-cols-3"
          >
            <legend className="sr-only">Where to save</legend>
            {(
              [
                ["edit", "Edit chunks", Pencil],
                ["existing", "Add information", BookOpen],
                ["new", "Create document", FilePlus2],
              ] as const
            ).map(([value, label, Icon]) => (
              <Button
                key={value}
                type="button"
                variant={mode === value ? "default" : "outline"}
                aria-pressed={mode === value}
                onClick={() => setMode(value)}
              >
                <Icon className="mr-2 h-4 w-4" />
                {label}
              </Button>
            ))}
          </fieldset>
          {mode !== "new" ? (
            <div className="space-y-2">
              <Label htmlFor={`${id}-search`}>Find a document in this space</Label>
              <Input
                id={`${id}-search`}
                type="search"
                placeholder="Search by name…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                disabled={busy || editingChunk}
              />
              <Label htmlFor={`${id}-document`} className="sr-only">
                Document
              </Label>
              <select
                id={`${id}-document`}
                className="h-10 w-full rounded-md border bg-background px-3 text-sm"
                value={documentId}
                onChange={(event) => setDocumentId(event.target.value)}
                disabled={busy || editingChunk || isPending || isError}
                required
              >
                <option value="">{isPending ? "Loading documents…" : "Select a document"}</option>
                {matches.map((doc) => (
                  <option key={doc.id} value={doc.id}>
                    {sourceIds.has(doc.id) ? "Source · " : ""}
                    {doc.filename} · #{doc.id}
                  </option>
                ))}
                {selected && !matches.some((doc) => doc.id === selected.id) && (
                  <option value={selected.id}>{selected.filename}</option>
                )}
              </select>
              {isError && (
                <p role="alert" className="text-sm">
                  Could not load documents.{" "}
                  <button type="button" className="underline" onClick={() => void refetch()}>
                    Retry
                  </button>
                </p>
              )}
              {!isPending && !isError && matches.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No matching documents. Try another name or create a new document.
                </p>
              )}
              {selected && (
                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <Badge variant="secondary">{selected.visibility.replaceAll("_", " ")}</Badge>
                  <span>Existing access rights are preserved.</span>
                </div>
              )}
              <p className="text-xs text-muted-foreground">
                Answer sources appear first. Saving requires permission to edit the selected
                document.
              </p>
            </div>
          ) : (
            <fieldset disabled={busy || !!created} className="space-y-3">
              <div className="space-y-2">
                <Label htmlFor={`${id}-title`}>Document title</Label>
                <Input
                  id={`${id}-title`}
                  value={title}
                  maxLength={255}
                  onChange={(event) => setTitle(event.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor={`${id}-visibility`}>Visibility</Label>
                <Select
                  value={visibility}
                  onValueChange={setVisibility}
                  disabled={busy || !!created}
                >
                  <SelectTrigger id={`${id}-visibility`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="source" disabled={sources.length === 0}>
                      Same as source
                    </SelectItem>
                    <SelectItem value="internal_private">
                      {visibilityLabels.internal_private}
                    </SelectItem>
                    {isAdmin && (
                      <SelectItem value="internal_public">
                        {visibilityLabels.internal_public}
                      </SelectItem>
                    )}
                    <SelectItem value="internal_group">
                      {visibilityLabels.internal_group}
                    </SelectItem>
                    {canUploadForClients && (
                      <>
                        <SelectItem value="client_private">
                          {visibilityLabels.client_private}
                        </SelectItem>
                        <SelectItem value="client_public">
                          {visibilityLabels.client_public}
                        </SelectItem>
                      </>
                    )}
                  </SelectContent>
                </Select>
              </div>
              {inherited && (
                <div className="space-y-2">
                  <Label htmlFor={`${id}-source`}>Source document</Label>
                  <Select
                    value={source ? String(source.id) : ""}
                    onValueChange={setSourceId}
                    disabled={busy || !!created}
                  >
                    <SelectTrigger id={`${id}-source`}>
                      <SelectValue placeholder="Select a source" />
                    </SelectTrigger>
                    <SelectContent>
                      {sources.map((doc) => (
                        <SelectItem key={doc.id} value={String(doc.id)}>
                          {doc.filename} · {visibilityLabels[doc.visibility]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {source && (
                    <p className="text-sm text-muted-foreground">
                      {visibilityLabels[source.visibility]}
                      {source.visibility === "internal_group" && group && ` · ${group.name}`}
                      {source.visibility === "client_private" && client && ` · ${client.email}`}
                    </p>
                  )}
                  {!canCreate && (
                    <p className="text-sm text-muted-foreground">
                      This source’s access settings cannot be copied with your current permissions.
                      Choose another source or set visibility directly.
                    </p>
                  )}
                </div>
              )}
              {!inherited && visibility === "internal_group" && (
                <div className="space-y-2">
                  <Label htmlFor={`${id}-group`}>Group</Label>
                  <Select value={groupId} onValueChange={setGroupId} disabled={busy || !!created}>
                    <SelectTrigger id={`${id}-group`}>
                      <SelectValue placeholder="Select a group" />
                    </SelectTrigger>
                    <SelectContent>
                      {groups?.map((group) => (
                        <SelectItem key={group.id} value={String(group.id)}>
                          {group.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              {!inherited && visibility === "client_private" && (
                <div className="space-y-2">
                  <Label htmlFor={`${id}-client`}>Client</Label>
                  <Select value={clientId} onValueChange={setClientId} disabled={busy || !!created}>
                    <SelectTrigger id={`${id}-client`}>
                      <SelectValue placeholder="Select a client" />
                    </SelectTrigger>
                    <SelectContent>
                      {clients?.map((client) => (
                        <SelectItem key={client.id} value={String(client.id)}>
                          {client.email}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </fieldset>
          )}
          {mode === "edit" && selected && (
            <KnowledgeChunkEditor
              key={selected.id}
              documentId={selected.id}
              onEditingChange={setEditingChunk}
              sourceHashes={
                entry.sources
                  ?.filter((source) => source.document_id === selected.id)
                  .flatMap((source) => source.content_hashes ?? []) ?? []
              }
            />
          )}
          {mode !== "edit" && (
            <>
              {created && (
                <p role="status" className="rounded-lg border p-3 text-sm">
                  Document #{created.id} was created. Retry to add the passage to this document.
                </p>
              )}
              <div className="space-y-2">
                <Label htmlFor={`${id}-content`}>Verified information</Label>
                <Textarea
                  id={`${id}-content`}
                  className="min-h-44"
                  placeholder="Write the missing facts, instructions or a verified answer. Include enough context for this passage to make sense on its own."
                  value={content}
                  onChange={(event) => setContent(event.target.value)}
                  maxLength={10000}
                  disabled={busy}
                  required
                />
                <p className="text-xs text-muted-foreground">
                  {content.length.toLocaleString()} / 10,000 characters · Saved as a new passage for
                  search.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor={`${id}-section`}>Section or reference (optional)</Label>
                <Input
                  id={`${id}-section`}
                  placeholder="e.g. Returns policy · section 4"
                  value={section}
                  onChange={(event) => setSection(event.target.value)}
                  disabled={busy}
                />
              </div>
              {error && (
                <p role="alert" className="text-sm text-destructive">
                  {error}
                </p>
              )}
              <div className="flex justify-end gap-2 border-t pt-4">
                <Button
                  type="submit"
                  disabled={
                    busy ||
                    !content.trim() ||
                    (mode !== "new" ? !selected : !title.trim() || (!created && !canCreate))
                  }
                >
                  {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {busy
                    ? "Saving…"
                    : created
                      ? "Retry adding information"
                      : mode === "new"
                        ? "Create and add knowledge"
                        : "Add to document"}
                </Button>
              </div>
            </>
          )}
        </form>
      )}
    </section>
  );
}
