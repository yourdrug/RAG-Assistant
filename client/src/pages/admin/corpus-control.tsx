import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { apiClient } from "@/shared/api/client";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";

interface CorpusSnapshot {
  endpoint: string;
  alias: string;
  physical_collection: string;
  embedding_identity: string;
  generation: string;
}
interface CorpusStatus {
  phase: string;
  active: CorpusSnapshot | null;
  candidate: CorpusSnapshot | null;
  error: string | null;
  processes: {
    process_id: string;
    role: string;
    alive: boolean;
    applied_generation: string | null;
    prepared_generation: string | null;
    error: string | null;
  }[];
}
export function CorpusControl() {
  const id = useId();
  const cache = useQueryClient();
  const [endpoint, setEndpoint] = useState("");
  const [collection, setCollection] = useState("");
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const status = useQuery({
    queryKey: ["corpus-status"],
    refetchInterval: 2000,
    queryFn: async () => (await apiClient.get<CorpusStatus>("/admin/corpus/status")).data,
  });
  async function apply() {
    if (!status.data?.active) return;
    setSwitching(true);
    setError("");
    setSuccess(false);
    try {
      await apiClient.post("/admin/corpus/switch", {
        endpoint: endpoint.trim() || status.data.active.endpoint,
        collection: collection.trim() || status.data.active.alias,
        expected_generation: status.data.active.generation,
      });
      setSuccess(true);
      setEndpoint("");
      setCollection("");
      await cache.invalidateQueries({ queryKey: ["admin", "config"] });
      await cache.invalidateQueries({ queryKey: ["admin", "config-effective"] });
    } catch {
      setError(
        "The switch could not be confirmed. Refresh the status and check the active collection before retrying.",
      );
    } finally {
      setSwitching(false);
      await status.refetch();
    }
  }
  const active = status.data?.active;
  const phase = status.data?.phase;
  const busy = switching || phase === "preparing" || phase === "committing";
  const targetEndpoint = endpoint.trim() || active?.endpoint;
  const targetCollection = collection.trim() || active?.alias;
  const changed =
    !!active && (targetEndpoint !== active.endpoint || targetCollection !== active.alias);
  const phaseLabels: Record<string, string> = {
    active: "Collection active",
    preparing: "Checking new collection…",
    committing: "Applying switch…",
    failed: "Last switch failed",
    uninitialized: "Search collection is not initialized",
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle>Where does document search look?</CardTitle>
        <CardDescription>
          Search uses a Qdrant collection: the index of your documents. View the current index or
          switch to an existing one.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <section
          className="rounded-lg border bg-muted/30 p-4 space-y-4"
          aria-label="Current search collection"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-medium">Currently selected</h3>
            <Badge
              variant={
                status.isError
                  ? "warning"
                  : phase === "failed"
                    ? "destructive"
                    : phase === "active"
                      ? "success"
                      : "secondary"
              }
            >
              {status.isError
                ? "Status unavailable"
                : phase
                  ? (phaseLabels[phase] ?? phase)
                  : "Checking…"}
            </Badge>
          </div>
          <dl className="grid gap-4 sm:grid-cols-2">
            <div>
              <dt className="text-xs text-muted-foreground">Collection or alias</dt>
              <dd className="mt-1 break-all font-mono text-sm">
                {active?.alias ?? "Not available"}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Qdrant server</dt>
              <dd className="mt-1 break-all font-mono text-sm">
                {active?.endpoint ?? "Not available"}
              </dd>
            </div>
          </dl>
          {status.isError && (
            <p role="alert" className="text-sm text-destructive">
              Could not load the current status. Displayed values may be out of date.
            </p>
          )}
          <Button
            size="sm"
            variant="outline"
            disabled={status.isFetching}
            onClick={() => void status.refetch()}
          >
            Refresh status
          </Button>
        </section>
        {(error || status.data?.error) && (
          <div
            role="alert"
            className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm space-y-2"
          >
            <p className="font-medium">The last switch did not complete normally</p>
            <p>
              {error ||
                "Review the error below and check the currently selected collection before trying again."}
            </p>
            {status.data?.error && (
              <p className="break-words text-destructive">{status.data.error}</p>
            )}
          </div>
        )}
        {success && !status.isError && phase === "active" && (
          <p role="status" className="rounded-lg border bg-emerald-500/5 p-4 text-sm">
            Switch completed. The collection shown above is now selected for search.
          </p>
        )}
        {busy && (
          <div role="status" className="rounded-lg border p-4 text-sm space-y-2">
            <p className="font-medium">Collection switch in progress</p>
            <p className="text-muted-foreground">
              New document writes pause while servers and workers apply the switch. Existing
              requests finish on their original collection.
            </p>
            {status.data?.candidate && (
              <p className="break-all font-mono text-xs">
                Target: {status.data.candidate.alias} · {status.data.candidate.endpoint}
              </p>
            )}
          </div>
        )}
        <section className="space-y-4" aria-labelledby={`${id}-switch-title`}>
          <div className="space-y-1">
            <h3 id={`${id}-switch-title`} className="font-medium">
              Switch to another collection
            </h3>
            <p className="text-sm text-muted-foreground">
              Use this when an existing index is ready to replace the current one. This does not
              upload documents, create a collection or move an alias.
            </p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor={`${id}-collection`}>New collection or alias</Label>
              <Input
                id={`${id}-collection`}
                placeholder="Enter an existing collection name"
                disabled={busy}
                value={collection}
                onChange={(e) => {
                  setCollection(e.target.value);
                  setSuccess(false);
                }}
                aria-describedby={`${id}-collection-help`}
              />
              <p id={`${id}-collection-help`} className="text-xs text-muted-foreground">
                An alias is a name pointing to a collection. Leave blank to keep{" "}
                {active?.alias ?? "the current name"}.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor={`${id}-endpoint`}>Qdrant server URL</Label>
              <Input
                id={`${id}-endpoint`}
                placeholder={active?.endpoint ?? "http://qdrant:6333"}
                disabled={busy}
                value={endpoint}
                onChange={(e) => {
                  setEndpoint(e.target.value);
                  setSuccess(false);
                }}
                aria-describedby={`${id}-endpoint-help`}
              />
              <p id={`${id}-endpoint-help`} className="text-xs text-muted-foreground">
                Leave blank to use the current server. Change only if the target index is on a
                different server.
              </p>
            </div>
          </div>
          <div className="rounded-lg bg-muted/50 p-4 text-sm space-y-2">
            <p className="font-medium">What happens when you switch</p>
            <ol className="list-decimal pl-5 space-y-1 text-muted-foreground">
              <li>
                The system checks document consistency, access permissions and embedding model
                compatibility.
              </li>
              <li>
                If checks pass, API servers and workers switch to the target collection. Document
                writes pause during the switch.
              </li>
              <li>New search requests use the selected collection after the switch completes.</li>
            </ol>
          </div>
          {changed && (
            <p className="text-sm break-all">
              Target: <span className="font-mono">{targetCollection}</span> on{" "}
              <span className="font-mono">{targetEndpoint}</span>
            </p>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <Button
              disabled={
                busy || !changed || status.isError || !["active", "failed"].includes(phase ?? "")
              }
              onClick={apply}
            >
              {busy ? "Switch in progress…" : "Check compatibility and switch"}
            </Button>
            <p className="text-xs text-muted-foreground">
              {!active
                ? "Current collection must be available before switching."
                : !changed
                  ? "Enter a different collection or server to continue."
                  : "This will change the index used for document search."}
            </p>
          </div>
        </section>
        <details className="rounded-lg border p-4">
          <summary className="cursor-pointer text-sm font-medium">
            Technical details & server readiness
          </summary>
          <dl className="my-4 space-y-3 text-sm">
            <div>
              <dt className="text-muted-foreground">
                Physical collection (the index behind the alias)
              </dt>
              <dd className="break-all font-mono text-xs">
                {active?.physical_collection ?? "Not reported"}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Embedding model identity</dt>
              <dd className="break-all font-mono text-xs">
                {active?.embedding_identity ?? "Not reported"}
              </dd>
            </div>
          </dl>
          <p className="mb-3 text-xs text-muted-foreground">
            Both API servers and workers must be online and using the current index before another
            switch can start.
          </p>
          <ul className="divide-y text-sm">
            {status.data?.processes.map((p) => (
              <li key={p.process_id} className="py-2 space-y-1">
                <div className="flex flex-wrap gap-2">
                  <span className="font-medium">{p.role}</span>
                  <span className="font-mono text-muted-foreground">
                    {p.process_id.slice(0, 8)}
                  </span>
                  <Badge variant={p.error || !p.alive ? "warning" : "secondary"}>
                    {p.alive ? "Online" : "Offline"}
                  </Badge>
                  <span className="text-muted-foreground">
                    {active && p.applied_generation === active.generation
                      ? "Using current index"
                      : "Current index not confirmed"}
                  </span>
                </div>
                {p.error && <p className="break-words text-destructive">{p.error}</p>}
              </li>
            ))}
          </ul>
          {!status.data?.processes.length && (
            <p className="text-sm text-muted-foreground">No process status reported.</p>
          )}
        </details>
      </CardContent>
    </Card>
  );
}
