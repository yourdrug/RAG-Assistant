import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { apiClient } from "@/shared/api/client";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";

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
  const cache = useQueryClient();
  const [endpoint, setEndpoint] = useState("");
  const [collection, setCollection] = useState("");
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState("");
  const status = useQuery({
    queryKey: ["corpus-status"],
    refetchInterval: 2000,
    queryFn: async () => (await apiClient.get<CorpusStatus>("/admin/corpus/status")).data,
  });
  async function apply() {
    if (!status.data?.active) return;
    setSwitching(true);
    setError("");
    try {
      await apiClient.post("/admin/corpus/switch", {
        endpoint: endpoint || status.data.active.endpoint,
        collection: collection || status.data.active.alias,
        expected_generation: status.data.active.generation,
      });
      setEndpoint("");
      setCollection("");
      await cache.invalidateQueries({ queryKey: ["admin", "config"] });
      await cache.invalidateQueries({ queryKey: ["admin", "config-effective"] });
    } catch {
      setError(
        "Switch did not finish. Check the journal and process status below before retrying.",
      );
    } finally {
      setSwitching(false);
      await status.refetch();
    }
  }
  const active = status.data?.active;
  return (
    <details className="rounded-lg border p-4">
      <summary className="cursor-pointer font-medium">Qdrant corpus</summary>
      <p className="my-3 text-sm">
        Switching requires API and worker readiness, finished ingestion, matching PostgreSQL chunks
        and ACL, and the same embedding identity. New writes pause until all serving processes apply
        the corpus. Existing requests finish on their physical collection. Use an existing alias or
        collection; creating or moving an alias is a separate operation.
      </p>
      {(error || status.isError || status.data?.error) && (
        <p role="alert">{error || status.data?.error || "Corpus status is unavailable."}</p>
      )}
      <dl className="my-3 grid gap-1 text-sm">
        <dt>State</dt>
        <dd>{status.data?.phase ?? "Checking…"}</dd>
        <dt>Endpoint</dt>
        <dd>{active?.endpoint}</dd>
        <dt>Alias / selected name</dt>
        <dd>{active?.alias}</dd>
        <dt>Physical collection</dt>
        <dd>{active?.physical_collection}</dd>
        <dt>Embedding identity</dt>
        <dd>{active?.embedding_identity}</dd>
      </dl>
      <div className="grid gap-2 md:grid-cols-3">
        <Input
          aria-label="Qdrant endpoint"
          placeholder={active?.endpoint ?? "Qdrant endpoint"}
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
        />
        <Input
          aria-label="Collection or alias"
          placeholder={active?.alias ?? "Collection or alias"}
          value={collection}
          onChange={(e) => setCollection(e.target.value)}
        />
        <Button
          disabled={
            switching || !active || !["active", "failed"].includes(status.data?.phase ?? "")
          }
          onClick={apply}
        >
          {switching ? "Preparing…" : "Prepare and switch corpus"}
        </Button>
      </div>
      <ul className="my-3 text-sm">
        {status.data?.processes.map((p) => (
          <li key={p.process_id}>
            {p.role} {p.process_id.slice(0, 8)}: {p.alive ? "online" : "offline"} ·
            {p.applied_generation === active?.generation ? "applied" : "waiting"}
            {p.error && ` · ${p.error}`}
          </li>
        ))}
      </ul>
    </details>
  );
}
