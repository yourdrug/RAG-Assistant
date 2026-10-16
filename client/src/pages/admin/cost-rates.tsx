import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { apiClient } from "@/shared/api/client";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";

type ActiveModels = {
  enabled: boolean;
  models: { provider: string; model: string; ready: boolean; reason: string | null }[];
};

export function CostRates() {
  const cache = useQueryClient();
  const [provider, setProvider] = useState("openrouter");
  const [model, setModel] = useState("");
  const [version, setVersion] = useState("1");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [context, setContext] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [operation, setOperation] = useState("");
  const [actual, setActual] = useState("");
  const [evidence, setEvidence] = useState("");
  async function reconcile() {
    setSaving(true);
    setError("");
    try {
      await apiClient.post(`/admin/cost/pending/${encodeURIComponent(operation)}/reconcile`, {
        actual_usd: actual,
        evidence,
      });
      await cache.invalidateQueries({ queryKey: ["cost-pending"] });
      await cache.invalidateQueries({ queryKey: ["cost-usage"] });
      setOperation("");
      setActual("");
      setEvidence("");
    } catch {
      setError("Reconciliation failed. Confirm the provider charge and operation ID.");
    } finally {
      setSaving(false);
    }
  }
  const rates = useQuery({
    queryKey: ["cost-rates"],
    queryFn: async () => (await apiClient.get("/admin/cost/rates")).data,
  });
  const models = useQuery({
    queryKey: ["cost-models"],
    queryFn: async () => (await apiClient.get<ActiveModels>("/admin/cost/models")).data,
    refetchInterval: 30_000,
  });
  const usage = useQuery({
    queryKey: ["cost-usage"],
    queryFn: async () => (await apiClient.get("/admin/cost/usage")).data,
  });
  const pending = useQuery({
    queryKey: ["cost-pending"],
    queryFn: async () => (await apiClient.get("/admin/cost/pending")).data,
  });
  async function save() {
    setSaving(true);
    setError("");
    try {
      await apiClient.post("/admin/cost/rates", {
        provider,
        model,
        version: Number(version),
        unit: "tokens",
        input_usd_per_million: input,
        output_usd_per_million: output,
        context_tokens: Number(context),
      });
      await cache.invalidateQueries({ queryKey: ["cost-rates"] });
      await cache.invalidateQueries({ queryKey: ["cost-models"] });
    } catch {
      setError("Tariff was not saved. Check the values and use a new version.");
    } finally {
      setSaving(false);
    }
  }
  return (
    <details className="rounded-lg border p-4">
      <summary className="cursor-pointer font-medium">USD budgets and tariffs</summary>
      <p className="my-3 text-sm">
        When enabled, rolling hour/day limits cover all paid operations of each authenticated user,
        including background jobs. Unknown prices block calls. Timeouts and cancelled calls retain
        their reservations until reconciliation.
      </p>
      <p className="my-3 text-sm">
        When limits are enabled, every paid attempt requires a published tariff. When limits are
        disabled, calls bypass budget reservations and accounting. Supported: OpenRouter text and
        structured calls, BAAI/bge-m3 embeddings and Qwen/Qwen3-Reranker-0.6B / 4B reranking. Local
        models are not billed. Unit: USD per million tokens. Versions are immutable. Reconciliation
        records provider cost, tariff calculation or administrative confirmation as its source.
      </p>
      <p className="my-3 text-sm">
        Limits: {models.data ? (models.data.enabled ? "enabled" : "disabled") : "unavailable"}
      </p>
      {models.data?.models.map((item) => (
        <p key={`${item.provider}/${item.model}`} role={item.ready ? undefined : "alert"}>
          {item.model}:{" "}
          {item.ready
            ? "Contract and tariff available"
            : `Blocked when limits are enabled: ${item.reason}`}
        </p>
      ))}
      <div className="grid gap-2 md:grid-cols-3">
        <select
          aria-label="Provider"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
        >
          <option value="openrouter">OpenRouter</option>
          <option value="deepinfra">DeepInfra</option>
        </select>
        <Input
          aria-label="Model"
          placeholder="Exact provider model ID"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        />
        <Input
          aria-label="Tariff version"
          type="number"
          min="1"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
        />
        <Input
          aria-label="Maximum model context tokens"
          placeholder="Maximum model context tokens"
          type="number"
          min="1"
          value={context}
          onChange={(e) => setContext(e.target.value)}
        />
        <Input
          aria-label="Input USD per million tokens"
          placeholder="Input USD / million tokens"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <Input
          aria-label="Output USD per million tokens"
          placeholder="Output USD / million tokens"
          value={output}
          onChange={(e) => setOutput(e.target.value)}
        />
        <Button disabled={saving} onClick={save}>
          Publish tariff version
        </Button>
      </div>
      {error && <p role="alert">{error}</p>}
      {(rates.isError || pending.isError || usage.isError || models.isError) && (
        <p role="alert">Budget data is unavailable.</p>
      )}
      <pre className="my-3 overflow-auto text-xs">{JSON.stringify(rates.data ?? [], null, 2)}</pre>
      <details>
        <summary>Pending reservations (up to 100)</summary>
        <pre className="overflow-auto text-xs">{JSON.stringify(pending.data ?? [], null, 2)}</pre>
      </details>
      <details>
        <summary>Recent usage and reconciliation sources (up to 100)</summary>
        <pre className="overflow-auto text-xs">{JSON.stringify(usage.data ?? [], null, 2)}</pre>
      </details>
      <p className="my-3 text-sm">
        Reconcile a pending reservation only after confirming the actual provider charge. A timeout
        alone does not prove a zero charge.
      </p>
      <div className="grid gap-2 md:grid-cols-4">
        <Input
          aria-label="Operation ID"
          placeholder="Operation ID"
          value={operation}
          onChange={(e) => setOperation(e.target.value)}
        />
        <Input
          aria-label="Confirmed charge USD"
          placeholder="Confirmed charge USD"
          value={actual}
          onChange={(e) => setActual(e.target.value)}
        />
        <Input
          aria-label="Charge confirmation reference"
          placeholder="Charge confirmation reference"
          value={evidence}
          onChange={(e) => setEvidence(e.target.value)}
        />
        <Button disabled={saving || !operation || !actual || !evidence.trim()} onClick={reconcile}>
          Reconcile charge
        </Button>
      </div>
    </details>
  );
}
