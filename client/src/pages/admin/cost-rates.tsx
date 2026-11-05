import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import { RefreshCw } from "lucide-react";
import { useId, useState } from "react";
import toast from "react-hot-toast";
import { apiClient } from "@/shared/api/client";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/tabs";

interface CostRate {
  provider: string;
  model: string;
  version: number;
  input_usd_per_million: string | number;
  output_usd_per_million: string | number;
  context_tokens: number;
}
interface UsageRow {
  operation_id: string;
  principal: string;
  provider: string;
  model: string;
  operation: string;
  reserved_units: number;
  actual_units: number | null;
  created_at: string;
  settled_at: string | null;
  cost_source: string | null;
  reconciliation_evidence: string | null;
}
interface ActiveModels {
  enabled: boolean;
  models: { provider: string; model: string; ready: boolean; reason: string | null }[];
}

function errorMessage(error: unknown, fallback: string) {
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail ?? error.response?.data?.message;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}
function dollars(units: number | null) {
  if (units === null) return "—";
  return `$${(units / 1_000_000_000).toLocaleString("en-US", { maximumFractionDigits: 9 })}`;
}

export function CostRates() {
  const cache = useQueryClient();
  const id = useId();
  const [showForm, setShowForm] = useState(false);
  const [provider, setProvider] = useState("openrouter");
  const [model, setModel] = useState("");
  const [version, setVersion] = useState("1");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [context, setContext] = useState("");
  const [selected, setSelected] = useState<UsageRow | null>(null);
  const [actual, setActual] = useState("");
  const [evidence, setEvidence] = useState("");
  const rates = useQuery({
    queryKey: ["cost-rates"],
    queryFn: async () => (await apiClient.get<CostRate[]>("/admin/cost/rates")).data,
  });
  const models = useQuery({
    queryKey: ["cost-models"],
    queryFn: async () => (await apiClient.get<ActiveModels>("/admin/cost/models")).data,
    refetchInterval: 30_000,
  });
  const usage = useQuery({
    queryKey: ["cost-usage"],
    queryFn: async () => (await apiClient.get<UsageRow[]>("/admin/cost/usage")).data,
    refetchInterval: 30_000,
  });
  const pending = useQuery({
    queryKey: ["cost-pending"],
    queryFn: async () => (await apiClient.get<UsageRow[]>("/admin/cost/pending")).data,
    refetchInterval: 30_000,
  });
  const publish = useMutation({
    mutationFn: async () =>
      apiClient.post("/admin/cost/rates", {
        provider,
        model: model.trim(),
        version: Number(version),
        unit: "tokens",
        input_usd_per_million: input,
        output_usd_per_million: provider === "deepinfra" ? "0" : output,
        context_tokens: Number(context),
      }),
    onSuccess: () => {
      void cache.invalidateQueries({ queryKey: ["cost-rates"] });
      void cache.invalidateQueries({ queryKey: ["cost-models"] });
      setShowForm(false);
      toast.success("Tariff published");
    },
  });
  const reconcile = useMutation({
    mutationFn: async () => {
      if (!selected) throw new Error("Select an operation");
      await apiClient.post(
        `/admin/cost/pending/${encodeURIComponent(selected.operation_id)}/reconcile`,
        {
          actual_usd: actual,
          evidence: evidence.trim(),
        },
      );
    },
    onSuccess: () => {
      void cache.invalidateQueries({ queryKey: ["cost-pending"] });
      void cache.invalidateQueries({ queryKey: ["cost-usage"] });
      setSelected(null);
      setActual("");
      setEvidence("");
      toast.success("Charge reconciled");
    },
  });
  const queries = [rates, models, usage, pending];
  const refresh = () => {
    for (const query of queries) void query.refetch();
  };
  const validTariff =
    model.trim() &&
    Number.isInteger(Number(version)) &&
    Number(version) >= 1 &&
    Number.isInteger(Number(context)) &&
    Number(context) >= 1 &&
    Number(context) <= 10_000_000 &&
    input.trim() &&
    Number.isFinite(Number(input)) &&
    Number(input) > 0 &&
    (provider === "deepinfra" ||
      (output.trim() && Number.isFinite(Number(output)) && Number(output) > 0));

  function operationTable(
    rows: UsageRow[] | undefined,
    loading: boolean,
    failed: boolean,
    canReconcile: boolean,
  ) {
    if (failed)
      return (
        <p role="alert" className="py-6 text-sm text-destructive">
          Operations could not be loaded. Refresh to retry.
        </p>
      );
    if (loading) return <p className="py-6 text-sm text-muted-foreground">Loading operations…</p>;
    if (!rows?.length)
      return (
        <p className="py-6 text-sm text-muted-foreground">
          {canReconcile ? "No charges awaiting confirmation." : "No recorded operations yet."}
        </p>
      );
    return (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Operation</TableHead>
            <TableHead>User</TableHead>
            <TableHead>Reserved</TableHead>
            <TableHead>Actual cost</TableHead>
            <TableHead>Status / source</TableHead>
            <TableHead>Date</TableHead>
            {canReconcile && (
              <TableHead>
                <span className="sr-only">Actions</span>
              </TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.operation_id}>
              <TableCell>
                <p className="font-medium">{row.model}</p>
                <p className="text-xs text-muted-foreground">
                  {row.provider} · {row.operation}
                </p>
                <details className="mt-1 text-xs">
                  <summary className="cursor-pointer">Operation details</summary>
                  <p className="max-w-xs break-all py-2">{row.operation_id}</p>
                  {row.reconciliation_evidence && (
                    <p className="max-w-xs break-words">{row.reconciliation_evidence}</p>
                  )}
                </details>
              </TableCell>
              <TableCell>{row.principal}</TableCell>
              <TableCell className="whitespace-nowrap tabular-nums">
                {dollars(row.reserved_units)}
              </TableCell>
              <TableCell className="whitespace-nowrap tabular-nums">
                {dollars(row.actual_units)}
              </TableCell>
              <TableCell>
                <Badge variant={row.settled_at ? "secondary" : "warning"}>
                  {row.settled_at ? "Settled" : "Awaiting confirmation"}
                </Badge>
                {row.cost_source && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    {row.cost_source.replaceAll("_", " ")}
                  </p>
                )}
              </TableCell>
              <TableCell className="whitespace-nowrap text-xs">
                {new Date(row.created_at).toLocaleString()}
              </TableCell>
              {canReconcile && (
                <TableCell>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={reconcile.isPending}
                    onClick={() => {
                      setSelected(row);
                      setActual("");
                      setEvidence("");
                      reconcile.reset();
                    }}
                  >
                    Reconcile
                  </Button>
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-2">
              <CardTitle>Costs & tariffs</CardTitle>
              <CardDescription>
                Manage model pricing and review paid operations. All amounts are in USD.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={queries.some((q) => q.isFetching)}
              onClick={refresh}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Spending limits</span>
              <Badge
                variant={
                  models.isError ? "destructive" : models.data?.enabled ? "success" : "secondary"
                }
              >
                {models.isError
                  ? "Unavailable"
                  : !models.data
                    ? "Loading…"
                    : models.data.enabled
                      ? "Enabled"
                      : "Disabled"}
              </Badge>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              {!models.data || models.isError
                ? "Budget status could not be confirmed. Refresh to check current limits."
                : models.data.enabled
                  ? "Hourly and daily limits apply per user, including background jobs. Paid calls require a published tariff."
                  : "When disabled, calls bypass budget reservations and cost accounting. Configure spending limits below."}
            </p>
          </div>
          {models.data?.models.map((item) => (
            <div
              key={`${item.provider}/${item.model}`}
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3"
            >
              <div>
                <p className="text-sm font-medium">{item.model}</p>
                <p className="text-xs text-muted-foreground">
                  {item.provider}
                  {item.reason && ` · ${item.reason}`}
                </p>
              </div>
              <Badge variant={item.ready ? "success" : "warning"}>
                {item.ready ? "Ready for budgeted calls" : "Blocked when limits are enabled"}
              </Badge>
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="text-lg">Model tariffs</CardTitle>
              <CardDescription>
                Prices per million tokens. Published versions cannot be edited.
              </CardDescription>
            </div>
            <Button
              variant={showForm ? "outline" : "default"}
              onClick={() => {
                setShowForm(!showForm);
                publish.reset();
              }}
              disabled={publish.isPending}
            >
              {showForm ? "Cancel" : "Add tariff"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          {showForm && (
            <form
              className="space-y-4 rounded-lg border bg-muted/20 p-4"
              onSubmit={(event) => {
                event.preventDefault();
                if (validTariff) publish.mutate();
              }}
            >
              <fieldset
                disabled={publish.isPending}
                className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
              >
                <div className="space-y-2">
                  <Label htmlFor={`${id}-provider`}>Provider</Label>
                  <select
                    id={`${id}-provider`}
                    className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={provider}
                    onChange={(event) => setProvider(event.target.value)}
                  >
                    <option value="openrouter">OpenRouter</option>
                    <option value="deepinfra">DeepInfra</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor={`${id}-model`}>Model ID</Label>
                  <Input
                    id={`${id}-model`}
                    required
                    maxLength={200}
                    placeholder="Exact model ID from provider"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor={`${id}-version`}>Version</Label>
                  <Input
                    id={`${id}-version`}
                    required
                    type="number"
                    min="1"
                    step="1"
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor={`${id}-context`}>Context window (tokens)</Label>
                  <Input
                    id={`${id}-context`}
                    required
                    type="number"
                    min="1"
                    max="10000000"
                    step="1"
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor={`${id}-input`}>Input price ($ / 1M tokens)</Label>
                  <Input
                    id={`${id}-input`}
                    required
                    type="number"
                    min="0"
                    step="any"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                  />
                </div>
                {provider !== "deepinfra" && (
                  <div className="space-y-2">
                    <Label htmlFor={`${id}-output`}>Output price ($ / 1M tokens)</Label>
                    <Input
                      id={`${id}-output`}
                      required
                      type="number"
                      min="0"
                      step="any"
                      value={output}
                      onChange={(e) => setOutput(e.target.value)}
                    />
                  </div>
                )}
              </fieldset>
              <p className="text-xs text-muted-foreground">
                Use a new version for price changes.{" "}
                {provider === "deepinfra" &&
                  "Embedding and reranking models are charged for input tokens only."}
              </p>
              {publish.isError && (
                <p role="alert" className="text-sm text-destructive">
                  {errorMessage(
                    publish.error,
                    "Tariff could not be published. Check the values and retry.",
                  )}
                </p>
              )}
              <Button type="submit" disabled={!validTariff || publish.isPending}>
                {publish.isPending ? "Publishing…" : "Publish tariff"}
              </Button>
            </form>
          )}
          {rates.isError ? (
            <p role="alert" className="text-sm text-destructive">
              Tariffs could not be loaded. Refresh to retry.
            </p>
          ) : rates.isPending ? (
            <p className="text-sm text-muted-foreground">Loading tariffs…</p>
          ) : !rates.data.length ? (
            <p className="py-4 text-sm text-muted-foreground">
              No tariffs published. Add pricing before enabling limits for paid models.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Version</TableHead>
                  <TableHead>Input / 1M tokens</TableHead>
                  <TableHead>Output / 1M tokens</TableHead>
                  <TableHead>Context tokens</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rates.data.map((rate) => (
                  <TableRow key={`${rate.provider}/${rate.model}/${rate.version}`}>
                    <TableCell>
                      <p className="font-medium">{rate.model}</p>
                      <p className="text-xs text-muted-foreground">{rate.provider}</p>
                    </TableCell>
                    <TableCell>v{rate.version}</TableCell>
                    <TableCell>${rate.input_usd_per_million}</TableCell>
                    <TableCell>
                      {rate.provider === "deepinfra"
                        ? "Not billed"
                        : `$${rate.output_usd_per_million}`}
                    </TableCell>
                    <TableCell>{rate.context_tokens.toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Operations</CardTitle>
          <CardDescription>
            Up to 100 records per view. A timeout does not confirm that the provider charged
            nothing.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="pending">
            <TabsList className="mb-4">
              <TabsTrigger value="pending">
                Awaiting confirmation
                {pending.data
                  ? ` (${pending.data.length}${pending.data.length === 100 ? "+" : ""})`
                  : ""}
              </TabsTrigger>
              <TabsTrigger value="recent">Recent activity</TabsTrigger>
            </TabsList>
            <TabsContent value="pending">
              {operationTable(pending.data, pending.isPending, pending.isError, true)}
            </TabsContent>
            <TabsContent value="recent">
              {operationTable(usage.data, usage.isPending, usage.isError, false)}
            </TabsContent>
          </Tabs>
          {selected && (
            <form
              className="mt-5 space-y-4 rounded-lg border p-4"
              onSubmit={(event) => {
                event.preventDefault();
                reconcile.mutate();
              }}
            >
              <fieldset disabled={reconcile.isPending} className="space-y-4">
                <div>
                  <h3 className="font-medium">Confirm provider charge</h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {selected.model} · {selected.principal}
                  </p>
                  <p className="mt-1 break-all font-mono text-xs">{selected.operation_id}</p>
                </div>
                <p className="text-sm text-muted-foreground">
                  Enter the verified charge and a billing reference. Use $0 only when you have
                  confirmed there was no charge.
                </p>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor={`${id}-actual`}>Confirmed charge (USD)</Label>
                    <Input
                      id={`${id}-actual`}
                      required
                      type="number"
                      min="0"
                      step="any"
                      value={actual}
                      onChange={(e) => setActual(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor={`${id}-evidence`}>Billing reference or confirmation</Label>
                    <Input
                      id={`${id}-evidence`}
                      required
                      maxLength={512}
                      value={evidence}
                      onChange={(e) => setEvidence(e.target.value)}
                    />
                  </div>
                </div>
                {reconcile.isError && (
                  <p role="alert" className="text-sm text-destructive">
                    {errorMessage(
                      reconcile.error,
                      "Charge could not be reconciled. Refresh operations and retry.",
                    )}
                  </p>
                )}
                <div className="flex gap-2">
                  <Button
                    type="submit"
                    disabled={
                      reconcile.isPending ||
                      !actual.trim() ||
                      !Number.isFinite(Number(actual)) ||
                      Number(actual) < 0 ||
                      !evidence.trim()
                    }
                  >
                    {reconcile.isPending ? "Saving…" : "Confirm charge"}
                  </Button>
                  <Button
                    variant="outline"
                    type="button"
                    disabled={reconcile.isPending}
                    onClick={() => setSelected(null)}
                  >
                    Cancel
                  </Button>
                </div>
              </fieldset>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
