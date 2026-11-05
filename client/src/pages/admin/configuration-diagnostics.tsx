import { useState } from "react";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

export interface EffectiveParameter {
  key: string;
  source: string;
  configured_value: string | null;
  applied_value: string | null;
  application_mode: string;
  present: boolean | null;
  domain_key: string | null;
}

const SOURCES: Record<string, string> = {
  database: "Application settings",
  deployment: "Server environment",
  secret: "Credentials",
};
const APPLICATION: Record<string, string> = {
  next_operation: "Next request or task",
  next_cron: "Next hourly cleanup",
  recreate: "Recreate containers to change",
  corpus_switch: "Use the Search collection tab",
};

export function ConfigurationDiagnostics({
  parameters,
  descriptions,
  unavailable,
  status,
  statusUnavailable,
  onRefresh,
}: {
  parameters: EffectiveParameter[] | undefined;
  descriptions:
    | { key: string; domain_key: string | null; description: string | null }[]
    | undefined;
  unavailable: boolean;
  status:
    | {
        ready: boolean;
        error: string | null;
        saved_version: string | null;
        applied_version: string | null;
      }
    | undefined;
  statusUnavailable: boolean;
  onRefresh: () => void;
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const differs = (p: EffectiveParameter) =>
    p.source === "database" &&
    p.configured_value !== null &&
    p.configured_value !== p.applied_value;
  const differences = parameters?.filter(differs).length ?? 0;
  const rows = parameters?.filter((p) => {
    const description =
      descriptions?.find((d) => d.key === p.key && d.domain_key === p.domain_key)?.description ??
      "";
    return (
      `${p.key} ${p.domain_key ?? ""} ${description}`
        .toLowerCase()
        .includes(search.trim().toLowerCase()) &&
      (filter === "all" || (filter === "differences" ? differs(p) : p.source === filter))
    );
  });
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <CardTitle>Are my settings in use?</CardTitle>
            <CardDescription>
              Compare saved settings with the values this API server currently uses.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={onRefresh}>
            Refresh status
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="rounded-lg border bg-muted/30 p-4 space-y-2" role="status">
          <Badge
            variant={
              statusUnavailable || unavailable
                ? "warning"
                : status?.error
                  ? "destructive"
                  : status?.ready
                    ? "success"
                    : "secondary"
            }
          >
            {statusUnavailable || unavailable
              ? "Unable to verify"
              : !status
                ? "Checking…"
                : status.error
                  ? "Settings could not be applied"
                  : status.ready
                    ? "Saved settings are active"
                    : "Saved changes are pending"}
          </Badge>
          <p className="text-sm text-muted-foreground">
            {statusUnavailable || unavailable
              ? "Status could not be loaded. Refresh to try again; displayed values may be out of date."
              : status?.error
                ? "This server is still using its previous settings. Review the error and correct the saved values in the relevant settings tab."
                : status?.ready
                  ? "No action is needed for saved application settings."
                  : "Wait for the server to apply the changes, then refresh. If they remain pending, inspect the differences below."}
          </p>
          {status?.error && (
            <p role="alert" className="break-words text-sm text-destructive">
              {status.error}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            This check covers the API server answering this request. Other API servers and workers
            may have a different status.
          </p>
        </div>
        <div className="space-y-3">
          <Input
            aria-label="Find a configuration parameter"
            placeholder="Find a setting by name or description…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="flex flex-wrap gap-2" aria-label="Filter configuration parameters">
            {[
              ["all", "All settings"],
              ["differences", `Saved / in use differ (${differences})`],
              ["database", "Application settings"],
              ["deployment", "Server environment"],
              ["secret", "Credentials"],
            ].map(([value, label]) => (
              <Button
                key={value}
                size="sm"
                variant={filter === value ? "default" : "outline"}
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
              >
                {label}
              </Button>
            ))}
          </div>
          <p className="text-sm text-muted-foreground">
            Read-only view. Edit application settings in the other tabs. Environment values and
            credentials are managed on the server.
          </p>
        </div>
        {!parameters ? (
          <p role="status">
            {unavailable
              ? "Configuration values could not be loaded."
              : "Loading configuration values…"}
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Setting</TableHead>
                <TableHead>Saved value</TableHead>
                <TableHead>In use on this server</TableHead>
                <TableHead>When changes take effect</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows?.map((p) => (
                <TableRow
                  key={`${p.domain_key ?? "global"}:${p.key}`}
                  className={differs(p) ? "bg-amber-500/5" : undefined}
                >
                  <TableCell className="min-w-52 max-w-sm align-top">
                    <p className="font-mono text-xs break-all">{p.key}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {SOURCES[p.source] ?? p.source}
                      {p.domain_key ? ` · ${p.domain_key} domain` : ""}
                    </p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {
                        descriptions?.find((d) => d.key === p.key && d.domain_key === p.domain_key)
                          ?.description
                      }
                    </p>
                  </TableCell>
                  <TableCell className="max-w-xs break-all align-top">
                    {p.source === "secret" ? "Hidden" : (p.configured_value ?? "No saved override")}
                  </TableCell>
                  <TableCell className="max-w-xs break-all align-top">
                    {p.present === null
                      ? (p.applied_value ?? "Not reported")
                      : p.present
                        ? "Configured (hidden)"
                        : "Not configured"}
                    {differs(p) && (
                      <Badge variant="warning" className="mt-2 block w-fit">
                        Differs from saved
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="min-w-40 align-top text-sm text-muted-foreground">
                    {APPLICATION[p.application_mode] ?? p.application_mode}
                  </TableCell>
                </TableRow>
              ))}
              {rows?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="py-8 text-center text-muted-foreground">
                    {filter === "differences" && !search
                      ? "No differences between saved values and values in use."
                      : "No settings match this filter."}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
        <details className="rounded-lg border p-4 text-sm">
          <summary className="cursor-pointer font-medium">
            Application timing & technical details
          </summary>
          <div className="mt-3 space-y-3 text-muted-foreground">
            <p>
              Application settings affect the next request or task. Job retention takes effect
              during the next hourly cleanup; reducing it deletes older completed and failed jobs.
            </p>
            <p>
              Environment changes require container recreation. Upload requests have a 64 MiB total
              limit, including metadata, and a 63 MiB limit per file.
            </p>
            <p className="break-all font-mono text-xs">
              Saved revision: {status?.saved_version ?? "Not reported"}
              <br />
              Applied revision: {status?.applied_version ?? "Not reported"}
            </p>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
