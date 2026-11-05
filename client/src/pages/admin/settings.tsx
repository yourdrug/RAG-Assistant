"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCcw, Save } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import { apiClient } from "@/shared/api/client";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Skeleton } from "@/shared/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/tabs";
import { ConfigurationDiagnostics, type EffectiveParameter } from "./configuration-diagnostics";
import { CorpusControl } from "./corpus-control";
import { CostRates } from "./cost-rates";

interface ConfigParam {
  key: string;
  value: string;
  value_type: string;
  category: string;
  description: string | null;
  min_value: number | null;
  max_value: number | null;
  domain_key: string | null;
}

interface ConfigurationStatus {
  saved_version: string | null;
  applied_version: string | null;
  ready: boolean;
  error: string | null;
}

interface OpenRouterModel {
  id: string;
  name: string;
  context_length: number;
  pricing: { prompt: number; completion: number };
}

interface SettingsBlock {
  id: string;
  label: string;
  description: string;
  categories?: string[];
  keys?: string[];
}

const SETTINGS_SECTIONS = [
  {
    id: "chat",
    label: "Chat & answers",
    description: "Control conversation memory, answer generation and privacy.",
    blocks: [
      {
        id: "generation",
        label: "Answer generation",
        description: "Model selection, creativity and response length.",
        categories: ["llm"],
        keys: ["openrouter_model"],
      },
      {
        id: "conversation",
        label: "Conversation & memory",
        description: "Use previous messages to understand follow-up questions.",
        keys: ["history_window", "condense_enabled", "rolling_summary_enabled"],
      },
      {
        id: "privacy",
        label: "Privacy & caching",
        description: "Personal data protection and reuse of previous answers.",
        keys: ["pii_redaction_enabled", "cache_enabled"],
      },
    ],
  },
  {
    id: "search",
    label: "Search & sources",
    description: "Tune how documents are found, ranked and selected as sources.",
    blocks: [
      {
        id: "retrieval",
        label: "Document retrieval",
        description: "Candidate and result counts for focused and broad questions.",
        categories: ["rag"],
        keys: ["decomposition_enabled", "relevance_gate_enabled"],
      },
      {
        id: "hybrid",
        label: "Hybrid search",
        description: "Combine semantic similarity with keyword matches.",
        categories: ["hybrid"],
      },
      {
        id: "ranking",
        label: "Ranking & citations",
        description: "Refine result ordering and filter sources for the answer.",
        categories: ["reranker"],
      },
    ],
  },
  {
    id: "documents",
    label: "Document processing",
    description: "Configure uploads, text splitting, indexing and scanned pages.",
    blocks: [
      {
        id: "uploads",
        label: "Uploads & previews",
        description: "File size limits and preview lifetime.",
        keys: ["max_upload_size_mb", "preview_max_size_mb", "preview_cache_ttl_seconds"],
      },
      {
        id: "indexing",
        label: "Text splitting & indexing",
        description: "Split document text into overlapping chunks and create embeddings.",
        categories: ["ingestion"],
        keys: ["chunk_size", "chunk_overlap"],
      },
      {
        id: "ocr",
        label: "Scanned pages (OCR)",
        description: "Recognize text in images and pages without a usable text layer.",
        categories: ["ocr"],
      },
    ],
  },
  {
    id: "models",
    label: "Model connections",
    description: "Service endpoints and embedding and reranking model connections.",
    blocks: [
      {
        id: "openrouter",
        label: "OpenRouter connection",
        description: "Connection used by the answer model selected in Chat & answers.",
        categories: ["openrouter"],
      },
      {
        id: "embeddings",
        label: "Embedding service",
        description: "Convert document chunks and questions into vectors for semantic search.",
        keys: [
          "tei_embed_url",
          "deepinfra_embed_model",
          "embedding_http_max_connections",
          "embedding_http_max_keepalive_connections",
        ],
      },
      {
        id: "reranking-service",
        label: "Reranking service",
        description: "Model connection used to refine the order of retrieved documents.",
        keys: [
          "tei_rerank_url",
          "deepinfra_rerank_model",
          "rerank_http_max_connections",
          "rerank_http_max_keepalive_connections",
        ],
      },
      {
        id: "ml",
        label: "Shared ML connection",
        description: "Provider endpoint shared by embedding and reranking models.",
        categories: ["ml"],
      },
    ],
  },
  {
    id: "costs",
    label: "Spending & pricing",
    description: "Set user spending limits and maintain model prices.",
    blocks: [
      {
        id: "cost",
        label: "User spending limits",
        description: "Hourly and daily USD budgets, including pending reservations.",
        categories: ["cost"],
      },
    ],
  },
  {
    id: "system",
    label: "System & limits",
    description: "Request limits, background job recovery and retention.",
    blocks: [
      {
        id: "requests",
        label: "Request limits",
        description: "Limit how often users can sign in, chat, search and upload.",
        categories: ["rate_limit"],
      },
      {
        id: "jobs",
        label: "Background jobs",
        description: "Recover stuck work and clean up old job records.",
        categories: ["jobs"],
      },
      {
        id: "storage",
        label: "Storage",
        description: "Application storage settings.",
        categories: ["storage"],
      },
      {
        id: "other",
        label: "Other settings",
        description: "Additional application parameters.",
        categories: ["toggles"],
      },
    ],
  },
] satisfies { id: string; label: string; description: string; blocks: SettingsBlock[] }[];

const PARAMETER_LABELS: Record<string, string> = {
  llm_provider: "Answer provider",
  openrouter_model: "Answer model",
  llm_temperature: "Creativity (temperature)",
  llm_top_p: "Token diversity (top-p)",
  llm_num_ctx_narrow: "Context tokens · focused questions",
  llm_num_ctx_broad: "Context tokens · broad questions",
  llm_num_predict_narrow: "Maximum answer tokens · focused questions",
  llm_num_predict_broad: "Maximum answer tokens · broad questions",
  retriever_fetch_k: "Candidates · focused questions",
  retriever_fetch_k_broad: "Candidates · broad questions",
  retriever_top_k: "Results · focused questions",
  retriever_top_k_broad: "Results · broad questions",
  history_window: "Conversation history window",
  condense_enabled: "Rewrite follow-up questions",
  rolling_summary_enabled: "Summarize conversation history",
  decomposition_enabled: "Split complex questions",
  relevance_gate_enabled: "Check question relevance",
  pii_redaction_enabled: "Redact personal data",
  cache_enabled: "Cache answers",
  chunk_size: "Chunk size (characters)",
  chunk_overlap: "Chunk overlap (characters)",
  source_min_score: "Minimum source relevance",
  cost_rate_limit_enabled: "Enable spending limits",
  cost_hourly_limit: "Hourly limit per user (USD)",
  cost_daily_limit: "Daily limit per user (USD)",
};

const DOMAIN_LABELS: Record<string, string> = { legal: "Legal", decree: "Decree" };
const SETTINGS_BLOCKS: SettingsBlock[] = SETTINGS_SECTIONS.flatMap((section) => section.blocks);

function parameterBlock(param: ConfigParam): string | undefined {
  return (
    SETTINGS_BLOCKS.find((block) => block.keys?.includes(param.key)) ??
    SETTINGS_BLOCKS.find((block) =>
      block.categories?.includes(param.category.replace(/^domain:/, "")),
    )
  )?.id;
}

function parameterLabel(param: ConfigParam): string {
  const label = param.key.replaceAll("_", " ");
  return PARAMETER_LABELS[param.key] ?? label.charAt(0).toUpperCase() + label.slice(1);
}

function editKey(param: ConfigParam): string {
  return param.domain_key ? `${param.domain_key}:${param.key}` : param.key;
}

export function AdminSettingsPage() {
  const queryClient = useQueryClient();
  const [edits, setEdits] = useState<Record<string, string>>({});
  const {
    data: status,
    isError: statusUnavailable,
    refetch: refreshStatus,
  } = useQuery({
    queryKey: ["admin", "config-status"],
    queryFn: async () => (await apiClient.get<ConfigurationStatus>("/admin/config/status")).data,
    refetchInterval: 3000,
  });

  const { data: effective, isError: effectiveUnavailable } = useQuery({
    queryKey: ["admin", "config-effective"],
    queryFn: async () =>
      (await apiClient.get<EffectiveParameter[]>("/admin/config/effective")).data,
    refetchInterval: 3000,
  });

  const { data: params, isLoading } = useQuery({
    queryKey: ["admin", "config"],
    queryFn: async () => (await apiClient.get<ConfigParam[]>("/admin/config")).data,
  });

  const { data: openrouterData, isLoading: isLoadingOpenRouter } = useQuery({
    queryKey: ["admin", "openrouter-models"],
    queryFn: async () =>
      (await apiClient.get<{ models: OpenRouterModel[] }>("/admin/models/openrouter")).data,
  });

  const updateMutation = useMutation({
    mutationFn: async ({ key, value, domain }: { key: string; value: string; domain?: string }) => {
      const url = domain ? `/admin/config/${key}?domain=${domain}` : `/admin/config/${key}`;
      return (await apiClient.put(url, { value })).data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["admin", "config"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "config-status"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "models"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "openrouter-models"] });
      if (variables.key.startsWith("cost_")) {
        void queryClient.invalidateQueries({ queryKey: ["cost-models"] });
      }
      setEdits((prev) => {
        const next = { ...prev };
        delete next[variables.domain ? `${variables.domain}:${variables.key}` : variables.key];
        return next;
      });
      toast.success(`"${variables.key}" saved; checking application`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || error.response?.data?.detail || "Update failed");
    },
  });

  const handleSave = (key: string, domain?: string) => {
    const val = edits[domain ? `${domain}:${key}` : key];
    if (val === undefined) return;
    updateMutation.mutate({ key, value: val, domain });
  };

  const handleReset = (key: string) => {
    setEdits((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const renderValue = (p: ConfigParam) => {
    const isEdited = edits[editKey(p)] !== undefined;
    const displayValue = isEdited ? edits[editKey(p)] : p.value;

    if (p.value_type === "bool") {
      return (
        <div className="flex items-center gap-2">
          <Button
            aria-label={`Enable ${parameterLabel(p)}`}
            aria-pressed={displayValue === "true"}
            size="sm"
            variant={displayValue === "true" ? "default" : "outline"}
            onClick={() => setEdits((prev) => ({ ...prev, [editKey(p)]: "true" }))}
          >
            ON
          </Button>
          <Button
            aria-label={`Disable ${parameterLabel(p)}`}
            aria-pressed={displayValue === "false"}
            size="sm"
            variant={displayValue === "false" ? "destructive" : "outline"}
            onClick={() => setEdits((prev) => ({ ...prev, [editKey(p)]: "false" }))}
          >
            OFF
          </Button>
        </div>
      );
    }

    if (p.key === "llm_provider") {
      return <span className="text-sm">OpenRouter (cloud)</span>;
    }

    // Special handling for openrouter_model - show OpenRouter models dropdown
    if (p.key === "openrouter_model") {
      if (isLoadingOpenRouter) {
        return <Skeleton className="h-9 w-64" />;
      }

      if (openrouterData?.models && openrouterData.models.length > 0) {
        return (
          <div className="space-y-2">
            <select
              aria-label={parameterLabel(p)}
              value={displayValue}
              onChange={(e) => setEdits((prev) => ({ ...prev, [editKey(p)]: e.target.value }))}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {openrouterData.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.context_length.toLocaleString()} ctx, ${m.pricing.prompt.toFixed(2)}
                  /M)
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              {openrouterData.models.length} models available. Pricing per 1M tokens.
            </p>
          </div>
        );
      }

      // Fallback to text input if no models fetched
      return (
        <div className="space-y-2">
          <Input
            aria-label={parameterLabel(p)}
            type="text"
            value={displayValue}
            onChange={(e) => setEdits((prev) => ({ ...prev, [editKey(p)]: e.target.value }))}
            className="w-full"
            placeholder="e.g. qwen/qwen-2.5-7b-instruct"
          />
        </div>
      );
    }

    if (p.value_type === "str") {
      return (
        <Input
          aria-label={parameterLabel(p)}
          type="text"
          value={displayValue}
          onChange={(e) => setEdits((prev) => ({ ...prev, [editKey(p)]: e.target.value }))}
          className="w-full"
        />
      );
    }

    return (
      <div className="flex flex-wrap items-center gap-2">
        <Input
          aria-label={parameterLabel(p)}
          type="number"
          step={p.value_type === "float" ? "0.1" : "1"}
          value={displayValue}
          onChange={(e) => setEdits((prev) => ({ ...prev, [editKey(p)]: e.target.value }))}
          className="w-28"
        />
        {p.min_value !== null && p.max_value !== null && (
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {p.min_value} – {p.max_value}
          </span>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="p-[14px] space-y-6">
        <div>
          <Skeleton className="h-8 w-48 mb-2" />
          <Skeleton className="h-4 w-72" />
        </div>
        {[1, 2].map((i) => (
          <Card key={i}>
            <CardContent className="p-[14px]">
              <Skeleton className="h-40 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const availableSections = SETTINGS_SECTIONS.filter(
    (section) =>
      section.id === "costs" ||
      section.blocks.some((block) => params?.some((param) => parameterBlock(param) === block.id)),
  );

  const renderParameter = (param: ConfigParam) => {
    const id = editKey(param);
    const isEdited = edits[id] !== undefined;
    return (
      <div
        key={id}
        className={`grid items-center gap-3 py-[10px] md:grid-cols-[minmax(0,1fr)_minmax(240px,360px)] md:gap-8 ${isEdited ? "bg-muted/30" : ""}`}
      >
        <div className="min-w-0 space-y-1">
          <p className="text-sm font-medium">{parameterLabel(param)}</p>
          {param.description && (
            <p className="text-sm text-muted-foreground">{param.description}</p>
          )}
          <p className="break-all font-mono text-xs text-muted-foreground/70">{param.key}</p>
        </div>
        <div className="min-w-0 space-y-2" role="group" aria-label={parameterLabel(param)}>
          {renderValue(param)}
          {isEdited && (
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => handleSave(param.key, param.domain_key ?? undefined)}
                disabled={updateMutation.isPending}
              >
                <Save className="mr-2 h-3.5 w-3.5" />
                Save
              </Button>
              <Button size="sm" variant="ghost" onClick={() => handleReset(id)}>
                <RotateCcw className="mr-2 h-3.5 w-3.5" />
                Discard
              </Button>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="p-[14px] space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground">
          Configure application behavior, model pricing and document search.
        </p>
      </div>

      <div
        role="status"
        aria-live="polite"
        className="flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 text-sm"
      >
        <Badge
          variant={
            statusUnavailable
              ? "warning"
              : status?.error
                ? "destructive"
                : status?.ready
                  ? "success"
                  : "secondary"
          }
        >
          {statusUnavailable
            ? "Status unavailable"
            : !status
              ? "Checking settings…"
              : status.error
                ? "Application failed"
                : status.ready
                  ? "Settings applied"
                  : "Applying settings…"}
        </Badge>
        <span className="text-muted-foreground">
          {statusUnavailable
            ? "Could not check whether saved changes have been applied."
            : status?.error
              ? "Previous settings remain active. Review your changes and check diagnostics."
              : status?.ready
                ? "Saved changes are active on this server."
                : "Waiting for this server to apply saved changes."}
        </span>
      </div>

      <Tabs
        defaultValue={availableSections[0]?.id ?? "corpus"}
        orientation="vertical"
        className="flex flex-col gap-6 lg:flex-row lg:items-start"
      >
        <TabsList
          aria-label="Settings sections"
          className="h-auto w-full shrink-0 flex-row flex-wrap justify-start gap-1 bg-transparent p-0 lg:sticky lg:top-6 lg:w-56 lg:flex-col lg:items-stretch"
        >
          {availableSections.map((section) => (
            <TabsTrigger
              key={section.id}
              value={section.id}
              className="justify-start whitespace-normal px-3 py-2.5 text-left data-[state=active]:bg-muted data-[state=active]:shadow-none"
            >
              {section.label}
            </TabsTrigger>
          ))}
          <div className="hidden border-t my-2 lg:block" />
          <TabsTrigger
            value="corpus"
            className="justify-start px-3 py-2.5 data-[state=active]:bg-muted"
          >
            Search collection
          </TabsTrigger>
          <TabsTrigger
            value="diagnostics"
            className="justify-start px-3 py-2.5 data-[state=active]:bg-muted"
          >
            Diagnostics
          </TabsTrigger>
        </TabsList>

        <div className="min-w-0 flex-1">
          {availableSections.map((section) => (
            <TabsContent key={section.id} value={section.id} className="mt-0 space-y-6">
              <div>
                <h2 className="text-xl font-semibold">{section.label}</h2>
                <p className="mt-1 text-sm text-muted-foreground">{section.description}</p>
              </div>
              {section.blocks.map((block) => {
                const blockParams =
                  params?.filter((param) => parameterBlock(param) === block.id) ?? [];
                if (blockParams.length === 0) return null;
                const globalParams = blockParams.filter((param) => !param.domain_key);
                const domains = [
                  ...new Set(
                    blockParams.flatMap((param) => (param.domain_key ? [param.domain_key] : [])),
                  ),
                ];
                return (
                  <Card key={block.id}>
                    <CardHeader className="p-[14px] pb-0">
                      <CardTitle className="text-base">{block.label}</CardTitle>
                      <CardDescription>{block.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="p-[14px] pt-0">
                      <div className="divide-y">{globalParams.map(renderParameter)}</div>
                      {domains.map((domain) => (
                        <div key={domain} className="mt-4 rounded-lg border px-4">
                          <div className="border-b py-3">
                            <p className="text-sm font-medium">
                              {DOMAIN_LABELS[domain] ?? domain} overrides
                            </p>
                            <p className="text-xs text-muted-foreground">
                              Applies only to documents in this domain.
                            </p>
                          </div>
                          <div className="divide-y">
                            {blockParams
                              .filter((param) => param.domain_key === domain)
                              .map(renderParameter)}
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                );
              })}
              {section.id === "costs" && <CostRates />}
            </TabsContent>
          ))}
          <TabsContent value="corpus">
            <CorpusControl />
          </TabsContent>
          <TabsContent value="diagnostics">
            <ConfigurationDiagnostics
              parameters={effective}
              descriptions={params}
              unavailable={effectiveUnavailable}
              status={status}
              statusUnavailable={statusUnavailable}
              onRefresh={() => {
                void refreshStatus();
                void queryClient.invalidateQueries({ queryKey: ["admin", "config-effective"] });
              }}
            />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
