"use client";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCcw, Save } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";
import { apiClient } from "@/shared/api/client";
import { getErrorMessage } from "@/shared/lib/errors";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { QueryError } from "@/shared/ui/query-error";
import { Skeleton } from "@/shared/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/tabs";
import { ConfigurationDiagnostics, type EffectiveParameter } from "./configuration-diagnostics";
import { CorpusControl } from "./corpus-control";
import { LLM_MODELS_QUERY_KEY, llmModelLabel, useLLMModels } from "./llm-models";

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

interface SettingsBlock {
  id: string;
  label: string;
  description: string;
  categories?: string[];
  keys?: string[];
  lastKey?: string;
}

const LLM_CONNECTION_BLOCK = "llm_connection";
const MODEL_PICKER_KEYS = new Set(["llm_model", "llm_auxiliary_model"]);
const RERANK_API_FORMATS: { value: string; label: string }[] = [
  { value: "cohere", label: "Cohere-compatible /rerank (vLLM, Jina, Cohere, Infinity)" },
  { value: "tei", label: "Text Embeddings Inference /rerank" },
  { value: "deepinfra", label: "DeepInfra /inference/{model}" },
];
/** Deployment-owned credentials shown (presence only) inside their connection block. */
const BLOCK_SECRETS: Record<string, { name: string; env: string; keyless: string }> = {
  [LLM_CONNECTION_BLOCK]: { name: "llm_api_key", env: "LLM_API_KEY", keyless: "vLLM, Ollama" },
  embeddings: { name: "embedding_api_key", env: "EMBEDDING_API_KEY", keyless: "local TEI" },
  "reranking-service": { name: "rerank_api_key", env: "RERANK_API_KEY", keyless: "local TEI" },
};

const SETTINGS_SECTIONS: {
  id: string;
  label: string;
  description: string;
  blocks: SettingsBlock[];
}[] = [
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
        keys: ["llm_model", "llm_auxiliary_model"],
        lastKey: "llm_disable_reasoning",
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
        id: LLM_CONNECTION_BLOCK,
        label: "LLM endpoint",
        description:
          "Any OpenAI-compatible API: OpenRouter, OpenAI, vLLM, Ollama, LM Studio. The URL origin must be approved by the deployment (LLM_ALLOWED_ORIGINS / ML_INTERNAL_ALLOWED_ORIGINS). Models are chosen in Chat & answers → Answer generation.",
        categories: ["llm_connection"],
      },
      {
        id: "embeddings",
        label: "Embedding endpoint",
        description:
          "OpenAI-compatible /embeddings API (TEI /v1, DeepInfra /v1/openai, vLLM, OpenAI). The model and dimension are fixed by the deployment (EMBEDDING_MODEL, EMBED_DIM) because they define the search collection.",
        keys: [
          "embedding_base_url",
          "embedding_http_max_connections",
          "embedding_http_max_keepalive_connections",
        ],
      },
      {
        id: "reranking-service",
        label: "Reranking endpoint",
        description:
          "Model used to refine the order of retrieved documents. Choose the API format the server speaks.",
        keys: [
          "rerank_base_url",
          "rerank_model",
          "rerank_api_format",
          "rerank_http_max_connections",
          "rerank_http_max_keepalive_connections",
        ],
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
];

const PARAMETER_LABELS: Record<string, string> = {
  llm_base_url: "Endpoint URL",
  embedding_base_url: "Endpoint URL",
  rerank_base_url: "Endpoint URL",
  rerank_model: "Reranker model",
  rerank_api_format: "API format",
  llm_model: "Answer model",
  llm_auxiliary_model: "Auxiliary model",
  llm_temperature: "Creativity (temperature)",
  llm_top_p: "Token diversity (top-p)",
  llm_disable_reasoning: "Answer without reasoning (thinking off)",
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

const SELECT_CLASS =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";
const PARAMETER_ROW_LAYOUT =
  "grid grid-cols-1 gap-4 border-l-2 px-6 py-5 @min-[36rem]:grid-cols-[minmax(0,1fr)_minmax(260px,340px)] @min-[36rem]:gap-6";

export function AdminSettingsPage() {
  const queryClient = useQueryClient();
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
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

  const parameters = useQuery({
    queryKey: ["admin", "config"],
    queryFn: async () => (await apiClient.get<ConfigParam[]>("/admin/config")).data,
  });

  const { data: params, isLoading } = parameters;
  const modelCatalog = useLLMModels();
  const { data: llmModels, isLoading: isLoadingModels } = modelCatalog;

  const setEdit = (p: ConfigParam, value: string) =>
    setEdits((prev) => ({ ...prev, [editKey(p)]: value }));

  const discard = (ids: string[]) =>
    setEdits((prev) => {
      const next = { ...prev };
      for (const id of ids) delete next[id];
      return next;
    });

  /** Saves the given edits one by one; a failure keeps it and the rest for review. */
  const saveEdits = async (ids: string[]) => {
    const changes = (params ?? []).filter(
      (p) => ids.includes(editKey(p)) && edits[editKey(p)] !== undefined,
    );
    if (changes.length === 0) return;
    setSaving(true);
    const saved: string[] = [];
    try {
      for (const p of changes) {
        const url = p.domain_key
          ? `/admin/config/${p.key}?domain=${p.domain_key}`
          : `/admin/config/${p.key}`;
        await apiClient.put(url, { value: edits[editKey(p)] });
        saved.push(editKey(p));
      }
      toast.success(
        saved.length === 1
          ? `"${changes[0].key}" saved; checking application`
          : `${saved.length} settings saved; checking application`,
      );
    } catch (error) {
      const failed = changes[saved.length];
      toast.error(
        `${parameterLabel(failed)}: ${getErrorMessage(error, "Could not save the setting.")}`,
      );
    } finally {
      setSaving(false);
      if (saved.length > 0) {
        discard(saved);
        queryClient.invalidateQueries({ queryKey: ["admin", "config"] });
        queryClient.invalidateQueries({ queryKey: ["admin", "config-status"] });
        queryClient.invalidateQueries({ queryKey: ["admin", "models"] });
        queryClient.invalidateQueries({ queryKey: LLM_MODELS_QUERY_KEY });
      }
    }
  };

  const renderValue = (p: ConfigParam) => {
    const isEdited = edits[editKey(p)] !== undefined;
    const displayValue = isEdited ? edits[editKey(p)] : p.value;

    if (p.value_type === "bool") {
      const enabled = displayValue === "true";
      return (
        <div className="inline-flex rounded-lg border bg-muted/40 p-0.5">
          <Button
            aria-label={`Enable ${parameterLabel(p)}`}
            aria-pressed={enabled}
            size="sm"
            variant={enabled ? "default" : "ghost"}
            className="h-7 min-w-14"
            onClick={() => setEdit(p, "true")}
          >
            On
          </Button>
          <Button
            aria-label={`Disable ${parameterLabel(p)}`}
            aria-pressed={!enabled}
            size="sm"
            variant={!enabled ? "secondary" : "ghost"}
            className={`h-7 min-w-14 ${!enabled ? "bg-background shadow-sm" : ""}`}
            onClick={() => setEdit(p, "false")}
          >
            Off
          </Button>
        </div>
      );
    }

    if (p.key === "rerank_api_format") {
      return (
        <select
          aria-label={parameterLabel(p)}
          value={displayValue}
          onChange={(e) => setEdit(p, e.target.value)}
          className={SELECT_CLASS}
        >
          {RERANK_API_FORMATS.map((format) => (
            <option key={format.value} value={format.value}>
              {format.label}
            </option>
          ))}
        </select>
      );
    }

    // Models served by the configured endpoint (GET /models); free text when it lists none.
    if (MODEL_PICKER_KEYS.has(p.key)) {
      if (isLoadingModels) {
        return <Skeleton className="h-9 w-full" />;
      }

      const auxiliary = p.key === "llm_auxiliary_model";
      const models = llmModels?.models ?? [];

      if (models.length > 0) {
        return (
          <div className="space-y-1.5">
            <select
              aria-label={parameterLabel(p)}
              value={displayValue}
              onChange={(e) => setEdit(p, e.target.value)}
              className={SELECT_CLASS}
            >
              {auxiliary && <option value="">Same as answer model</option>}
              {displayValue && !models.some((m) => m.id === displayValue) && (
                <option value={displayValue}>{displayValue}</option>
              )}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {llmModelLabel(m)}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              {models.length} models served by the endpoint.
              {models.some((m) => m.pricing) && " Pricing per 1M tokens."}
            </p>
          </div>
        );
      }

      return (
        <div className="space-y-1.5">
          <Input
            aria-label={parameterLabel(p)}
            type="text"
            value={displayValue}
            onChange={(e) => setEdit(p, e.target.value)}
            className="w-full"
            placeholder={
              auxiliary ? "Empty: same as answer model" : "e.g. qwen/qwen3-30b-a3b-instruct-2507"
            }
          />
          <p className="text-xs text-muted-foreground">
            The endpoint does not list models; enter the model id.
          </p>
        </div>
      );
    }

    if (p.value_type === "str") {
      return (
        <Input
          aria-label={parameterLabel(p)}
          type="text"
          value={displayValue}
          onChange={(e) => setEdit(p, e.target.value)}
          className="w-full"
        />
      );
    }

    return (
      <div className="flex flex-wrap items-center gap-3">
        <Input
          aria-label={parameterLabel(p)}
          type="number"
          step={p.value_type === "float" ? "0.1" : "1"}
          value={displayValue}
          onChange={(e) => setEdit(p, e.target.value)}
          className="w-32"
        />
        {p.min_value !== null && p.max_value !== null && (
          <span className="whitespace-nowrap text-xs text-muted-foreground">
            Range {p.min_value} – {p.max_value}
          </span>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <div>
          <Skeleton className="mb-2 h-8 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
        {[1, 2].map((i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <Skeleton className="h-40 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const availableSections = SETTINGS_SECTIONS.filter((section) =>
    section.blocks.some((block) => params?.some((param) => parameterBlock(param) === block.id)),
  );
  const pendingIds = Object.keys(edits);

  const renderParameter = (param: ConfigParam) => {
    const id = editKey(param);
    const isEdited = edits[id] !== undefined;
    return (
      <div
        key={id}
        className={`${PARAMETER_ROW_LAYOUT} ${
          isEdited ? "border-l-primary bg-muted/40" : "border-l-transparent"
        }`}
      >
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium leading-6">{parameterLabel(param)}</p>
            {isEdited && <Badge variant="secondary">Unsaved</Badge>}
          </div>
          {param.description && (
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              {param.description}
            </p>
          )}
          <code className="mt-2 inline-block max-w-full wrap-anywhere rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
            {param.key}
          </code>
        </div>
        <div className="min-w-0 space-y-3" role="group" aria-label={parameterLabel(param)}>
          {renderValue(param)}
          {isEdited && (
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={() => void saveEdits([id])} disabled={saving}>
                <Save className="h-3.5 w-3.5" />
                Save
              </Button>
              <Button size="sm" variant="ghost" onClick={() => discard([id])} disabled={saving}>
                <RotateCcw className="h-3.5 w-3.5" />
                Discard
              </Button>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="mt-1 text-muted-foreground">
            Configure models, answer generation and document search.
          </p>
        </div>
        <div
          role="status"
          aria-live="polite"
          className="flex max-w-xl items-center gap-3 rounded-lg border bg-card px-4 py-2.5 text-sm"
        >
          <Badge
            className="shrink-0"
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
      </div>

      {parameters.isError && (
        <QueryError query={parameters} fallback="Could not load editable settings." />
      )}
      {modelCatalog.isError && (
        <QueryError
          query={modelCatalog}
          fallback="Could not load the model list. You can enter a model name manually."
        />
      )}
      <Tabs
        defaultValue={availableSections[0]?.id ?? "corpus"}
        orientation="vertical"
        className="flex flex-col gap-8 lg:flex-row lg:items-start"
      >
        <TabsList
          aria-label="Settings sections"
          className="h-auto w-full shrink-0 flex-row flex-wrap justify-start gap-1 rounded-xl border bg-card p-2 lg:sticky lg:top-6 lg:w-64 lg:flex-col lg:items-stretch"
        >
          {availableSections.map((section) => (
            <TabsTrigger
              key={section.id}
              value={section.id}
              className="flex-col items-start gap-0.5 whitespace-normal rounded-lg px-3 py-2.5 text-left data-[state=active]:bg-muted data-[state=active]:shadow-none"
            >
              <span>{section.label}</span>
              <span className="hidden text-xs font-normal leading-snug text-muted-foreground lg:block">
                {section.description}
              </span>
            </TabsTrigger>
          ))}
          <div className="my-1 hidden border-t lg:block" />
          <TabsTrigger
            value="corpus"
            className="justify-start rounded-lg px-3 py-2.5 data-[state=active]:bg-muted data-[state=active]:shadow-none"
          >
            Search collection
          </TabsTrigger>
          <TabsTrigger
            value="diagnostics"
            className="justify-start rounded-lg px-3 py-2.5 data-[state=active]:bg-muted data-[state=active]:shadow-none"
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
                if (block.lastKey) {
                  blockParams.sort(
                    (a, b) => Number(a.key === block.lastKey) - Number(b.key === block.lastKey),
                  );
                }
                const globalParams = blockParams.filter((param) => !param.domain_key);
                const domains = [
                  ...new Set(
                    blockParams.flatMap((param) => (param.domain_key ? [param.domain_key] : [])),
                  ),
                ];
                return (
                  <Card key={block.id} className="overflow-hidden">
                    <CardHeader className="space-y-1 border-b bg-muted/20 px-6 py-4">
                      <CardTitle className="text-base">{block.label}</CardTitle>
                      <CardDescription className="leading-relaxed">
                        {block.description}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="@container p-0">
                      <div className="divide-y">
                        {globalParams.map(renderParameter)}
                        {BLOCK_SECRETS[block.id] && (
                          <SecretKeyRow
                            {...BLOCK_SECRETS[block.id]}
                            present={
                              effective?.find(
                                (p) => p.key === BLOCK_SECRETS[block.id].name && !p.domain_key,
                              )?.present ?? null
                            }
                          />
                        )}
                      </div>
                      {domains.map((domain) => (
                        <div key={domain} className="border-t">
                          <div className="bg-muted/30 px-6 py-3">
                            <p className="text-sm font-semibold">
                              {DOMAIN_LABELS[domain] ?? domain} overrides
                            </p>
                            <p className="text-xs text-muted-foreground">
                              Applies only to documents in this domain.
                            </p>
                          </div>
                          <div className="divide-y border-t">
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
            </TabsContent>
          ))}
          <TabsContent value="corpus" className="mt-0">
            <CorpusControl />
          </TabsContent>
          <TabsContent value="diagnostics" className="mt-0">
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

      {pendingIds.length > 0 && (
        <div className="sticky bottom-4 z-20 flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-background/95 px-4 py-3 shadow-lg backdrop-blur">
          <span className="text-sm">
            {pendingIds.length} unsaved {pendingIds.length === 1 ? "change" : "changes"}
          </span>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={() => discard(pendingIds)} disabled={saving}>
              <RotateCcw className="h-3.5 w-3.5" />
              Discard all
            </Button>
            <Button size="sm" onClick={() => void saveEdits(pendingIds)} disabled={saving}>
              <Save className="h-3.5 w-3.5" />
              {saving ? "Saving…" : "Save all"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

/** The key is deployment-owned (server/.env.secrets); the API exposes only its presence. */
function SecretKeyRow({
  name,
  env,
  keyless,
  present,
}: {
  name: string;
  env: string;
  keyless: string;
  present: boolean | null;
}) {
  return (
    <div className={`${PARAMETER_ROW_LAYOUT} border-l-transparent`}>
      <div className="min-w-0">
        <p className="text-sm font-medium leading-6">API key</p>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          Set {env} in server/.env.secrets and restart. Leave it empty for keyless servers (
          {keyless}).
        </p>
        <code className="mt-2 inline-block max-w-full wrap-anywhere rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
          {name}
        </code>
      </div>
      <div className="flex min-w-0 items-center gap-2">
        <Input
          aria-label="API key"
          type="password"
          value={present ? "••••••••••••" : ""}
          placeholder="Not set"
          readOnly
          disabled
          className="w-full"
        />
        <Badge variant={present === null ? "secondary" : present ? "success" : "warning"}>
          {present === null ? "Unknown" : present ? "Set" : "Not set"}
        </Badge>
      </div>
    </div>
  );
}
