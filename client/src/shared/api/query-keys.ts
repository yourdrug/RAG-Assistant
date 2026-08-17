export const queryKeys = {
  auth: {
    all: ["auth"] as const,
    me: () => [...queryKeys.auth.all, "me"] as const,
    users: () => [...queryKeys.auth.all, "users"] as const,
  },
  documents: {
    all: ["documents"] as const,
    list: () => [...queryKeys.documents.all, "list"] as const,
    detail: (id: number) => [...queryKeys.documents.all, "detail", id] as const,
    clients: () => [...queryKeys.documents.all, "clients"] as const,
  },
  chunks: {
    all: (documentId: number) => ["chunks", documentId] as const,
    list: (documentId: number, limit?: number, offset?: number) =>
      [...queryKeys.chunks.all(documentId), "list", { limit, offset }] as const,
  },
  conversations: {
    all: ["conversations"] as const,
    list: () => [...queryKeys.conversations.all, "list"] as const,
    history: (id: number) => [...queryKeys.conversations.all, "history", id] as const,
  },
  groups: {
    all: ["groups"] as const,
    list: () => [...queryKeys.groups.all, "list"] as const,
    members: (id: number) => [...queryKeys.groups.all, "members", id] as const,
  },
  apiKeys: {
    all: ["apiKeys"] as const,
    clientKeys: (id: number) => [...queryKeys.apiKeys.all, "clientKeys", id] as const,
  },
  ingest: {
    all: ["ingest"] as const,
    registry: () => [...queryKeys.ingest.all, "registry"] as const,
  },
  health: {
    all: ["health"] as const,
  },
  jobs: {
    all: ["jobs"] as const,
    list: (params?: { limit?: number; offset?: number }) =>
      [...queryKeys.jobs.all, "list", params] as const,
    stats: () => [...queryKeys.jobs.all, "stats"] as const,
  },
  metrics: {
    all: ["metrics"] as const,
  },
  logs: {
    all: ["logs"] as const,
    list: (params?: { limit?: number; level?: string; search?: string }) =>
      [...queryKeys.logs.all, "list", params] as const,
  },
  chatLogs: {
    all: ["chatLogs"] as const,
    list: (params?: Record<string, unknown>) =>
      [...queryKeys.chatLogs.all, "list", params] as const,
  },
  admin: {
    all: ["admin"] as const,
    config: () => [...queryKeys.admin.all, "config"] as const,
    models: () => [...queryKeys.admin.all, "models"] as const,
    vectordb: () => [...queryKeys.admin.all, "vectordb"] as const,
  },
  benchmark: {
    all: ["benchmark"] as const,
    results: () => [...queryKeys.benchmark.all, "results"] as const,
    detail: (filename: string) => [...queryKeys.benchmark.all, "detail", filename] as const,
  },
  quality: {
    all: ["quality"] as const,
    list: () => [...queryKeys.quality.all, "list"] as const,
    diagnose: (id: number) => [...queryKeys.quality.all, "diagnose", id] as const,
  },
} as const;
