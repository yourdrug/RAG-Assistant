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
  },
  conversations: {
    all: ["conversations"] as const,
    history: (id: number) => [...queryKeys.conversations.all, "history", id] as const,
  },
  groups: {
    all: ["groups"] as const,
    list: () => [...queryKeys.groups.all, "list"] as const,
    members: (id: number) => [...queryKeys.groups.all, "members", id] as const,
  },
  clients: {
    all: ["clients"] as const,
    assignments: (id: number) => [...queryKeys.clients.all, "assignments", id] as const,
  },
  ingest: {
    all: ["ingest"] as const,
    registry: () => [...queryKeys.ingest.all, "registry"] as const,
  },
  health: {
    all: ["health"] as const,
  },
} as const;
