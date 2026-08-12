import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { IngestStatusResponse, IngestRegistryResponse } from "../types";

export function useIngestRegistry() {
  return useQuery({
    queryKey: queryKeys.ingest.registry(),
    queryFn: async () => (await apiClient.get<IngestRegistryResponse>("/ingest/registry")).data,
  });
}

export function useIngestAll() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ docsDir, reset, domain }: { docsDir: string; reset?: boolean; domain?: string }) =>
      (await apiClient.post<IngestStatusResponse>("/ingest", null, { params: { docs_dir: docsDir, reset: reset ?? false, domain: domain ?? "auto" } })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.ingest.all }),
  });
}

export function useIngestFile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ filePath, force, domain }: { filePath: string; force?: boolean; domain?: string }) =>
      (await apiClient.post<IngestStatusResponse>("/ingest/file", null, { params: { file_path: filePath, force: force ?? false, domain: domain ?? "auto" } })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.ingest.all }),
  });
}

export function useUploadFiles() {
  return useMutation({
    mutationFn: async (files: File[]) => {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      return (await apiClient.post<{ files: string[] }>("/upload", fd, { headers: { "Content-Type": "multipart/form-data" } })).data;
    },
  });
}
