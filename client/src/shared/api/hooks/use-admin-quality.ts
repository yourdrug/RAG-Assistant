import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  DocumentDiagnoseResponse,
  DocumentQualityListResponse,
  DryRunResponse,
} from "../types";

export function useQualityDocuments() {
  return useQuery({
    queryKey: queryKeys.quality.list(),
    queryFn: async () =>
      (await apiClient.get<DocumentQualityListResponse>("/admin/documents/quality")).data,
  });
}

export function useDocumentDiagnosis(documentId: number) {
  return useQuery({
    queryKey: queryKeys.quality.diagnose(documentId),
    queryFn: async () =>
      (
        await apiClient.post<DocumentDiagnoseResponse>(
          `/admin/documents/${documentId}/diagnose`,
        )
      ).data,
    enabled: !!documentId,
  });
}

/** Phase 1: instant text-layer analysis (no OCR) */
export function useDryRun() {
  return useMutation({
    mutationFn: async (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return (
        await apiClient.post<DryRunResponse>("/admin/documents/preview", fd, {
          headers: { "Content-Type": "multipart/form-data" },
        })
      ).data;
    },
  });
}

/** Phase 2: run OCR on specific problem pages */
export function useDryRunOcr() {
  return useMutation({
    mutationFn: async ({ file, pages }: { file: File; pages: number[] }) => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("pages", pages.join(","));
      return (
        await apiClient.post<DryRunResponse>("/admin/documents/preview-ocr", fd, {
          headers: { "Content-Type": "multipart/form-data" },
        })
      ).data;
    },
  });
}
