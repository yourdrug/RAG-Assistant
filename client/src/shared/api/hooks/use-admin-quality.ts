import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type {
  DocumentDiagnoseResponse,
  DocumentQualityListResponse,
  DryRunResponse,
  IndexFromPreviewResponse,
  PageImageResponse,
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
      (await apiClient.post<DocumentDiagnoseResponse>(`/admin/documents/${documentId}/diagnose`))
        .data,
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
    mutationFn: async ({
      file,
      previewId,
      pages,
    }: {
      file?: File | null;
      previewId?: string;
      pages: number[];
    }) => {
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (previewId) fd.append("preview_id", previewId);
      fd.append("pages", pages.join(","));
      return (
        await apiClient.post<DryRunResponse>("/admin/documents/preview-ocr", fd, {
          headers: { "Content-Type": "multipart/form-data" },
        })
      ).data;
    },
  });
}

/** Fetch rendered page image (base64 PNG) */
export function usePageImage() {
  return useMutation({
    mutationFn: async ({ previewId, page }: { previewId: string; page: number }) => {
      const fd = new FormData();
      fd.append("preview_id", previewId);
      fd.append("page", String(page));
      return (
        await apiClient.post<PageImageResponse>("/admin/documents/preview/page-image", fd, {
          headers: { "Content-Type": "multipart/form-data" },
        })
      ).data;
    },
  });
}

/** Index the cached PDF through the standard ingestion pipeline */
export function useIndexFromPreview() {
  return useMutation({
    mutationFn: async ({
      previewId,
      visibility = "internal_public",
      groupId,
      clientId,
      docDomain,
    }: {
      previewId: string;
      visibility?: string;
      groupId?: number | null;
      clientId?: number | null;
      docDomain?: string | null;
    }) => {
      const fd = new FormData();
      fd.append("visibility", visibility);
      if (groupId != null) fd.append("group_id", String(groupId));
      if (clientId != null) fd.append("client_id", String(clientId));
      if (docDomain) fd.append("doc_domain", docDomain);
      return (
        await apiClient.post<IndexFromPreviewResponse>(
          `/admin/documents/preview/${previewId}/index`,
          fd,
          { headers: { "Content-Type": "multipart/form-data" } },
        )
      ).data;
    },
  });
}
