import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { apiClient } from "../client";
import { queryKeys } from "../query-keys";
import type { DocumentResponse, JobResponse, UploadStatusResponse } from "../types";

export function useUploadableClients() {
  return useQuery({
    queryKey: queryKeys.documents.clients(),
    queryFn: async () =>
      (await apiClient.get<{ id: number; email: string }[]>("/documents/clients")).data,
  });
}

export function useDocuments() {
  return useQuery({
    queryKey: queryKeys.documents.list(),
    queryFn: async () => (await apiClient.get<DocumentResponse[]>("/documents")).data,
  });
}

export function useDocument(id: number) {
  return useQuery({
    queryKey: queryKeys.documents.detail(id),
    queryFn: async () => (await apiClient.get<DocumentResponse>(`/documents/${id}`)).data,
    enabled: !!id,
  });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({
      file,
      visibility,
      groupId,
      clientId,
      renameOnConflict,
      docDomain,
    }: {
      file: File;
      visibility: string;
      groupId?: number | null;
      clientId?: number | null;
      renameOnConflict?: boolean;
      docDomain?: string | null;
    }) => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("visibility", visibility);
      if (groupId != null) fd.append("group_id", String(groupId));
      if (clientId != null) fd.append("client_id", String(clientId));
      if (renameOnConflict) fd.append("rename_on_conflict", "true");
      if (docDomain) fd.append("doc_domain", docDomain);
      return (
        await apiClient.post<UploadStatusResponse>("/documents", fd, {
          headers: { "Content-Type": "multipart/form-data" },
        })
      ).data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.documents.all }),
  });
}

interface DeletionJob {
  document_id: number;
  job_id: number;
  status: JobResponse["status"];
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  const [pending, setPending] = useState<DeletionJob[]>([]);
  const jobs = useQueries({
    queries: pending.map((job) => ({
      queryKey: ["documents", "deletion", job.document_id, job.job_id],
      queryFn: async () => {
        try {
          return (
            await apiClient.get<DeletionJob>(`/documents/${job.document_id}/deletion/${job.job_id}`)
          ).data;
        } catch (error) {
          if (isAxiosError(error) && error.response?.status === 404) {
            return { ...job, status: "done" as const };
          }
          throw error;
        }
      },
      retry: 2,
      refetchInterval: 2000,
    })),
  });
  useEffect(() => {
    const finished = new Set<number>();
    jobs.forEach((query, index) => {
      if (query.data?.status === "done") {
        toast.success(`Document #${pending[index].document_id} deleted`);
        finished.add(pending[index].job_id);
      } else if (query.data?.status === "failed" || query.isError) {
        toast.error(`Could not delete document #${pending[index].document_id}. Refresh and retry.`);
        finished.add(pending[index].job_id);
      }
    });
    if (finished.size) {
      setPending((current) => current.filter((job) => !finished.has(job.job_id)));
      void qc.invalidateQueries({ queryKey: queryKeys.documents.all });
      void qc.invalidateQueries({ queryKey: queryKeys.jobs.all });
    }
  }, [jobs, pending, qc]);
  const mutation = useMutation({
    mutationFn: async (id: number) =>
      (await apiClient.delete<DeletionJob>(`/documents/${id}`)).data,
    onSuccess: (job) => {
      setPending((current) => [...current, job]);
      void qc.invalidateQueries({ queryKey: queryKeys.jobs.all });
    },
  });
  return {
    ...mutation,
    isDeleting: (id: number) => pending.some((job) => job.document_id === id),
  };
}

export function useRenameDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, filename }: { id: number; filename: string }) =>
      (await apiClient.patch<DocumentResponse>(`/documents/${id}/rename`, { filename })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.documents.all }),
  });
}
