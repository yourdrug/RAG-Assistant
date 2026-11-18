import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../client";

// ─── Types ────────────────────────────────────────────────────────────────

export const DIAGNOSIS_CLASSES = [
  "retrieval_miss",
  "false_refusal",
  "context_loss",
  "generation_error",
  "false_answer",
  "unlabeled",
  "ok",
] as const;

export type DiagnosisClass = (typeof DIAGNOSIS_CLASSES)[number];

export interface DiagnosisTransition {
  from_run_id: number;
  to_run_id: number;
  compared: number;
  matrix: Record<string, Record<string, number>>;
  changes: Array<{
    key: string;
    id: string | number | null;
    question: string | null;
    from: string;
    to: string;
  }>;
}

export interface AcceptanceThresholds {
  min_fragment_recall_at_k: number | null;
  min_context_fragment_recall: number | null;
  min_correctness: number | null;
  max_false_answer_rate: number | null;
}

export interface DatasetPolicy {
  dataset: string;
  thresholds: AcceptanceThresholds;
  noise: {
    sigma: Record<string, number>;
    judge_model: string | null;
    judge_rubric_version: string | null;
    run_ids: number[];
    measured_at: string;
  } | null;
  calibration: JudgeCalibration | null;
  updated_by: number | null;
  update_date: string | null;
}

export interface JudgeCalibration {
  id?: number | null;
  run_id: number | null;
  judge_model: string | null;
  judge_rubric_version: string | null;
  metrics: Record<string, AgreementMetric>;
  measured_at: string;
  created_by?: number | null;
}

export interface DatasetComposition {
  dataset: string;
  total: number;
  counts: Record<string, number>;
  shares: Record<string, number>;
  failures: string[];
  passed: boolean;
}

export type LabelMetric = "correctness" | "faithfulness";

export interface HumanLabel {
  id: number | null;
  run_id: number;
  question_key: string;
  metric: LabelMetric;
  score: number;
  labeler_id: number;
  note: string | null;
  update_date: string;
}

export interface AgreementMetric {
  n: number;
  mae: number | null;
  spearman: number | null;
  kappa: number | null;
  min_kappa?: number;
}

export interface JudgeAgreement {
  run_id: number;
  dataset: string;
  judge_model: string | null;
  judge_rubric_version: string | null;
  metrics: Record<LabelMetric, AgreementMetric>;
}

/** Same identity as the server's `result_key`: question id (or case/question text) and repeat index. */
export function resultKey(row: {
  id?: unknown;
  case_id?: unknown;
  question?: unknown;
  run?: unknown;
}) {
  const identity =
    row.id !== undefined && row.id !== null && row.id !== ""
      ? row.id
      : (row.case_id ?? row.question);
  return `${identity}#${row.run ?? 1}`;
}

// ─── Dataset policy ───────────────────────────────────────────────────────

const datasetPath = (dataset: string) => `/admin/benchmark/datasets/${encodeURIComponent(dataset)}`;

export function useDatasetPolicy(dataset: string) {
  return useQuery({
    queryKey: ["benchmark", "policy", dataset],
    queryFn: async () =>
      (await apiClient.get<DatasetPolicy>(`${datasetPath(dataset)}/policy`)).data,
    enabled: Boolean(dataset),
  });
}

export function useJudgeCalibrations(dataset: string) {
  return useQuery({
    queryKey: ["benchmark", "calibrations", dataset],
    queryFn: async () =>
      (
        await apiClient.get<{ dataset: string; calibrations: JudgeCalibration[] }>(
          `${datasetPath(dataset)}/calibrations`,
        )
      ).data.calibrations,
    enabled: Boolean(dataset),
  });
}

export function useDatasetComposition(dataset: string) {
  return useQuery({
    queryKey: ["benchmark", "composition", dataset],
    queryFn: async () =>
      (await apiClient.get<DatasetComposition>(`${datasetPath(dataset)}/composition`)).data,
    enabled: Boolean(dataset),
  });
}

export function useUpdateThresholds(dataset: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (thresholds: AcceptanceThresholds) =>
      (await apiClient.put<DatasetPolicy>(`${datasetPath(dataset)}/policy/thresholds`, thresholds))
        .data,
    onSuccess: (policy) => queryClient.setQueryData(["benchmark", "policy", dataset], policy),
  });
}

export function useMeasureNoise(dataset: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (runIds: number[]) =>
      (
        await apiClient.post<DatasetPolicy>(`${datasetPath(dataset)}/policy/noise`, {
          run_ids: runIds,
        })
      ).data,
    onSuccess: (policy) => queryClient.setQueryData(["benchmark", "policy", dataset], policy),
  });
}

// ─── Human labels and judge agreement ─────────────────────────────────────

export function useRunLabels(runId: number | null) {
  return useQuery({
    queryKey: ["benchmark", "labels", runId],
    queryFn: async () =>
      (await apiClient.get<{ labels: HumanLabel[] }>(`/admin/benchmark/runs/${runId}/labels`)).data
        .labels,
    enabled: runId != null,
  });
}

export function useJudgeAgreement(runId: number | null) {
  return useQuery({
    queryKey: ["benchmark", "agreement", runId],
    queryFn: async () =>
      (await apiClient.get<JudgeAgreement>(`/admin/benchmark/runs/${runId}/agreement`)).data,
    enabled: runId != null,
  });
}

export function useSaveLabel(runId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      question_key: string;
      metric: LabelMetric;
      score: number;
      note?: string | null;
    }) => (await apiClient.put<HumanLabel>(`/admin/benchmark/runs/${runId}/labels`, body)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["benchmark", "labels", runId] });
      queryClient.invalidateQueries({ queryKey: ["benchmark", "agreement", runId] });
    },
  });
}

export function useDeleteLabel(runId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: { question_key: string; metric: LabelMetric }) =>
      apiClient.delete(`/admin/benchmark/runs/${runId}/labels`, { params }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["benchmark", "labels", runId] });
      queryClient.invalidateQueries({ queryKey: ["benchmark", "agreement", runId] });
    },
  });
}

export function useSaveCalibration(runId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () =>
      (await apiClient.post<DatasetPolicy>(`/admin/benchmark/runs/${runId}/calibration`)).data,
    onSuccess: (policy) => {
      queryClient.setQueryData(["benchmark", "policy", policy.dataset], policy);
      queryClient.invalidateQueries({ queryKey: ["benchmark", "calibrations", policy.dataset] });
    },
  });
}
