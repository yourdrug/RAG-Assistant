import type { AnswerOutcome, FeedbackReason } from "@/shared/api/types";

export const FEEDBACK_REASON_LABELS: Record<FeedbackReason, string> = {
  no_answer: "No answer",
  wrong_fact: "Incorrect fact",
  irrelevant_sources: "Irrelevant sources",
  incomplete: "Incomplete answer",
  outdated: "Outdated information",
  too_long: "Too long",
  other: "Other",
};
export const OUTCOME_LABELS: Record<AnswerOutcome, string> = {
  generated: "Generated",
  grounded: "Grounded",
  cache_hit: "Cached",
  not_found: "No answer",
  out_of_domain: "Out of domain",
  input_blocked: "Input blocked",
  output_blocked: "Output blocked",
  error: "Error",
  cancelled: "Cancelled",
};
