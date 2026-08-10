import { isAxiosError } from "axios";

/**
 * Extract a user-friendly error message from any error type.
 * Centralizes the scattered `any`-typed error casts across the codebase.
 */
export function getErrorMessage(error: unknown): string {
  if (isAxiosError(error)) {
    return error.response?.data?.detail || error.message || "Request failed";
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "An unexpected error occurred";
}
