import { isAxiosError, isCancel } from "axios";

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asText(value: unknown): string | undefined {
  return typeof value === "string" ? value.trim() || undefined : undefined;
}

function validationDetails(body: Record<string, unknown>): string[] {
  const detail = Array.isArray(body.detail) ? body.detail : [];
  const legacyFields = detail.flatMap((item) => {
    const { loc, msg } = asRecord(item);
    const text = asText(msg);
    if (!text) return [];
    const field = Array.isArray(loc) ? loc.slice(1).join(".") : "";
    return [field ? `${field}: ${text}` : text];
  });
  const fields = Object.entries(asRecord(body.errors)).flatMap(([field, value]) => {
    const text = asText(value);
    return field !== "code" && text ? [`${field}: ${text}`] : [];
  });
  return [...new Set([...fields, ...legacyFields])];
}

export function describeHttpError(
  status: number,
  payload: unknown,
  retryAfter?: unknown,
  fallback = "The request failed. Please try again.",
): string {
  const body = asRecord(payload);
  const message = asText(body.message) ?? asText(body.detail);
  const seconds = typeof retryAfter === "string" ? Number(retryAfter) : retryAfter;
  const retry =
    typeof seconds === "number" && Number.isSafeInteger(seconds) && seconds > 0
      ? `Try again in ${seconds} seconds.`
      : "Please try again later.";

  if (status === 401) return "Your session has expired. Please sign in again.";
  if (status === 429) {
    const busy = asRecord(body.errors).code === "semaphore_timeout";
    return `${busy ? "The server is busy." : "Too many requests."} ${retry}`;
  }
  if (status === 502 || status === 503 || status === 504) {
    return `The service is temporarily unavailable. ${retry}`;
  }
  if (status >= 500) return "Something went wrong on the server. Please try again later.";
  if (status === 422) {
    const fields = validationDetails(body);
    return fields.length ? fields.join(" · ") : (message ?? "Check the submitted values.");
  }
  if (status === 403) return "You do not have permission to do this.";
  if (status === 408) return "The request timed out. Please try again.";
  if (status === 404) return message ?? "The requested item could not be found.";
  if (status === 409) return message ?? "The item has changed. Refresh and try again.";
  if (status === 413)
    return message ?? "The file or request is too large. Reduce the size and try again.";
  if (status === 415) return message ?? "This file format is not supported.";
  return message ?? fallback;
}

export function getErrorStatus(error: unknown): number | undefined {
  return isAxiosError(error) ? error.response?.status : undefined;
}

export function getErrorMessage(
  error: unknown,
  fallback = "Something went wrong. Please try again.",
): string {
  if (!isAxiosError(error)) return fallback;
  if (error.response) {
    return describeHttpError(
      error.response.status,
      error.response.data,
      error.response.headers?.["retry-after"],
      fallback,
    );
  }
  if (error.code === "ECONNABORTED" || error.code === "ETIMEDOUT") {
    return "The request timed out. Please try again.";
  }
  if (error.code === "ERR_NETWORK") {
    return "Could not reach the server. Check your connection and try again.";
  }
  return fallback;
}

export function shouldRetryRequest(failureCount: number, error: unknown): boolean {
  if (failureCount > 0 || isCancel(error)) return false;
  const status = getErrorStatus(error);
  return status === undefined || (status >= 500 && status !== 503);
}

export function canRetryRequest(error: unknown): boolean {
  const status = getErrorStatus(error);
  return status === undefined || status === 408 || status === 429 || status >= 500;
}
