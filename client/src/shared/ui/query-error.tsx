import { canRetryRequest, getErrorMessage } from "@/shared/lib/errors";
import { RequestError } from "@/shared/ui/request-error";

export function QueryError({
  query,
  fallback,
  retryLabel,
}: {
  query: { error: unknown; isFetching: boolean; refetch: () => Promise<unknown> };
  fallback: string;
  retryLabel?: string;
}) {
  return (
    <RequestError
      message={getErrorMessage(query.error, fallback)}
      onRetry={canRetryRequest(query.error) ? () => void query.refetch() : undefined}
      retrying={query.isFetching}
      retryLabel={retryLabel}
    />
  );
}
