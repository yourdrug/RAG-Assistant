import { Button } from "@/shared/ui/button";

export function RequestError({
  message,
  onRetry,
  retrying,
  retryLabel = "Try again",
}: {
  message: string;
  onRetry?: () => void;
  retrying?: boolean;
  retryLabel?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm">
      <p className="min-w-0 break-words text-destructive" role="alert">
        {message}
      </p>
      {onRetry && (
        <Button type="button" variant="outline" size="sm" onClick={onRetry} disabled={retrying}>
          {retryLabel}
        </Button>
      )}
    </div>
  );
}
