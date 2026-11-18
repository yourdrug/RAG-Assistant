import { useSpaces } from "@/shared/api/hooks/use-spaces";
import type { ChatLogEntry } from "@/shared/api/types";
import { Badge } from "@/shared/ui/badge";

export function ChatLogContext({ entry }: { entry: ChatLogEntry }) {
  const { data: spaces, isPending } = useSpaces();
  const space = spaces?.find((space) => space.id === entry.space_id);

  return (
    <aside className="min-w-0 space-y-4 rounded-xl border bg-card p-5">
      <Badge variant="secondary" className="max-w-full whitespace-normal wrap-anywhere">
        {space?.name ?? (isPending ? "Loading space name…" : "Space name unavailable")}
      </Badge>
      <h2 className="text-sm font-semibold">Question</h2>
      <p className="whitespace-pre-wrap wrap-anywhere">{entry.question}</p>
      <h2 className="border-t pt-4 text-sm font-semibold">Original answer</h2>
      <p className="max-h-80 overflow-auto whitespace-pre-wrap wrap-anywhere text-sm text-muted-foreground">
        {entry.answer || "No answer delivered"}
      </p>
      {entry.feedback?.comment && (
        <>
          <h2 className="border-t pt-4 text-sm font-semibold">User feedback</h2>
          <p className="whitespace-pre-wrap wrap-anywhere text-sm">{entry.feedback.comment}</p>
        </>
      )}
    </aside>
  );
}
