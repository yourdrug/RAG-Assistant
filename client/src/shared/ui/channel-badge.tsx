import { GitBranch } from "lucide-react";
import { Badge } from "@/shared/ui/badge";

/** A document kept in line with a repository: it is changed in its storage, not here. */
export function ChannelBadge({ channelId, path }: { channelId: number; path?: string | null }) {
  return (
    <Badge
      variant="outline"
      className="w-fit gap-1 text-xs"
      title={path ? `Managed by sync channel #${channelId}: ${path}` : undefined}
    >
      <GitBranch className="h-3 w-3" />
      Managed by channel #{channelId}
    </Badge>
  );
}
