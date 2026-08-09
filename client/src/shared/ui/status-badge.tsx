import { Badge, type BadgeProps } from "@/shared/ui/badge";

const statusVariantMap: Record<string, BadgeProps["variant"]> = {
  done: "success",
  active: "success",
  success: "success",
  failed: "destructive",
  error: "destructive",
  processing: "warning",
  running: "warning",
  pending: "secondary",
  idle: "secondary",
};

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const variant = statusVariantMap[status] || "secondary";
  return (
    <Badge variant={variant} className={className}>
      {status}
    </Badge>
  );
}
