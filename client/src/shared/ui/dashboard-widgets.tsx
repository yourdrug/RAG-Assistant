import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { CheckCircle2, XCircle, BarChart3 } from "lucide-react";

export function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function StatusIndicator({ status }: { status: string }) {
  const isOk = status === "ok";
  return (
    <span className={`inline-flex items-center gap-1.5 ${isOk ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
      {isOk ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
      <span className="text-sm font-medium">{isOk ? "Healthy" : "Error"}</span>
    </span>
  );
}

export function ServiceCard({
  title, icon, status, latency, details
}: {
  title: string; icon: React.ReactNode; status: string;
  latency?: number | null; details?: string[];
}) {
  const isOk = status === "ok";
  return (
    <Card className={`transition-colors ${isOk ? "" : "border-red-200 dark:border-red-800"}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${isOk ? "bg-green-500/10" : "bg-red-500/10"}`}>
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            <StatusIndicator status={status} />
          </div>
        </div>
        {latency != null && (
          <div className="text-right">
            <div className="text-lg font-bold font-mono">{latency}</div>
            <div className="text-[10px] text-muted-foreground uppercase">ms</div>
          </div>
        )}
      </CardHeader>
      {details && details.length > 0 && (
        <CardContent className="pt-0">
          <div className="flex flex-wrap gap-1.5">
            {details.map((d, i) => (
              <Badge key={i} variant="secondary" className="text-xs font-mono">{d}</Badge>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

export function StatCard({ label, value, icon: Icon, color }: { label: string; value: string | number; icon: typeof BarChart3; color: string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${color}`}>
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <div className="text-2xl font-bold font-mono">{value}</div>
            <div className="text-xs text-muted-foreground">{label}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
