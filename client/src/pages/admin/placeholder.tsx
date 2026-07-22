"use client";
import { Card, CardContent } from "@/shared/ui/card";
import { Construction } from "lucide-react";

interface Props { title: string; description: string; endpoint?: string; }

export function PlaceholderPage({ title, description, endpoint }: Props) {
  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">{title}</h1><p className="text-muted-foreground">{description}</p></div>
      <Card>
        <CardContent className="pt-6">
          <div className="text-center py-12">
            <Construction className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
            <p className="text-lg font-medium text-muted-foreground">Coming Soon</p>
            <p className="text-sm text-muted-foreground/70 mt-2 max-w-md mx-auto">This feature requires backend endpoints that are not yet implemented.</p>
            {endpoint && <code className="inline-block mt-4 rounded bg-muted px-3 py-1 text-xs font-mono">{endpoint}</code>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
