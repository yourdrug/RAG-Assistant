"use client";
import { useCurrentUser } from "@/shared/api/hooks";
import { useAuthStore } from "@/stores/auth-store";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Badge } from "@/shared/ui/badge";
import { Avatar, AvatarFallback } from "@/shared/ui/avatar";
import { ThemeToggle } from "@/features/auth/theme-toggle";
import { User, Shield, Key, Palette } from "lucide-react";
import { Skeleton } from "@/shared/ui/skeleton";

export function ProfilePage() {
  const { data: user, isLoading } = useCurrentUser();
  const token = useAuthStore((s) => s.token);

  return (
    <div className="p-6 space-y-6 max-w-2xl">
      <div><h1 className="text-2xl font-bold">Profile</h1><p className="text-muted-foreground">Manage your account settings</p></div>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><User className="h-5 w-5" />Profile Information</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          {isLoading ? <div className="space-y-3"><Skeleton className="h-12 w-12 rounded-full" /><Skeleton className="h-4 w-48" /></div>
            : user ? (
              <div className="flex items-center gap-4">
                <Avatar className="h-16 w-16"><AvatarFallback className="text-lg">{user.email.charAt(0).toUpperCase()}</AvatarFallback></Avatar>
                <div>
                  <p className="text-lg font-medium">{user.email}</p>
                  <div className="flex gap-2 mt-1">
                    <Badge variant={user.role === "admin" ? "default" : "secondary"}><Shield className="h-3 w-3 mr-1" />{user.role}</Badge>
                    <Badge variant="outline">{user.kind}</Badge>
                    <Badge variant={user.is_active ? "success" : "destructive"}>{user.is_active ? "Active" : "Inactive"}</Badge>
                  </div>
                </div>
              </div>
            ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Key className="h-5 w-5" />API Token</CardTitle><CardDescription>Your authentication token</CardDescription></CardHeader>
        <CardContent>
          <code className="block rounded bg-muted px-3 py-2 text-sm font-mono break-all">{token ? `${token.substring(0, 50)}...` : "Not authenticated"}</code>
          <p className="text-xs text-muted-foreground mt-2">Use: Authorization: Bearer &lt;token&gt;</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Palette className="h-5 w-5" />Appearance</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div><p className="text-sm font-medium">Theme</p><p className="text-xs text-muted-foreground">Toggle light/dark mode</p></div>
            <ThemeToggle />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Key className="h-5 w-5" />Change Password</CardTitle><CardDescription>Requires backend endpoint (POST /auth/change-password)</CardDescription></CardHeader>
        <CardContent><p className="text-sm text-muted-foreground">Password change will be available once the backend provides the endpoint.</p></CardContent>
      </Card>
    </div>
  );
}
