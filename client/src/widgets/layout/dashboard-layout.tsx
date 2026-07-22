"use client";
import { useEffect, useState } from "react";
import { Outlet, useNavigate, useLocation, Link } from "react-router-dom";
import { useAuthStore } from "@/stores/auth-store";
import { useCurrentUser } from "@/shared/api/hooks";
import { userNavItems, adminNavItems } from "@/shared/config/nav";
import { ThemeToggle } from "@/features/auth/theme-toggle";
import { Button } from "@/shared/ui/button";
import { Avatar, AvatarFallback } from "@/shared/ui/avatar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/shared/ui/dropdown-menu";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Separator } from "@/shared/ui/separator";
import { LogOut, User, Shield, Menu, X } from "lucide-react";
import { cn } from "@/shared/lib/utils";

export function DashboardLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { token, user, isAuthenticated, logout } = useAuthStore();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { data: currentUser } = useCurrentUser();

  useEffect(() => {
    if (!isAuthenticated && !token) navigate("/login");
  }, [isAuthenticated, token, navigate]);

  const handleLogout = () => { logout(); navigate("/login"); };
  const displayUser = currentUser || user;
  const isAdmin = displayUser?.role === "admin";

  return (
    <div className="flex h-screen bg-background">
      {sidebarOpen && <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} />}

      <aside className={cn("fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r bg-sidebar transition-transform lg:static lg:translate-x-0", sidebarOpen ? "translate-x-0" : "-translate-x-full")}>
        <div className="flex h-14 items-center border-b px-4">
          <Link to="/chat" className="flex items-center gap-2" onClick={() => setSidebarOpen(false)}>
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground text-sm font-bold">R</div>
            <span className="font-semibold text-sidebar-foreground">RAG Assistant</span>
          </Link>
          <Button variant="ghost" size="icon" className="ml-auto lg:hidden" onClick={() => setSidebarOpen(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="flex-1 px-3 py-4">
          <nav className="space-y-1">
            {userNavItems.map((item) => {
              const Icon = item.icon;
              const active = location.pathname === item.href || location.pathname.startsWith(item.href + "/");
              return (
                <Link key={item.href} to={item.href} onClick={() => setSidebarOpen(false)}
                  className={cn("flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/70")}>
                  <Icon className="h-4 w-4" />{item.title}
                </Link>
              );
            })}
          </nav>

          {isAdmin && (
            <>
              <Separator className="my-4" />
              <div className="mb-2 px-3 text-xs font-semibold uppercase text-sidebar-foreground/50">Admin</div>
              <nav className="space-y-1">
                {adminNavItems.map((item) => {
                  const Icon = item.icon;
                  const active = location.pathname === item.href || (item.href !== "/admin" && location.pathname.startsWith(item.href));
                  return (
                    <Link key={item.href} to={item.disabled ? "#" : item.href}
                      onClick={(e) => { if (item.disabled) { e.preventDefault(); } else { setSidebarOpen(false); } }}
                      className={cn("flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                        item.disabled ? "cursor-not-allowed opacity-50" : "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                        active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/70")}>
                      <Icon className="h-4 w-4" />{item.title}
                    </Link>
                  );
                })}
              </nav>
            </>
          )}
        </ScrollArea>
      </aside>

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center border-b px-4 lg:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setSidebarOpen(true)}>
            <Menu className="h-5 w-5" />
          </Button>
          <div className="ml-auto flex items-center gap-2">
            <ThemeToggle />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                  <Avatar className="h-8 w-8"><AvatarFallback>{displayUser?.email?.charAt(0).toUpperCase() || "U"}</AvatarFallback></Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end">
                <div className="flex items-center gap-2 p-2">
                  <div className="flex flex-col space-y-1 leading-none">
                    <p className="font-medium">{displayUser?.email}</p>
                    <p className="text-xs text-muted-foreground">{displayUser?.role} · {displayUser?.kind}</p>
                  </div>
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate("/profile")}><User className="mr-2 h-4 w-4" />Profile</DropdownMenuItem>
                {isAdmin && <DropdownMenuItem onClick={() => navigate("/admin")}><Shield className="mr-2 h-4 w-4" />Admin Panel</DropdownMenuItem>}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout}><LogOut className="mr-2 h-4 w-4" />Log out</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>
        <main className="flex-1 overflow-auto"><Outlet /></main>
      </div>
    </div>
  );
}
