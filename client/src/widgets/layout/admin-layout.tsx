"use client";
import { useEffect, useState } from "react";
import { Outlet, useNavigate, useLocation, Link } from "react-router-dom";
import { useAuthStore } from "@/stores/auth-store";
import { useCurrentUser } from "@/shared/api/hooks";
import { adminNavItems } from "@/shared/config/nav";
import { Button } from "@/shared/ui/button";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { ArrowLeft, Shield, Menu, X } from "lucide-react";
import { cn } from "@/shared/lib/utils";

export function AdminLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { token, isAuthenticated } = useAuthStore();
  const { data: user } = useCurrentUser();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    if (!isAuthenticated && !token) navigate("/login");
  }, [isAuthenticated, token, navigate]);

  useEffect(() => {
    if (user && user.role !== "admin") navigate("/chat");
  }, [user, navigate]);

  return (
    <div className="flex h-screen bg-background">
      {sidebarOpen && <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} />}

      <aside className={cn("fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r bg-sidebar transition-transform lg:static lg:translate-x-0", sidebarOpen ? "translate-x-0" : "-translate-x-full")}>
        <div className="flex h-14 items-center border-b px-4">
          <Button variant="ghost" size="icon" className="mr-2" onClick={() => navigate("/chat")}><ArrowLeft className="h-4 w-4" /></Button>
          <Shield className="h-4 w-4 mr-2 text-primary" />
          <span className="font-semibold text-sidebar-foreground">Admin Panel</span>
          <Button variant="ghost" size="icon" className="ml-auto lg:hidden" onClick={() => setSidebarOpen(false)}><X className="h-4 w-4" /></Button>
        </div>

        <ScrollArea className="flex-1 px-3 py-4">
          <nav className="space-y-1">
            {adminNavItems.map((item) => {
              const Icon = item.icon;
              const active = location.pathname === item.href || (item.href !== "/admin" && location.pathname.startsWith(item.href));
              return (
                <Link key={item.href} to={item.disabled ? "#" : item.href}
                  onClick={(e) => { if (item.disabled) e.preventDefault(); else setSidebarOpen(false); }}
                  className={cn("flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    item.disabled ? "cursor-not-allowed opacity-50" : "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/70")}>
                  <Icon className="h-4 w-4" />{item.title}
                  {item.disabled && <span className="ml-auto text-[10px] text-muted-foreground">Soon</span>}
                </Link>
              );
            })}
          </nav>
        </ScrollArea>
      </aside>

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center border-b px-4 lg:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setSidebarOpen(true)}><Menu className="h-5 w-5" /></Button>
          <div className="ml-4 text-sm text-muted-foreground">Admin / {location.pathname.split("/").pop() || "Dashboard"}</div>
        </header>
        <main className="flex-1 overflow-auto"><Outlet /></main>
      </div>
    </div>
  );
}
