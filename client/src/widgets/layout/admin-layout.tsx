"use client";
import { Menu, Shield, X } from "lucide-react";
import { useState } from "react";
import { Link, Navigate, Outlet, useLocation } from "react-router-dom";
import { useCurrentUser } from "@/shared/api/hooks";
import { adminNavItems, userNavItems } from "@/shared/config/nav";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/button";
import { ErrorBoundary } from "@/shared/ui/error-boundary";
import { QueryError } from "@/shared/ui/query-error";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Separator } from "@/shared/ui/separator";
import { useAuthStore } from "@/stores/auth-store";
import { PageBreadcrumbs } from "./page-breadcrumbs";

export function AdminLayout() {
  const location = useLocation();
  const token = useAuthStore((s) => s.token);
  const account = useCurrentUser();
  const { data: user } = account;
  const [sidebarOpen, setSidebarOpen] = useState(false);

  if (!token) return <Navigate to="/login" replace />;
  if (user && user.role !== "admin" && user.role !== "curator") {
    return <Navigate to="/chat" replace />;
  }

  const isAdmin = user?.role === "admin";
  const visibleNavItems = adminNavItems.filter((item) => !item.adminOnly || isAdmin);

  return (
    <div className="flex h-screen bg-background">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r bg-sidebar transition-transform lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-14 items-center border-b px-4">
          <Shield className="h-4 w-4 mr-2 text-primary" />
          <span className="font-semibold text-sidebar-foreground">Admin Panel</span>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="flex-1 px-3 py-4">
          <nav className="space-y-1">
            {userNavItems.map((item) => {
              const Icon = item.icon;
              const active =
                location.pathname === item.href || location.pathname.startsWith(item.href + "/");
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.title}
                </Link>
              );
            })}
          </nav>

          <Separator className="my-4" />
          <div className="mb-2 px-3 text-xs font-semibold uppercase text-sidebar-foreground/75">
            Admin
          </div>
          <nav className="space-y-1">
            {visibleNavItems.map((item) => {
              const Icon = item.icon;
              const active = (item.activePaths ?? [item.href]).some(
                (path) =>
                  location.pathname === path ||
                  (path !== "/admin" && location.pathname.startsWith(path + "/")),
              );
              return (
                <Link
                  key={item.href}
                  to={item.disabled ? "#" : item.href}
                  onClick={(e) => {
                    if (item.disabled) e.preventDefault();
                    else setSidebarOpen(false);
                  }}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    item.disabled
                      ? "cursor-not-allowed opacity-50"
                      : "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.title}
                  {item.disabled && (
                    <span className="ml-auto text-[10px] text-muted-foreground">Soon</span>
                  )}
                </Link>
              );
            })}
          </nav>
        </ScrollArea>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4 lg:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            aria-label="Open navigation"
            aria-expanded={sidebarOpen}
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </Button>
          <PageBreadcrumbs canOpenAdmin={isAdmin} />
        </header>
        <main className="flex-1 overflow-auto">
          {user ? (
            <ErrorBoundary key={location.pathname}>
              <Outlet />
            </ErrorBoundary>
          ) : account.isError ? (
            <div className="p-6">
              <QueryError query={account} fallback="Could not verify your account." />
            </div>
          ) : (
            <p role="status" className="p-6 text-muted-foreground">
              Checking your account…
            </p>
          )}
        </main>
      </div>
    </div>
  );
}
