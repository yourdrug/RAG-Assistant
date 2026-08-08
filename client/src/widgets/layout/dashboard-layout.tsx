"use client";
import { useEffect, useRef, useState } from "react";
import { Outlet, useNavigate, useLocation, Link, useSearchParams } from "react-router-dom";
import { useAuthStore } from "@/stores/auth-store";
import { useCurrentUser, useConversations } from "@/shared/api/hooks";
import { userNavItems, adminNavItems } from "@/shared/config/nav";
import { ThemeToggle } from "@/features/auth/theme-toggle";
import { Button } from "@/shared/ui/button";
import { Avatar, AvatarFallback } from "@/shared/ui/avatar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/shared/ui/dropdown-menu";
import { ScrollArea } from "@/shared/ui/scroll-area";
import { Separator } from "@/shared/ui/separator";
import { LogOut, User, Shield, Menu, X, Plus, MessageSquare, History } from "lucide-react";
import { cn } from "@/shared/lib/utils";

export function DashboardLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { token, user, isAuthenticated, logout } = useAuthStore();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [convPanelOpen, setConvPanelOpen] = useState(false);
  const convPanelRef = useRef<HTMLDivElement>(null);
  const { data: currentUser } = useCurrentUser();
  const { data: convData } = useConversations();

  useEffect(() => {
    if (!isAuthenticated && !token) navigate("/login");
  }, [isAuthenticated, token, navigate]);

  useEffect(() => {
    if (currentUser && currentUser !== user) {
      useAuthStore.getState().setUser(currentUser);
    }
  }, [currentUser]);

  // Close conversation panel when clicking outside
  useEffect(() => {
    if (!convPanelOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (convPanelRef.current && !convPanelRef.current.contains(e.target as Node)) {
        const target = e.target as HTMLElement;
        if (!target.closest('[data-chat-trigger]')) {
          setConvPanelOpen(false);
        }
      }
    };
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, [convPanelOpen]);

  const handleLogout = () => { logout(); navigate("/login"); };
  const displayUser = currentUser || user;
  const isAdmin = displayUser?.role === "admin";
  const activeConversationId = searchParams.get("id");

  const handleNewChat = () => {
    navigate("/chat");
    setConvPanelOpen(false);
    setSidebarOpen(false);
  };

  const handleSelectConversation = (id: number) => {
    navigate(`/chat?id=${id}`);
    setConvPanelOpen(false);
    setSidebarOpen(false);
  };

  const handleChatClick = () => {
    if (location.pathname === "/chat") {
      setConvPanelOpen((p) => !p);
    } else {
      navigate("/chat");
    }
    setSidebarOpen(false);
  };

  const formatConvTitle = (title: string | null | undefined) => {
    if (!title) return "New chat";
    return title;
  };

  const formatConvDate = (dateStr: string | null | undefined) => {
    if (!dateStr) return "";
    const d = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
  };

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
              const active = item.href === "/chat"
                ? location.pathname === "/chat"
                : location.pathname === item.href || location.pathname.startsWith(item.href + "/");
              return (
                <Link key={item.href} to={item.href}
                  data-chat-trigger={item.href === "/chat" ? "" : undefined}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (item.href === "/chat") {
                      e.preventDefault();
                      handleChatClick();
                    } else {
                      setSidebarOpen(false);
                    }
                  }}
                  className={cn("flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/70")}>
                  <Icon className="h-4 w-4" />{item.title}
                  {item.href === "/chat" && (
                    <History className="ml-auto h-3.5 w-3.5 text-sidebar-foreground/40" />
                  )}
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

      {/* Conversation history panel — part of flex flow, shifts main content */}
      {convPanelOpen && (
        <div ref={convPanelRef} className="hidden lg:flex flex-col w-72 shrink-0 border-r bg-sidebar">
          <div className="flex h-14 items-center justify-between border-b px-4">
            <div className="flex items-center gap-2">
              <History className="h-4 w-4" />
              <span className="font-semibold text-sm">Dialogues</span>
            </div>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleNewChat} title="New chat">
                <Plus className="h-3.5 w-3.5" />
              </Button>
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setConvPanelOpen(false)}>
                <X className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
          <ScrollArea className="flex-1">
            <nav className="space-y-0.5 p-2">
              {convData?.conversations?.map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => handleSelectConversation(conv.id)}
                  className={cn(
                    "flex w-full items-start gap-2 rounded-md px-3 py-2 text-left text-sm transition-all duration-150 cursor-pointer",
                    activeConversationId === String(conv.id)
                      ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                      : "text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground hover:shadow-sm"
                  )}
                >
                  <MessageSquare className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="line-clamp-2 break-all">{formatConvTitle(conv.title)}</div>
                    <div className="text-xs text-sidebar-foreground/40">{formatConvDate(conv.created_at)}</div>
                  </div>
                </button>
              ))}
              {convData?.conversations?.length === 0 && (
                <p className="px-3 py-4 text-xs text-center text-sidebar-foreground/40">No dialogues yet</p>
              )}
            </nav>
          </ScrollArea>
        </div>
      )}

      <div className="flex flex-1 flex-col overflow-hidden min-w-0">
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
