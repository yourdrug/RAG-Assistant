import {
  Activity,
  BarChart3,
  Clock,
  Cpu,
  Database,
  FileText,
  Key,
  LayoutDashboard,
  MessageCircle,
  MessageSquare,
  ScrollText,
  Search,
  Server,
  Upload,
  User,
  UserCog,
  Users,
} from "lucide-react";

export const userNavItems = [
  { title: "Chat", href: "/chat", icon: MessageSquare },
  { title: "Documents", href: "/documents", icon: FileText },
  { title: "Search", href: "/search", icon: Search },
  { title: "Profile", href: "/profile", icon: User },
] as const;

export type AdminNavItem = {
  title: string;
  href: string;
  icon: typeof LayoutDashboard;
  disabled?: boolean;
  adminOnly?: boolean;
};

export const adminNavItems: AdminNavItem[] = [
  { title: "Dashboard", href: "/admin", icon: LayoutDashboard, adminOnly: true },
  { title: "Users", href: "/admin/users", icon: Users, adminOnly: true },
  { title: "Groups", href: "/admin/groups", icon: UserCog, adminOnly: true },
  { title: "Curators", href: "/admin/curators", icon: UserCog, adminOnly: true },
  { title: "API Keys", href: "/admin/api-keys", icon: Key, adminOnly: true },
  { title: "Documents", href: "/admin/documents", icon: FileText },
  { title: "Act Versions", href: "/admin/act-versions", icon: ScrollText, adminOnly: true },
  { title: "Quality", href: "/admin/quality", icon: Activity, adminOnly: true },
  { title: "Ingest", href: "/admin/ingest", icon: Upload, adminOnly: true },
  { title: "Models", href: "/admin/models", icon: Cpu, adminOnly: true },
  { title: "Vector DB", href: "/admin/vectordb", icon: Database, adminOnly: true },
  { title: "Settings", href: "/admin/settings", icon: Server, adminOnly: true },
  { title: "Jobs", href: "/admin/jobs", icon: Clock, adminOnly: true },
  { title: "Chat Logs", href: "/admin/chat-logs", icon: MessageCircle },
  { title: "Benchmark", href: "/admin/benchmark", icon: BarChart3, adminOnly: true },
  { title: "Actions", href: "/admin/logs", icon: ScrollText, adminOnly: true },
];
