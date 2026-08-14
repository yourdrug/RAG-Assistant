import {
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
};

export const adminNavItems: AdminNavItem[] = [
  { title: "Dashboard", href: "/admin", icon: LayoutDashboard },
  { title: "Users", href: "/admin/users", icon: Users },
  { title: "Groups", href: "/admin/groups", icon: UserCog },
  { title: "API Keys", href: "/admin/api-keys", icon: Key },
  { title: "Documents", href: "/admin/documents", icon: FileText },
  { title: "Ingest", href: "/admin/ingest", icon: Upload },
  { title: "Models", href: "/admin/models", icon: Cpu },
  { title: "Vector DB", href: "/admin/vectordb", icon: Database },
  { title: "Settings", href: "/admin/settings", icon: Server },
  { title: "Jobs", href: "/admin/jobs", icon: Clock },
  { title: "Monitoring", href: "/admin/monitoring", icon: BarChart3 },
  { title: "Chat Logs", href: "/admin/chat-logs", icon: MessageCircle },
  { title: "Actions", href: "/admin/logs", icon: ScrollText },
];
