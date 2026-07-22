import {
  MessageSquare, FileText, Search, User, LayoutDashboard, Users, UserCog,
  FolderOpen, Upload, Cpu, Settings, Database, ScanText, Clock, BarChart3,
  ScrollText, Server,
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
  { title: "Clients", href: "/admin/clients", icon: FolderOpen },
  { title: "Documents", href: "/admin/documents", icon: FileText },
  { title: "Ingest", href: "/admin/ingest", icon: Upload },
  { title: "Models", href: "/admin/models", icon: Cpu, disabled: true },
  { title: "RAG Settings", href: "/admin/rag", icon: Settings, disabled: true },
  { title: "Vector DB", href: "/admin/vectordb", icon: Database, disabled: true },
  { title: "OCR", href: "/admin/ocr", icon: ScanText, disabled: true },
  { title: "Jobs", href: "/admin/jobs", icon: Clock, disabled: true },
  { title: "Monitoring", href: "/admin/monitoring", icon: BarChart3, disabled: true },
  { title: "Logs", href: "/admin/logs", icon: ScrollText, disabled: true },
  { title: "Settings", href: "/admin/settings", icon: Server, disabled: true },
];
