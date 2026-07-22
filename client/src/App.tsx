import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "@/providers/theme-provider";
import { QueryProvider } from "@/providers/query-provider";
import { TooltipProvider } from "@/shared/ui/tooltip";
import { Toaster } from "react-hot-toast";
import { DashboardLayout } from "@/widgets/layout/dashboard-layout";
import { AdminLayout } from "@/widgets/layout/admin-layout";
import { LoginPage } from "@/pages/login";
import { ChatPage } from "@/pages/chat";
import { DocumentsPage } from "@/pages/documents";
import { SearchPage } from "@/pages/search";
import { ProfilePage } from "@/pages/profile";
import { AdminDashboardPage } from "@/pages/admin/dashboard";
import { AdminUsersPage } from "@/pages/admin/users";
import { AdminGroupsPage } from "@/pages/admin/groups";
import { AdminClientsPage } from "@/pages/admin/clients";
import { AdminDocumentsPage } from "@/pages/admin/documents";
import { AdminIngestPage } from "@/pages/admin/ingest";
import { PlaceholderPage } from "@/pages/admin/placeholder";

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryProvider>
        <TooltipProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<LoginPage />} />

              <Route element={<DashboardLayout />}>
                <Route path="/" element={<Navigate to="/chat" replace />} />
                <Route path="/chat" element={<ChatPage />} />
                <Route path="/documents" element={<DocumentsPage />} />
                <Route path="/search" element={<SearchPage />} />
                <Route path="/profile" element={<ProfilePage />} />
              </Route>

              <Route path="/admin" element={<AdminLayout />}>
                <Route index element={<AdminDashboardPage />} />
                <Route path="users" element={<AdminUsersPage />} />
                <Route path="groups" element={<AdminGroupsPage />} />
                <Route path="clients" element={<AdminClientsPage />} />
                <Route path="documents" element={<AdminDocumentsPage />} />
                <Route path="ingest" element={<AdminIngestPage />} />
                <Route path="models" element={<PlaceholderPage title="Models" description="LLM model configuration" endpoint="GET/POST /admin/models" />} />
                <Route path="rag" element={<PlaceholderPage title="RAG Settings" description="RAG pipeline configuration" endpoint="GET/PUT /admin/rag/settings" />} />
                <Route path="vectordb" element={<PlaceholderPage title="Vector DB" description="Qdrant collections" endpoint="GET /admin/vectordb/collections" />} />
                <Route path="ocr" element={<PlaceholderPage title="OCR" description="OCR settings and history" endpoint="GET /admin/ocr/settings" />} />
                <Route path="jobs" element={<PlaceholderPage title="Jobs" description="Background tasks" endpoint="GET /admin/jobs" />} />
                <Route path="monitoring" element={<PlaceholderPage title="Monitoring" description="System metrics" endpoint="GET /admin/metrics" />} />
                <Route path="logs" element={<PlaceholderPage title="Logs" description="System logs" endpoint="GET /admin/logs" />} />
                <Route path="settings" element={<PlaceholderPage title="Settings" description="System configuration" endpoint="GET /admin/settings" />} />
              </Route>
            </Routes>
          </BrowserRouter>
          <Toaster position="top-right" />
        </TooltipProvider>
      </QueryProvider>
    </ThemeProvider>
  );
}
