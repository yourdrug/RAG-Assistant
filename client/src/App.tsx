import { lazy, Suspense } from "react";
import { Toaster } from "react-hot-toast";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ChatPage } from "@/pages/chat";
import { DocumentsPage } from "@/pages/documents";
import { LoginPage } from "@/pages/login";
import { ProfilePage } from "@/pages/profile";
import { SearchPage } from "@/pages/search";
import { QueryProvider } from "@/providers/query-provider";
import { ThemeProvider } from "@/providers/theme-provider";
import { ErrorBoundary } from "@/shared/ui/error-boundary";
import { TooltipProvider } from "@/shared/ui/tooltip";
import { AdminLayout } from "@/widgets/layout/admin-layout";
import { DashboardLayout } from "@/widgets/layout/dashboard-layout";

// Lazy-loaded admin pages — separate chunk, only loaded when navigating to /admin
const AdminDashboardPage = lazy(() =>
  import("@/pages/admin/dashboard").then((m) => ({ default: m.AdminDashboardPage })),
);
const AdminUsersPage = lazy(() =>
  import("@/pages/admin/users").then((m) => ({ default: m.AdminUsersPage })),
);
const AdminGroupsPage = lazy(() =>
  import("@/pages/admin/groups").then((m) => ({ default: m.AdminGroupsPage })),
);
const AdminApiKeysPage = lazy(() =>
  import("@/pages/admin/api-keys").then((m) => ({ default: m.AdminApiKeysPage })),
);
const AdminDocumentsPage = lazy(() =>
  import("@/pages/admin/documents").then((m) => ({ default: m.AdminDocumentsPage })),
);
const AdminIngestPage = lazy(() =>
  import("@/pages/admin/ingest").then((m) => ({ default: m.AdminIngestPage })),
);
const AdminModelsPage = lazy(() =>
  import("@/pages/admin/models").then((m) => ({ default: m.AdminModelsPage })),
);
const AdminVectorDBPage = lazy(() =>
  import("@/pages/admin/vectordb").then((m) => ({ default: m.AdminVectorDBPage })),
);
const AdminSettingsPage = lazy(() =>
  import("@/pages/admin/settings").then((m) => ({ default: m.AdminSettingsPage })),
);
const JobsPage = lazy(() => import("@/pages/admin/jobs").then((m) => ({ default: m.JobsPage })));
const LogsPage = lazy(() => import("@/pages/admin/logs").then((m) => ({ default: m.LogsPage })));
const ChatLogsPage = lazy(() =>
  import("@/pages/admin/chat-logs").then((m) => ({ default: m.AdminChatLogsPage })),
);
const AdminBenchmarkPage = lazy(() =>
  import("@/pages/admin/benchmark").then((m) => ({ default: m.AdminBenchmarkPage })),
);

function AdminFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryProvider>
        <TooltipProvider>
          <BrowserRouter>
            <ErrorBoundary>
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
                  <Route
                    index
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminDashboardPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="users"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminUsersPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="groups"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminGroupsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="api-keys"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminApiKeysPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="documents"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminDocumentsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="ingest"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminIngestPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="models"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminModelsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="vectordb"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminVectorDBPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="jobs"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <JobsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="logs"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <LogsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="chat-logs"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <ChatLogsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="benchmark"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminBenchmarkPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="settings"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminSettingsPage />
                      </Suspense>
                    }
                  />
                </Route>

                <Route path="*" element={<Navigate to="/chat" replace />} />
              </Routes>
            </ErrorBoundary>
          </BrowserRouter>
          <Toaster position="top-right" />
        </TooltipProvider>
      </QueryProvider>
    </ThemeProvider>
  );
}
