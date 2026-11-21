import { lazy, Suspense } from "react";
import { Toaster } from "react-hot-toast";
import { BrowserRouter, Link, Navigate, Route, Routes } from "react-router-dom";
import { ChatPage } from "@/pages/chat";
import { DocumentReaderPage } from "@/pages/document-reader";
import { DocumentsPage } from "@/pages/documents";
import { LoginPage } from "@/pages/login";
import { ProfilePage } from "@/pages/profile";
import { SearchPage } from "@/pages/search";
import { QueryProvider } from "@/providers/query-provider";
import { ThemeProvider } from "@/providers/theme-provider";
import { ErrorBoundary } from "@/shared/ui/error-boundary";
import { TooltipProvider } from "@/shared/ui/tooltip";
import { useAuthStore } from "@/stores/auth-store";
import { AdminLayout } from "@/widgets/layout/admin-layout";
import { DashboardLayout } from "@/widgets/layout/dashboard-layout";
import { PeopleLayout } from "@/widgets/layout/people-layout";

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
const AdminSpacesPage = lazy(() =>
  import("@/pages/admin/spaces").then((m) => ({ default: m.AdminSpacesPage })),
);
const AdminSyncChannelsPage = lazy(() =>
  import("@/pages/admin/sync-channels").then((m) => ({ default: m.AdminSyncChannelsPage })),
);
const AdminCuratorsPage = lazy(() =>
  import("@/pages/admin/curators").then((m) => ({ default: m.AdminCuratorsPage })),
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
const ChatLogKnowledgePage = lazy(() =>
  import("@/pages/admin/chat-log-knowledge").then((m) => ({ default: m.ChatLogKnowledgePage })),
);
const ChatLogBenchmarkPage = lazy(() =>
  import("@/pages/admin/chat-log-benchmark").then((m) => ({ default: m.ChatLogBenchmarkPage })),
);
const AdminBenchmarkPage = lazy(() =>
  import("@/pages/admin/benchmark").then((m) => ({ default: m.AdminBenchmarkPage })),
);
const AdminQualityPage = lazy(() =>
  import("@/pages/admin/quality").then((m) => ({ default: m.AdminQualityPage })),
);
const DryRunResultPage = lazy(() =>
  import("@/pages/admin/dry-run/DryRunResultPage").then((m) => ({ default: m.DryRunResultPage })),
);
const AdminActVersionsPage = lazy(() =>
  import("@/pages/admin/act-versions").then((m) => ({ default: m.AdminActVersionsPage })),
);

function AdminFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
    </div>
  );
}

function NotFound() {
  const token = useAuthStore((state) => state.token);
  if (!token) return <Navigate to="/login" replace />;

  return (
    <main className="mx-auto flex min-h-dvh max-w-md flex-col justify-center gap-3 p-6">
      <h1 className="text-2xl font-bold">Page not found</h1>
      <p className="text-muted-foreground">Check the address or return to the workspace.</p>
      <Link className="w-fit text-primary underline" to="/chat">
        Go to workspace
      </Link>
    </main>
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
                  <Route path="/documents/:documentId" element={<DocumentReaderPage />} />
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
                  <Route element={<PeopleLayout />}>
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
                      path="curators"
                      element={
                        <Suspense fallback={<AdminFallback />}>
                          <AdminCuratorsPage />
                        </Suspense>
                      }
                    />
                  </Route>
                  <Route
                    path="spaces"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminSpacesPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="sync-channels"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminSyncChannelsPage />
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
                    path="act-versions"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminActVersionsPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="quality"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <AdminQualityPage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="quality/dry-run"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <DryRunResultPage />
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
                    path="chat-logs/:logId/knowledge"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <ChatLogKnowledgePage />
                      </Suspense>
                    }
                  />
                  <Route
                    path="chat-logs/:logId/benchmark-question"
                    element={
                      <Suspense fallback={<AdminFallback />}>
                        <ChatLogBenchmarkPage />
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

                <Route path="*" element={<NotFound />} />
              </Routes>
            </ErrorBoundary>
          </BrowserRouter>
          <Toaster position="top-right" />
        </TooltipProvider>
      </QueryProvider>
    </ThemeProvider>
  );
}
