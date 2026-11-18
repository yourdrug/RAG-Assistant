import { ChevronRight, House } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useDocument } from "@/shared/api/hooks/use-documents";
import { adminNavItems, peopleNavItems, userNavItems } from "@/shared/config/nav";

type Crumb = { label: string; href?: string };

export function PageBreadcrumbs({ canOpenAdmin = false }: { canOpenAdmin?: boolean }) {
  const { pathname } = useLocation();
  const documentMatch = /^\/documents\/(\d+)\/?$/.exec(pathname);
  const documentId = Number(documentMatch?.[1]);
  const { data: document } = useDocument(
    Number.isSafeInteger(documentId) && documentId > 0 ? documentId : 0,
  );
  const crumbs: Crumb[] = [{ label: "Home", href: "/chat" }];

  if (pathname === "/admin" || pathname.startsWith("/admin/")) {
    crumbs.push({ label: "Admin", href: canOpenAdmin ? "/admin" : undefined });
    const peoplePage = peopleNavItems.find((item) => item.href === pathname);
    if (peoplePage) {
      crumbs.push({ label: "People & Access" }, { label: peoplePage.title });
    } else if (/^\/admin\/chat-logs\/\d+\/(knowledge|benchmark-question)\/?$/.test(pathname)) {
      crumbs.push(
        { label: "Answer logs", href: "/admin/chat-logs" },
        {
          label: pathname.replace(/\/$/, "").endsWith("/knowledge")
            ? "Improve knowledge"
            : "Create a benchmark question",
        },
      );
    } else if (pathname === "/admin/quality/dry-run") {
      crumbs.push({ label: "Quality", href: "/admin/quality" }, { label: "Dry Run" });
    } else {
      const page = adminNavItems.find((item) => item.href === pathname);
      if (page) crumbs.push({ label: page.title });
    }
  } else if (documentMatch) {
    crumbs.push(
      { label: "Documents", href: "/documents" },
      { label: document?.filename || "Document" },
    );
  } else {
    const page = userNavItems.find((item) => item.href === pathname);
    if (page) crumbs.push({ label: page.title });
  }

  return (
    <nav aria-label="Breadcrumb" className="min-w-0 flex-1">
      <ol className="flex min-w-0 items-center gap-1 text-sm sm:gap-2">
        {crumbs.map((crumb, index) => {
          const current = index === crumbs.length - 1;
          return (
            <li
              key={`${index}-${crumb.label}`}
              className={`flex min-w-0 items-center gap-1 sm:gap-2 ${current ? "flex-1" : "shrink-0"}`}
            >
              {index > 0 && (
                <ChevronRight
                  aria-hidden="true"
                  className="h-3.5 w-3.5 shrink-0 text-muted-foreground/50"
                />
              )}
              {crumb.href && !current ? (
                <Link
                  to={crumb.href}
                  aria-label={index === 0 ? crumb.label : undefined}
                  title={crumb.label}
                  className="inline-flex min-h-8 items-center rounded-md px-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {index === 0 ? <House aria-hidden="true" className="h-4 w-4" /> : crumb.label}
                </Link>
              ) : (
                <span
                  aria-current={current ? "page" : undefined}
                  title={crumb.label}
                  className={`truncate py-1.5 ${current ? "font-medium text-foreground" : "max-w-24 text-muted-foreground sm:max-w-none"}`}
                >
                  {crumb.label}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
