import { NavLink, Outlet } from "react-router-dom";
import { peopleNavItems } from "@/shared/config/nav";
import { cn } from "@/shared/lib/utils";

export function PeopleLayout() {
  return (
    <>
      <div className="px-6 pt-6">
        <nav
          aria-label="People & Access"
          className="inline-flex max-w-full items-center gap-1 overflow-x-auto rounded-lg bg-muted p-1 text-muted-foreground"
        >
          {peopleNavItems.map(({ title, href, icon: Icon }) => (
            <NavLink
              key={href}
              to={href}
              className={({ isActive }) =>
                cn(
                  "flex shrink-0 items-center gap-2 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isActive
                    ? "bg-background text-foreground shadow-sm"
                    : "hover:bg-background/50 hover:text-foreground",
                )
              }
            >
              <Icon aria-hidden="true" className="hidden h-4 w-4 sm:block" />
              {title}
            </NavLink>
          ))}
        </nav>
      </div>
      <Outlet />
    </>
  );
}
