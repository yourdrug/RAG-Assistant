"use client";

import { Library, LogOut, Monitor, Moon, Sun, Users } from "lucide-react";
import { useTheme } from "next-themes";
import { useLogout } from "@/features/auth/use-logout";
import { useCurrentUser, useGroups, useSpaces } from "@/shared/api/hooks";
import type { UserResponse } from "@/shared/api/types";
import { Avatar, AvatarFallback } from "@/shared/ui/avatar";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card } from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";

const roleLabels: Record<UserResponse["role"], string> = {
  admin: "Administrator",
  curator: "Curator",
  user: "User",
};

const kindLabels: Record<UserResponse["kind"], string> = {
  internal: "Internal",
  client: "Client",
};

const themes = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const;

function AccountAccess({ user }: { user: UserResponse }) {
  const spaces = useSpaces();
  const groups = useGroups(user.kind === "internal");

  return (
    <Card className="min-w-0 divide-y shadow-sm">
      <section className="p-5 sm:px-6" aria-labelledby="profile-spaces">
        <h2 id="profile-spaces" className="flex items-center gap-2 font-semibold">
          <Library className="size-4 text-muted-foreground" aria-hidden="true" />
          Knowledge spaces
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">Spaces you can ask questions in.</p>
        {spaces.isPending ? (
          <Skeleton className="mt-4 h-9 w-full" />
        ) : spaces.isError ? (
          <p className="mt-4 text-sm text-muted-foreground" role="status">
            Couldn’t load knowledge spaces.
          </p>
        ) : spaces.data.length ? (
          <ul className="mt-4 divide-y rounded-lg border">
            {spaces.data.map((space) => (
              <li key={space.id} className="flex items-center justify-between gap-3 px-3 py-2.5">
                <span className="min-w-0 text-sm font-medium wrap-break-word">{space.name}</span>
                {space.can_upload && (
                  <Badge className="shrink-0" variant="secondary">
                    Can upload
                  </Badge>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-4 text-sm text-muted-foreground">No knowledge spaces available.</p>
        )}
      </section>
      {user.kind === "internal" && (
        <section className="p-5 sm:px-6" aria-labelledby="profile-groups">
          <h2 id="profile-groups" className="flex items-center gap-2 font-semibold">
            <Users className="size-4 text-muted-foreground" aria-hidden="true" />
            {user.role === "user" ? "Your groups" : "Available groups"}
          </h2>
          {groups.isPending ? (
            <Skeleton className="mt-3 h-6 w-28" />
          ) : groups.isError ? (
            <p className="mt-3 text-sm text-muted-foreground" role="status">
              Couldn’t load groups.
            </p>
          ) : groups.data.length ? (
            <ul className="mt-3 flex flex-wrap gap-2">
              {groups.data.map((group) => (
                <li key={group.id} className="max-w-full">
                  <Badge variant="outline" className="max-w-full wrap-break-word">
                    {group.name}
                  </Badge>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">No groups available.</p>
          )}
        </section>
      )}
    </Card>
  );
}

export function ProfilePage() {
  const account = useCurrentUser();
  const logout = useLogout();
  const { theme, setTheme } = useTheme();

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-6 sm:px-8 sm:py-10">
      <h1 className="text-3xl font-semibold tracking-tight">Profile</h1>

      <Card className="overflow-hidden shadow-sm" aria-label="Account">
        {account.isPending ? (
          <div
            className="flex items-center gap-4 p-5 sm:p-6"
            role="status"
            aria-label="Loading account"
          >
            <Skeleton className="size-14 shrink-0 rounded-full" />
            <Skeleton className="h-6 w-44 max-w-full" />
          </div>
        ) : account.isError ? (
          <div className="flex flex-wrap items-center justify-between gap-3 p-5 sm:p-6">
            <p className="text-sm text-muted-foreground" role="status">
              Couldn’t load your account.
            </p>
            <Button
              type="button"
              variant="outline"
              disabled={account.isFetching}
              onClick={() => void account.refetch()}
            >
              Try again
            </Button>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-4 bg-muted/45 p-5 sm:p-6">
              <Avatar className="size-14 shrink-0">
                <AvatarFallback className="bg-primary/10 text-xl font-semibold text-primary">
                  {account.data.email.charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground">Sign-in email</p>
                <h2 className="mt-0.5 text-lg font-semibold wrap-break-word sm:text-xl">
                  {account.data.email}
                </h2>
              </div>
            </div>
            <dl className="grid grid-cols-2 gap-4 border-t p-5 sm:grid-cols-4 sm:px-6">
              <div>
                <dt className="text-sm text-muted-foreground">Role</dt>
                <dd className="mt-1 text-sm font-medium">{roleLabels[account.data.role]}</dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">Account type</dt>
                <dd className="mt-1 text-sm font-medium">{kindLabels[account.data.kind]}</dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">Status</dt>
                <dd className="mt-1">
                  {account.data.is_active ? (
                    <Badge variant="success">Active</Badge>
                  ) : (
                    <Badge variant="destructive">Inactive</Badge>
                  )}
                </dd>
              </div>
              <div>
                <dt className="text-sm text-muted-foreground">Account ID</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">{account.data.id}</dd>
              </div>
            </dl>
          </>
        )}
      </Card>

      <div className="grid items-start gap-5 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        {account.isSuccess && <AccountAccess user={account.data} />}
        <Card className="min-w-0 divide-y shadow-sm">
          <fieldset className="min-w-0 p-5 sm:px-6">
            <legend className="sr-only">Appearance</legend>
            <h2 className="font-semibold">Appearance</h2>
            <p className="mt-1 text-sm text-muted-foreground">Theme for this browser.</p>
            <div className="mt-4 grid grid-cols-3 gap-2">
              {themes.map(({ value, label, icon: Icon }) => (
                <label key={value} className="min-w-0">
                  <input
                    type="radio"
                    name="profile-theme"
                    value={value}
                    className="peer sr-only"
                    checked={theme === value}
                    onChange={() => setTheme(value)}
                  />
                  <span className="flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border px-2 py-2.5 text-xs font-medium text-muted-foreground hover:bg-muted peer-checked:border-primary peer-checked:bg-primary/10 peer-checked:text-primary peer-focus-visible:ring-2 peer-focus-visible:ring-ring">
                    <Icon className="size-4" aria-hidden="true" />
                    {label}
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          <section className="flex flex-wrap items-center justify-between gap-3 p-5 sm:px-6">
            <h2 className="font-semibold">Current session</h2>
            <Button type="button" variant="outline" onClick={logout}>
              <LogOut aria-hidden="true" />
              Log out
            </Button>
          </section>
        </Card>
      </div>
    </div>
  );
}
