import { randomUUID } from "node:crypto";
import { type APIRequestContext, test as base, expect, type Page } from "@playwright/test";

import { admin, apiURL } from "./environment";

export { admin };

type Credentials = typeof admin;
type Account = Credentials & { id: number };
type Space = { id: number; name: string };
type TestFixtures = {
  adminPage: Page;
  makeUser: (options?: {
    kind?: "internal" | "client";
    role?: "user" | "curator";
  }) => Promise<Account>;
  space: Space;
  makeSpace: () => Promise<Space>;
  manualDocument: { id: number; filename: string };
};
type WorkerFixtures = { adminApi: APIRequestContext };

export function unique(label: string) {
  return `pw-${label}-${randomUUID().slice(0, 12)}`;
}

export async function login(page: Page, credentials = admin) {
  await page.goto("/login");
  await page.getByLabel("Login").fill(credentials.email);
  await page.getByLabel("Password").fill(credentials.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Chat", exact: true })).toBeVisible();
}

export async function choose(page: Page, option: string | RegExp) {
  await page.getByRole("option", { name: option, exact: typeof option === "string" }).click();
}

export async function deleteDocument(api: APIRequestContext, id: number) {
  const endpoint = `/documents/${id}`;
  const current = await api.get(endpoint);
  if (current.status() === 404) return;
  await expect(current).toBeOK();
  await expect(await api.delete(endpoint)).toBeOK();
  await expect
    .poll(async () => (await api.get(endpoint)).status(), {
      timeout: 20_000,
    })
    .toBe(404);
}

export const test = base.extend<TestFixtures, WorkerFixtures>({
  adminApi: [
    async ({ playwright }, use) => {
      const api = await playwright.request.newContext({
        baseURL: apiURL,
        extraHTTPHeaders: { Authorization: `Bearer ${process.env.PLAYWRIGHT_ADMIN_TOKEN}` },
        maxRedirects: 0,
      });
      try {
        await use(api);
      } finally {
        await api.dispose();
      }
    },
    { scope: "worker", auto: true },
  ],

  adminPage: async ({ page }, use) => {
    await page.addInitScript(
      ({ origin, token }) => {
        if (location.origin !== origin) return;
        const key = "rag-auth";
        // Seed once for this new context. Log out must remain effective on reload.
        if (!localStorage.getItem(key)) {
          localStorage.setItem(key, JSON.stringify({ state: { token }, version: 0 }));
        }
      },
      { origin: "http://127.0.0.1:13001", token: process.env.PLAYWRIGHT_ADMIN_TOKEN },
    );
    await page.goto("/chat");
    await expect(page.getByRole("heading", { name: "Chat", exact: true })).toBeVisible();
    await use(page);
  },

  makeUser: async ({ adminApi }, use) => {
    const ids: number[] = [];
    try {
      await use(async (options = {}) => {
        const credentials = {
          email: `${unique("user")}@playwright.example.com`,
          password: "Pw-local-user-51!",
        };
        const response = await adminApi.post("/auth/users", {
          data: { ...credentials, ...options },
        });
        await expect(response).toBeOK();
        const { id } = await response.json();
        ids.push(id);
        return { ...credentials, id };
      });
    } finally {
      for (const id of ids) {
        await expect(await adminApi.patch(`/auth/users/${id}?is_active=false`)).toBeOK();
      }
    }
  },

  makeSpace: async ({ adminApi }, use) => {
    const spaces: Space[] = [];
    try {
      await use(async () => {
        const packs = await adminApi.get("/spaces/packs");
        await expect(packs).toBeOK();
        const { pack_keys } = await packs.json();
        expect(pack_keys.length).toBeGreaterThan(0);
        const response = await adminApi.post("/spaces", {
          data: { name: unique("space"), pack_key: pack_keys[0] },
        });
        await expect(response).toBeOK();
        const space: Space = await response.json();
        spaces.push(space);
        return space;
      });
    } finally {
      for (const space of spaces) {
        await expect(await adminApi.delete(`/spaces/${space.id}`)).toBeOK();
      }
    }
  },

  space: async ({ makeSpace }, use) => use(await makeSpace()),

  manualDocument: async ({ adminApi, space }, use) => {
    const response = await adminApi.post("/documents/manual", {
      data: { title: unique("reader"), visibility: "internal_private", space_id: space.id },
    });
    await expect(response).toBeOK();
    const document = await response.json();
    try {
      await use(document);
    } finally {
      await deleteDocument(adminApi, document.id);
    }
  },
});

export { expect };
