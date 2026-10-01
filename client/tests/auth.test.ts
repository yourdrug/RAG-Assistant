import assert from "node:assert/strict";
import { test } from "node:test";

const saved = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {value: {
  getItem: (key: string) => saved.get(key) ?? null,
  setItem: (key: string, value: string) => saved.set(key, value),
  removeItem: (key: string) => saved.delete(key),
}, configurable: true});
Object.defineProperty(globalThis, "window", {value: {localStorage: globalThis.localStorage}, configurable: true});
const { useAuthStore } = await import("../src/stores/auth-store.ts");

test("authentication persists only the token and logout clears session state", () => {
  const user = { id: 1, email: "user@example.org", role: "user", kind: "internal", is_active: true } as const;
  useAuthStore.getState().setAuth("secret-token", user);
  assert.equal(useAuthStore.getState().isAuthenticated, true);
  assert.deepEqual(JSON.parse(saved.get("rag-auth")!).state, {token: "secret-token"});
  useAuthStore.getState().logout();
  assert.equal(useAuthStore.getState().token, null);
  assert.equal(useAuthStore.getState().user, null);
  assert.equal(useAuthStore.getState().isAuthenticated, false);
  assert.deepEqual(JSON.parse(saved.get("rag-auth")!).state, {token: null});
});
