import { admin, expect, login, test } from "./support/fixtures";

test("guest redirects from a protected page, root and unknown URLs", async ({ page }) => {
  for (const route of ["/profile", "/admin/users", "/", "/not-a-page"]) {
    await page.goto(route);
    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByLabel("Login")).toBeVisible();
  }
});

test("empty login validates both inputs locally", async ({ page }) => {
  await page.goto("/login");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText("Login required")).toBeVisible();
  await expect(page.getByText("Password required")).toBeVisible();
});

test("invalid credentials are rejected by the real API", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Login").fill(admin.email);
  await page.getByLabel("Password").fill("wrong-local-password");
  const result = page.waitForResponse(
    (r) => r.url().endsWith("/api/auth/login") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Sign in" }).click();
  const response = await result;
  expect(response.status()).toBe(422);
  expect(await response.json()).toMatchObject({ message: "Invalid email or password" });
  await expect(page.getByRole("status")).toHaveText("Login failed");
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("button", { name: "Sign in" })).toBeEnabled();
});

test("profile shows the account, persists theme and logs out", async ({ adminPage: page }) => {
  await page.getByRole("link", { name: "Profile", exact: true }).click();
  await expect(page.getByRole("heading", { name: admin.email })).toBeVisible();
  await expect(page.getByText("Administrator", { exact: true })).toBeVisible();
  await page.getByText("Dark", { exact: true }).click();
  await expect(page.getByRole("radio", { name: "Dark" })).toBeChecked();
  await page.reload();
  await expect(page.getByRole("radio", { name: "Dark" })).toBeChecked();
  await expect(page.getByRole("heading", { name: admin.email })).toBeVisible();
  await page.getByRole("button", { name: "Log out" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await page.goto("/profile");
  await expect(page).toHaveURL(/\/login$/);
});

test("regular user cannot open admin UI or list accounts", async ({ page, makeUser }) => {
  const user = await makeUser();
  await login(page, user);
  await expect(page.getByRole("link", { name: "People & Access" })).toHaveCount(0);
  await page.goto("/admin/users");
  await expect(page).toHaveURL(/\/chat$/);
  await expect(page.getByRole("heading", { name: "Chat", exact: true })).toBeVisible();
  const status = await page.evaluate(async () => {
    const { state } = JSON.parse(localStorage.getItem("rag-auth") ?? "{}");
    return (
      await fetch("/api/auth/users", {
        headers: { Authorization: `Bearer ${state.token}` },
      })
    ).status;
  });
  expect(status).toBe(403);
});
