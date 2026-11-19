import { expect, test } from "./support/fixtures";
import { logsControls } from "./support/logs";

test("mobile navigation opens documents and returns to chat", async ({ adminPage: page }) => {
  // The hamburger has no accessible name. Scope the interaction to the app header.
  const sidebar = page.getByRole("complementary");
  const menu = page.getByRole("banner").getByRole("button", { name: "", exact: true });
  await expect(sidebar).not.toBeInViewport();
  await menu.click();
  const documents = sidebar
    .getByRole("link", { name: "Documents", exact: true })
    .and(page.locator('a[href="/documents"]'));
  await expect(documents).toBeInViewport();
  await documents.click();
  await expect(page.getByRole("heading", { name: "Documents", exact: true })).toBeVisible();
  await expect(sidebar).not.toBeInViewport();
  await page
    .getByRole("navigation", { name: "Breadcrumb" })
    .getByRole("link", { name: "Home" })
    .click();
  await expect(page.getByRole("heading", { name: "Chat", exact: true })).toBeVisible();
});

test("mobile profile and admin logs fit the viewport", async ({ adminPage: page }) => {
  await page.goto("/profile");
  await expect(page.getByRole("heading", { name: "Profile", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Log out" })).toBeVisible();
  await page.goto("/admin/logs");
  const { search } = await logsControls(page);
  await expect(search).toBeVisible();
  const viewport = page.viewportSize();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    viewport?.width ?? 0,
  );
});
