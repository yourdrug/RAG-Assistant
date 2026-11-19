import { expect, test, unique } from "./support/fixtures";
import { logsControls } from "./support/logs";

test("admin dashboard renders the real health summary", async ({ adminPage: page }) => {
  const response = page.waitForResponse((r) => r.url().endsWith("/api/health"));
  const operationsResponse = page.waitForResponse((r) => r.url().endsWith("/api/admin/health"));
  await page.goto("/admin");
  const status = await response;
  const operationsStatus = await operationsResponse;
  expect(status.ok()).toBeTruthy();
  expect(operationsStatus.ok()).toBeTruthy();
  const health = await status.json();
  const operations = await operationsStatus.json();
  const healthy =
    health.status === "healthy" &&
    operations.storage.status === "ok" &&
    !operations.indexing?.degraded;
  await expect(page.getByRole("heading", { name: "Admin Dashboard" })).toBeVisible();
  await expect(page.getByText(`v${health.version}`, { exact: true })).toBeVisible();
  for (const service of ["Redis", "S3 Storage", "Indexing workers"]) {
    await expect(page.getByText(service, { exact: true })).toBeVisible();
  }
  await expect(
    page.locator("main header").getByText(healthy ? "Operational" : "Degraded", { exact: true }),
  ).toBeVisible();
});

test("models displays the embedding configuration returned by the API", async ({
  adminPage: page,
}) => {
  const response = page.waitForResponse((r) => r.url().endsWith("/api/admin/models/info"));
  await page.goto("/admin/models");
  const result = await response;
  expect(result.ok()).toBeTruthy();
  const { embed_model } = await result.json();
  expect(embed_model).toBeTruthy();
  await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  await expect(page.getByText(embed_model, { exact: true }).first()).toBeVisible();
});

test("vector DB refreshes and expands a real collection", async ({ adminPage: page }) => {
  const endpoint = "/api/admin/vectordb/info";
  const response = page.waitForResponse((r) => r.url().endsWith(endpoint));
  await page.goto("/admin/vectordb");
  const result = await response;
  expect(result.ok()).toBeTruthy();
  const { collections } = await result.json();
  expect(collections.length).toBeGreaterThan(0);
  const collection = page.getByRole("button", {
    name: `Details for ${collections[0].name}`,
    exact: true,
  });
  await collection.click();
  await expect(collection).toHaveAttribute("aria-expanded", "true");
  await collection.click();
  await expect(collection).toHaveAttribute("aria-expanded", "false");
  const refreshed = page.waitForResponse((r) => r.url().endsWith(endpoint));
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  expect((await refreshed).ok()).toBeTruthy();
});

test("settings diagnostics can find a known parameter", async ({ adminPage: page }) => {
  const response = page.waitForResponse((r) => r.url().endsWith("/api/admin/config/effective"));
  await page.goto("/admin/settings");
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  await page.getByRole("tab", { name: "Diagnostics" }).click();
  expect((await response).ok()).toBeTruthy();
  await expect(page.getByRole("tab", { name: "Diagnostics" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  const field = page.getByRole("textbox", { name: "Find a configuration parameter" });
  await field.fill("embedding_model");
  await expect(page.getByText("embedding_model", { exact: true })).toBeVisible();
});

test("jobs applies the failed-only filter to the backend", async ({ adminPage: page }) => {
  await page.goto("/admin/jobs");
  await expect(page.getByRole("heading", { name: "Jobs", exact: true })).toBeVisible();
  const result = page.waitForResponse((r) => {
    const url = new URL(r.url());
    return url.pathname === "/api/admin/jobs" && url.searchParams.get("status") === "failed";
  });
  const status = page.getByRole("combobox", { name: "Job status" });
  const showFailed = page.getByRole("button", { name: "Show failed" });
  await expect(status.or(showFailed)).toBeVisible();
  const statusDropdown = await status.isVisible();
  if (statusDropdown) await status.selectOption("failed");
  else await showFailed.click();
  const response = await result;
  expect(response.ok()).toBeTruthy();
  const { jobs } = await response.json();
  if (statusDropdown) await expect(status).toHaveValue("failed");
  else await expect(page.getByRole("button", { name: "Show all" })).toBeVisible();
  const table = page.getByRole("table");
  for (const job of jobs) {
    expect(job.status).toBe("failed");
    const row = table.getByRole("row").filter({
      has: page.getByText(statusDropdown ? `#${job.id}` : String(job.id), { exact: true }),
    });
    await expect(row.getByRole("cell", { name: "failed", exact: true })).toBeVisible();
  }
  if (!jobs.length) await expect(table.getByText("No jobs found", { exact: true })).toBeVisible();
});

test("logs submit and clear a search and apply the severity filter", async ({
  adminPage: page,
}) => {
  await page.goto("/admin/logs");
  const { actions, search: field } = await logsControls(page);
  await page.getByRole("button", { name: actions ? "Pause" : "Auto: ON", exact: true }).click();
  await expect(
    page.getByRole("button", { name: actions ? "Resume" : "Auto: OFF", exact: true }),
  ).toBeVisible();
  const needle = unique("no-event");
  const result = page.waitForResponse((r) => {
    const url = new URL(r.url());
    return url.pathname === "/api/admin/logs" && url.searchParams.get("search") === needle;
  });
  await field.fill(needle);
  await field.press("Enter");
  expect((await result).ok()).toBeTruthy();
  await expect(
    page.getByText(actions ? "No matching events" : "No logs found", { exact: true }),
  ).toBeVisible();
  if (actions) await page.getByRole("button", { name: "Clear search" }).click();
  else await field.clear();
  await expect(field).toHaveValue("");
  const severity = page.waitForResponse((r) => {
    const url = new URL(r.url());
    return url.pathname === "/api/admin/logs" && url.searchParams.get("level") === "WARNING";
  });
  if (actions) await page.getByRole("combobox", { name: "Level" }).selectOption("WARNING");
  else await page.getByRole("button", { name: "WARNING", exact: true }).click();
  expect((await severity).ok()).toBeTruthy();
});

test("vector DB reports an API outage and retry restores real data", async ({
  adminPage: page,
}) => {
  const endpoint = "**/api/admin/vectordb/info";
  // Simulate this one endpoint failing without stopping the service for other tests.
  const outage = (route: import("@playwright/test").Route) =>
    route.fulfill({ status: 503, json: { message: "Temporary service outage" } });
  await page.route(endpoint, outage);
  await page.goto("/admin/vectordb");
  const alert = page
    .getByRole("alert")
    .filter({ hasText: "Could not refresh Vector DB information" });
  await expect(alert).toBeVisible();
  await page.unroute(endpoint, outage);
  const realResponse = page.waitForResponse((r) => r.url().endsWith("/api/admin/vectordb/info"));
  await alert.getByRole("button", { name: "Try again" }).click();
  expect((await realResponse).ok()).toBeTruthy();
  await expect(alert).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Details for / }).first()).toBeVisible();
});

test("settings tabs display their configuration panels", async ({ adminPage: page }) => {
  await page.goto("/admin/settings");
  const tabs = page.getByRole("tablist", { name: "Settings sections" });
  for (const name of [
    "Chat & answers",
    "Search & sources",
    "Document processing",
    "Model connections",
    "System & limits",
    "Search collection",
  ]) {
    const tab = tabs.getByRole("tab", { name: new RegExp(`^${name}`) });
    await tab.click();
    await expect(tab).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("tabpanel", { name: new RegExp(`^${name}`) })).toBeVisible();
  }
});

test("settings keep the reasoning toggle below the main controls and adapt to the card width", async ({
  adminPage: page,
}) => {
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.goto("/admin/settings");
  const model = page.getByRole("group", { name: "Answer model", exact: true });
  const creativity = page.getByRole("group", { name: "Creativity (temperature)", exact: true });
  const key = page.getByText("llm_disable_reasoning", { exact: true });
  const toggle = page.getByRole("button", {
    name: "Enable Answer without reasoning (thinking off)",
  });
  await expect(toggle).toBeVisible();
  const keyBox = (await key.boundingBox())!;
  expect(keyBox.y).toBeGreaterThan((await model.boundingBox())!.y);
  expect(keyBox.y).toBeGreaterThan((await creativity.boundingBox())!.y);
  expect(keyBox.width).toBeGreaterThan(120);
  expect((await toggle.boundingBox())!.y).toBeGreaterThan(keyBox.y + keyBox.height);

  await page.setViewportSize({ width: 1600, height: 900 });
  await expect
    .poll(async () => (await toggle.boundingBox())!.x - (await key.boundingBox())!.x)
    .toBeGreaterThan(keyBox.width);
});
