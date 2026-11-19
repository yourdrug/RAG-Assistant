import { expect, test, unique } from "./support/fixtures";

test("knowledge space validates, creates, renames and deletes with confirmation", async ({
  adminPage: page,
  adminApi,
}) => {
  const name = unique("space-crud");
  const updated = `${name}-updated`;
  await page.goto("/admin/spaces");
  await expect(page.getByRole("heading", { name: "Knowledge Spaces" })).toBeVisible();
  await page.getByRole("button", { name: "Create Space" }).click();
  const create = page.getByRole("dialog", { name: "Create Space" });
  await expect(create.getByRole("button", { name: "Create", exact: true })).toBeDisabled();
  try {
    await create.getByPlaceholder("e.g., Customer support").fill(name);
    await create.getByRole("button", { name: "Create", exact: true }).click();
    const search = page.getByPlaceholder("Search spaces...");
    await search.fill(name);
    const row = page.getByRole("row").filter({ hasText: name });
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: "Edit" }).click();
    const edit = page.getByRole("dialog", { name: "Edit Space" });
    await expect(edit.getByRole("combobox")).toBeDisabled();
    await edit.getByPlaceholder("e.g., Customer support").fill(updated);
    await edit.getByRole("button", { name: "Save" }).click();
    await expect(row.getByText(updated, { exact: true })).toBeVisible();
    await row.getByRole("button", { name: "Delete" }).click();
    const confirmation = page.getByRole("alertdialog", { name: "Delete Space" });
    await confirmation.getByRole("button", { name: "Cancel" }).click();
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: "Delete" }).click();
    await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(row).toHaveCount(0);
    await page.reload();
    await search.fill(name);
    await expect(page.getByText("No results.", { exact: true })).toBeVisible();
  } finally {
    const listing = await adminApi.get("/spaces");
    await expect(listing).toBeOK();
    const records = await listing.json();
    for (const record of records.filter((item: { name: string }) =>
      [name, updated].includes(item.name),
    )) {
      await expect(await adminApi.delete(`/spaces/${record.id}`)).toBeOK();
    }
  }
});

test("admin documents filter and validate the manual document dialog", async ({
  adminPage: page,
}) => {
  await page.goto("/admin/documents");
  await expect(page.getByRole("heading", { name: "Documents", exact: true })).toBeVisible();
  await page.getByPlaceholder("Search documents...").fill(unique("no-document"));
  await expect(page.getByText("No results.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /manual document/i }).click();
  const dialog = page.getByRole("dialog", { name: "Create Manual Document" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Create", exact: true }).click();
  await expect(page.getByText("Title is required")).toBeVisible();
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toBeHidden();
});

test("act versions loads the real review list", async ({ adminPage: page }) => {
  const loaded = page.waitForResponse((r) => r.url().includes("/api/admin/act-versions"));
  await page.goto("/admin/act-versions");
  expect((await loaded).ok()).toBeTruthy();
  await expect(page.getByRole("heading", { name: "Act Versions" })).toBeVisible();
  await page.getByPlaceholder("Search by source...").fill(unique("no-source"));
  await expect(page.getByText("No results.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Next", exact: true })).toBeDisabled();
});

test("ingest prevents a missing single-file path and the registry can refresh", async ({
  adminPage: page,
}) => {
  await page.goto("/admin/ingest");
  await expect(page.getByRole("heading", { name: "Ingestion" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Ingest File" })).toBeDisabled();
  const refreshed = page.waitForResponse((r) => r.url().includes("/api/ingest/registry"));
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  expect((await refreshed).ok()).toBeTruthy();
  await expect(page.getByText("Ingestion Registry", { exact: true })).toBeVisible();
});

test("quality preview rejects invalid file formats and handles missing preview state", async ({
  adminPage: page,
}) => {
  await page.goto("/admin/quality");
  await expect(page.getByRole("heading", { name: "Indexing Quality" })).toBeVisible();
  await page.getByRole("button", { name: "Dry-Run Preview" }).click();
  const dialog = page.getByRole("dialog", { name: "Dry-Run Preview" });
  await dialog.locator('input[type="file"]').setInputFiles({
    name: "unsupported.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("test"),
  });
  await expect(page.getByText("Supported formats: PDF, DOCX, RTF")).toBeVisible();
  await dialog.getByRole("button", { name: "Close" }).click();
  await page.goto("/admin/quality/dry-run");
  await expect(page.getByText(/No dry-run results/)).toBeVisible();
  await page.getByRole("main").getByRole("link", { name: "Quality", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Indexing Quality" })).toBeVisible();
});

test("dry run extracts text from a real RTF preview", async ({ adminPage: page }) => {
  const name = `${unique("preview")}.rtf`;
  const text = "The Playwright preview validates extraction of this complete sentence.";
  await page.goto("/admin/quality");
  await page.getByRole("button", { name: "Dry-Run Preview" }).click();
  const dialog = page.getByRole("dialog", { name: "Dry-Run Preview" });
  const response = page.waitForResponse((r) => r.url().endsWith("/api/admin/documents/preview"));
  await dialog.locator('input[type="file"]').setInputFiles({
    name,
    mimeType: "application/rtf",
    buffer: Buffer.from(`{\\rtf1\\ansi ${text}\\par}`),
  });
  const result = await response;
  expect(result.status()).toBe(200);
  const preview = await result.json();
  expect(preview.pages).toHaveLength(1);
  await expect(page.getByRole("heading", { name: `Dry-Run: ${name}` })).toBeVisible();
  await page.getByRole("button", { name: new RegExp(preview.pages[0].label ?? "Unit 1") }).click();
  await expect(page.getByText("Extracted text")).toBeVisible();
  await expect(page.getByText(text, { exact: true })).toBeVisible();
});
