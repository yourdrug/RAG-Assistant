import { choose, deleteDocument, expect, test, unique } from "./support/fixtures";

test("new chat suppresses blank submission and can reset an unsent draft", async ({
  adminPage: page,
}) => {
  const composer = page.getByPlaceholder("Ask a question about your documents...");
  await expect(page.getByText("Start a conversation")).toBeVisible();
  await expect(page.getByRole("button", { name: "Отправить сообщение" })).toBeDisabled();
  // The availability of the composer depends on a selected knowledge space.
  await expect(composer).toBeVisible();
  await page.getByRole("button", { name: "New Chat", exact: true }).click();
  await expect(composer).toHaveValue("");
  await expect(page).toHaveURL(/\/chat$/);
});

test("documents filter, open the reader and delete the test document", async ({
  adminPage: page,
  adminApi,
  manualDocument: doc,
}) => {
  await page
    .getByRole("link", { name: "Documents", exact: true })
    .and(page.locator('a[href="/documents"]'))
    .click();
  await expect(page.getByRole("heading", { name: "Documents", exact: true })).toBeVisible();
  const search = page.getByPlaceholder("Search documents...");
  const row = page.getByRole("row").filter({ hasText: doc.filename });
  await search.fill(doc.filename);
  await expect(row).toBeVisible();
  await search.fill(unique("missing"));
  await expect(row).toHaveCount(0);
  await page.goto(`/documents/${doc.id}`);
  await expect(page.getByRole("heading", { name: doc.filename })).toBeVisible();
  await expect(page.getByText("Текст документа пока не готов.")).toBeVisible();
  await page.getByRole("link", { name: "Документы", exact: true }).click();
  await expect(page).toHaveURL(/\/documents$/);
  await search.fill(doc.filename);
  await row.getByRole("button", { name: "Delete" }).click();
  const confirmation = page.getByRole("alertdialog", { name: "Delete Document" });
  await confirmation.getByRole("button", { name: "Cancel" }).click();
  await expect(row).toBeVisible();
  await row.getByRole("button", { name: "Delete" }).click();
  await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
  await expect
    .poll(async () => (await adminApi.get(`/documents/${doc.id}`)).status(), {
      timeout: 20_000,
    })
    .toBe(404);
  await expect(row).toHaveCount(0);
});

test("document reader explains a missing record", async ({ adminPage: page }) => {
  await page.goto("/documents/2147483647");
  await expect(page.getByRole("alert")).toHaveText(/Документ недоступен/);
  await expect(page.getByRole("link", { name: "Документы", exact: true })).toBeVisible();
});

test("search validates, finds a real chunk in both modes and handles an empty result", async ({
  adminPage: page,
  space,
  makeSpace,
  manualDocument: doc,
  adminApi,
}) => {
  await makeSpace();
  const marker = unique("searchmarker");
  const content = `${marker} marks a synthetic handbook entry used exclusively for browser regression testing.`;
  const chunk = await adminApi.post(`/documents/${doc.id}/chunks`, { data: { content } });
  await expect(chunk).toBeOK();
  await page.goto(`/documents/${doc.id}`);
  await expect(page.getByRole("article").getByText(content, { exact: true })).toBeVisible();

  await page.goto("/search");
  const query = page.getByPlaceholder("Search documents...");
  await query.fill("ab");
  await expect(page.getByText("Minimum 3 characters to search")).toBeVisible();
  await page.getByRole("combobox", { name: "Knowledge space" }).click();
  await choose(page, space.name);
  await query.fill(marker);
  await expect(page.getByText(doc.filename, { exact: true })).toBeVisible();
  await expect(page.getByText(/Found 1 result .*pg_trgm ranked/)).toBeVisible();
  await page.getByRole("radio", { name: "Contains (ILIKE)" }).check();
  await expect(page.getByText(/Found 1 result .*contains/)).toBeVisible();
  await query.fill(unique("unmatched"));
  await expect(page.getByText("No results found")).toBeVisible();
});

test("document upload stores a real text file and lists the new record", async ({
  adminPage: page,
  space,
  adminApi,
}) => {
  const filename = `${unique("upload")}.txt`;
  await page.goto("/documents");
  await page.locator('input[type="file"]').setInputFiles({
    name: filename,
    mimeType: "text/plain",
    buffer: Buffer.from("Synthetic test handbook. This is a local document for Playwright."),
  });
  const dialog = page.getByRole("dialog", { name: "Upload Documents" });
  await expect(dialog.getByText(filename, { exact: true })).toBeVisible();
  const spacePicker = dialog
    .getByText("Knowledge Space", { exact: true })
    .locator("..")
    .getByRole("combobox");
  await spacePicker.click();
  await choose(page, space.name);

  let id: number | undefined;
  try {
    const accepted = page.waitForResponse(
      (r) => r.url().endsWith("/api/documents") && r.request().method() === "POST",
    );
    await dialog.getByRole("button", { name: "Upload", exact: true }).click();
    const upload = await accepted;
    expect(upload.ok()).toBeTruthy();
    const data = await upload.json();
    expect(data.filename).toBe(filename);
    id = data.document_id;
    expect(id).toBeGreaterThan(0);
    await expect(dialog).toBeHidden();
    await page.getByPlaceholder("Search documents...").fill(filename);
    await expect(page.getByRole("row").filter({ hasText: filename })).toBeVisible();
    const record = await adminApi.get(`/documents/${id}`);
    await expect(record).toBeOK();
    expect(await record.json()).toMatchObject({ filename, space_id: space.id });
  } finally {
    if (id !== undefined) await deleteDocument(adminApi, id);
  }
});
