import { choose, deleteDocument, expect, login, test, unique } from "./support/fixtures";

test("reader appends the next page and search moves between pages of real chunks", async ({
  adminPage: page,
  adminApi,
  manualDocument: doc,
  makeSpace,
  space,
}) => {
  await makeSpace();
  const marker = unique("paginated");
  const contents = Array.from(
    { length: 51 },
    (_, index) => `${marker} synthetic handbook fragment ${index + 1}.`,
  );
  for (const content of contents) {
    await expect(
      await adminApi.post(`/documents/${doc.id}/chunks`, { data: { content } }),
    ).toBeOK();
  }

  await test.step("Reader loads the continuation without replacing or duplicating earlier chunks", async () => {
    await page.goto(`/documents/${doc.id}`);
    const fragments = page.getByRole("article").locator("section");
    const more = page.getByRole("button", { name: "Показать продолжение" });
    await expect(fragments).toHaveText(contents.slice(0, 50));
    await more.click();
    await expect(fragments).toHaveText(contents);
    await expect(more).toBeHidden();
  });

  await test.step("Search replaces the first page with the remaining result and can return", async () => {
    await page.goto("/search");
    await page.getByRole("combobox", { name: "Knowledge space" }).click();
    await choose(page, space.name);
    await page.getByRole("radio", { name: "Contains (ILIKE)" }).check();
    await page.getByPlaceholder("Search documents...").fill(marker);
    const pagination = page.getByRole("navigation", { name: "Search result pages" });
    const previous = pagination.getByRole("button", { name: "Previous" });
    const next = pagination.getByRole("button", { name: "Next" });
    const matches = page.getByText(/^chunk #\d+$/);
    await expect(page.getByText("Showing 1–50 of 51")).toBeVisible();
    await expect(matches).toHaveCount(50);
    const firstPage = await matches.allTextContents();
    expect(new Set(firstPage).size).toBe(50);
    await expect(previous).toBeDisabled();
    await next.click();
    await expect(page.getByText("Showing 51–51 of 51")).toBeVisible();
    await expect(matches).toHaveCount(1);
    expect(firstPage).not.toContain(await matches.textContent());
    await expect(next).toBeDisabled();
    await previous.click();
    await expect(page.getByText("Showing 1–50 of 51")).toBeVisible();
    await expect(matches).toHaveText(firstPage);
  });
});

test("admin edits a chunk and can cancel or confirm deleting it", async ({
  adminPage: page,
  adminApi,
  manualDocument: doc,
}) => {
  const content = "The original synthetic handbook entry.";
  const updated = "The revised synthetic handbook entry.";
  await expect(await adminApi.post(`/documents/${doc.id}/chunks`, { data: { content } })).toBeOK();
  const reader = `/documents/${doc.id}`;
  const manager = `/admin/documents?docId=${doc.id}`;

  await page.goto(manager);
  await page
    .getByRole("row")
    .filter({ hasText: content })
    .getByRole("button", { name: "Edit chunk" })
    .click();
  const edit = page.getByRole("dialog", { name: /Edit Chunk #/ });
  await expect(edit.getByRole("textbox")).toHaveValue(content);
  await edit.getByRole("textbox").fill(updated);
  await edit.getByRole("button", { name: "Save Changes" }).click();
  await expect(edit).toBeHidden();
  const row = page.getByRole("row").filter({ hasText: updated });
  await expect(row).toBeVisible();

  await page.goto(reader);
  await expect(page.getByRole("article")).toHaveText(updated);

  await page.goto(manager);
  const remove = row.getByRole("button", { name: "Delete chunk" });
  const confirmation = page.getByRole("alertdialog", { name: "Delete Chunk" });
  await remove.click();
  await confirmation.getByRole("button", { name: "Cancel" }).click();
  await expect(row).toBeVisible();
  await remove.click();
  await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(row).toHaveCount(0);

  await page.goto(reader);
  await expect(page.getByText("Текст документа пока не готов.")).toBeVisible();
});

test("internal user sees public chunks and cannot read or search another owner's private chunks", async ({
  page,
  adminApi,
  makeUser,
  makeSpace,
  manualDocument: privateDoc,
  space,
}) => {
  await makeSpace();
  const user = await makeUser();
  const marker = unique("visibility");
  const publicTitle = unique("public-document");
  const publicText = `${marker} is shared with internal users.`;
  const privateText = `${marker} is private to the owner.`;
  const created = await adminApi.post("/documents/manual", {
    data: { title: publicTitle, visibility: "internal_public", space_id: space.id },
  });
  await expect(created).toBeOK();
  const { id: publicId } = await created.json();

  try {
    for (const [id, content] of [
      [publicId, publicText],
      [privateDoc.id, privateText],
    ]) {
      await expect(await adminApi.post(`/documents/${id}/chunks`, { data: { content } })).toBeOK();
    }

    await login(page, user);
    await page.goto(`/documents/${publicId}`);
    await expect(page.getByRole("heading", { name: publicTitle })).toBeVisible();
    await expect(page.getByRole("article")).toHaveText(publicText);

    await page.goto(`/documents/${privateDoc.id}`);
    await expect(page.getByRole("alert")).toBeVisible();
    await expect(page.getByRole("heading", { name: privateDoc.filename })).toHaveCount(0);
    await expect(page.getByRole("article")).toHaveCount(0);

    await page.goto("/search");
    await page.getByRole("combobox", { name: "Knowledge space" }).click();
    await choose(page, space.name);
    await page.getByRole("radio", { name: "Contains (ILIKE)" }).check();
    await page.getByPlaceholder("Search documents...").fill(marker);
    await expect(page.getByText("Showing 1–1 of 1")).toBeVisible();
    await expect(page.getByText(publicTitle, { exact: true })).toBeVisible();
    await expect(page.getByText(privateDoc.filename, { exact: true })).toHaveCount(0);
  } finally {
    await deleteDocument(adminApi, publicId);
  }
});
