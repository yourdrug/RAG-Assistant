import { choose, expect, test, unique } from "./support/fixtures";

test("users create, filter, edit role, deactivate and reactivate a test account", async ({
  adminPage: page,
  adminApi,
}) => {
  const email = `${unique("created-user")}@playwright.example.com`;
  await page.getByRole("link", { name: "People & Access" }).click();
  await expect(page.getByRole("heading", { name: "Users", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Add User" }).click();
  const creation = page.getByRole("dialog", { name: "Create User" });
  const submit = creation.getByRole("button", { name: "Create", exact: true });
  await expect(submit).toBeDisabled();
  await creation.getByPlaceholder("user@example.com").fill(email);
  await creation.getByPlaceholder("••••••••").fill("Pw-local-user-51!");
  const created = page.waitForResponse(
    (r) => r.url().endsWith("/api/auth/users") && r.request().method() === "POST",
  );
  await submit.click();
  const result = await created;
  expect(result.ok()).toBeTruthy();
  const user = await result.json();
  try {
    expect(user.email).toBe(email);
    const search = page.getByPlaceholder("Search by email...");
    await search.fill(email);
    const row = page.getByRole("row").filter({ hasText: email });
    await expect(row.getByText("Active", { exact: true })).toBeVisible();
    await row.getByRole("button", { name: "Role", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Change Role" });
    await expect(dialog.getByRole("button", { name: "Save" })).toBeDisabled();
    await dialog.getByRole("combobox").click();
    await choose(page, "Curator");
    await dialog.getByRole("button", { name: "Save" }).click();
    await expect(row.getByText("curator", { exact: true })).toBeVisible();
    await row.getByRole("button", { name: "Deactivate" }).click();
    await expect(row.getByText("Inactive", { exact: true })).toBeVisible();
    await row.getByRole("button", { name: "Activate" }).click();
    await expect(row.getByText("Active", { exact: true })).toBeVisible();
  } finally {
    await expect(await adminApi.patch(`/auth/users/${user.id}?is_active=false`)).toBeOK();
  }
});

test("groups create and persist membership changes", async ({ adminPage: page, makeUser }) => {
  const user = await makeUser();
  const name = unique("group");
  await page.goto("/admin/users");
  await page
    .getByRole("navigation", { name: "People & Access" })
    .getByRole("link", { name: "Groups" })
    .click();
  await expect(page.getByRole("heading", { name: "Groups", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Create Group" }).click();
  const dialog = page.getByRole("dialog", { name: "Create Group" });
  await expect(dialog.getByRole("button", { name: "Create", exact: true })).toBeDisabled();
  await dialog.getByPlaceholder("e.g., Marketing Team").fill(name);
  await dialog.getByRole("button", { name: "Create", exact: true }).click();
  await page.getByPlaceholder("Search groups...").fill(name);
  const row = page.getByRole("row").filter({ hasText: name });
  await expect(row).toBeVisible();
  await row.getByRole("button", { name: "Members" }).click();
  await expect(page.getByText("No members", { exact: true })).toBeVisible();
  await page.getByRole("combobox").click();
  await choose(page, user.email);
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.getByText(user.email, { exact: true })).toBeVisible();
  await page.reload();
  await page.getByPlaceholder("Search groups...").fill(name);
  await row.getByRole("button", { name: "Members" }).click();
  const member = page.getByText(user.email, { exact: true });
  await expect(member).toBeVisible();
  await member.locator("..").getByRole("button").click();
  await expect(page.getByText("No members", { exact: true })).toBeVisible();
});

test("curator scope starts empty and can grant and remove a user assignment", async ({
  adminPage: page,
  makeUser,
  adminApi,
}) => {
  const curator = await makeUser({ role: "curator" });
  const user = await makeUser();
  await page.goto("/admin/curators");
  await expect(page.getByRole("heading", { name: "Curators", exact: true })).toBeVisible();
  await page.getByText(curator.email, { exact: true }).click();
  await expect(page.getByText("No users assigned")).toBeVisible();
  await page
    .getByText("Assigned Users", { exact: true })
    .locator("..")
    .getByRole("button", { name: "Assign", exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "Assign User to Curator" });
  await expect(dialog.getByRole("button", { name: "Assign", exact: true })).toBeDisabled();
  await dialog.getByRole("combobox").click();
  await choose(page, `${user.email} (internal)`);
  await dialog.getByRole("button", { name: "Assign", exact: true }).click();
  const assignment = page.getByText(user.email, { exact: true });
  try {
    await expect(assignment).toBeVisible();
    await assignment.locator("../..").getByRole("button").click();
    await expect(page.getByText("No users assigned")).toBeVisible();
  } finally {
    const response = await adminApi.get(`/admin/curators/${curator.id}/scope`);
    await expect(response).toBeOK();
    const scope = await response.json();
    if (scope.managed_internal_ids.includes(user.id)) {
      await expect(
        await adminApi.delete(`/admin/curators/${curator.id}/users/${user.id}`),
      ).toBeOK();
    }
  }
});

test("client API key is shown once and can be revoked", async ({
  adminPage: page,
  makeUser,
  adminApi,
}) => {
  const client = await makeUser({ kind: "client" });
  const endpoint = `/clients/${client.id}/api-keys`;
  const name = unique("key");
  await page.goto("/admin/api-keys");
  await expect(page.getByRole("heading", { name: "Client API Keys" })).toBeVisible();
  await page.getByRole("combobox").click();
  await choose(page, `${client.email} (Active)`);
  await expect(page.getByText("No API keys found for this client")).toBeVisible();
  try {
    const keyName = page.getByRole("textbox", { name: "Key name" });
    const createButton = page.getByRole("button", { name: "Create", exact: true });
    await expect(createButton).toBeDisabled();
    await keyName.fill("   ");
    await expect(createButton).toBeDisabled();
    await keyName.fill(name);
    await createButton.click();
    const dialog = page.getByRole("dialog", { name: "API Key Created" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("textbox")).not.toHaveValue("");
    await dialog.getByRole("button", { name: "Done", exact: true }).click();
    const list = await adminApi.get(endpoint);
    await expect(list).toBeOK();
    expect(await list.json()).toEqual([expect.objectContaining({ name, is_active: true })]);
    await page
      .getByText(name, { exact: true })
      .locator("../..")
      .locator("..")
      .getByRole("button")
      .click();
    await expect(page.getByText("Revoked", { exact: true })).toBeVisible();
    const revoked = await adminApi.get(endpoint);
    await expect(revoked).toBeOK();
    expect(await revoked.json()).toEqual([expect.objectContaining({ name, is_active: false })]);
  } finally {
    const list = await adminApi.get(endpoint);
    await expect(list).toBeOK();
    for (const key of await list.json()) {
      if (key.is_active) await expect(await adminApi.delete(`${endpoint}/${key.id}`)).toBeOK();
    }
  }
});
