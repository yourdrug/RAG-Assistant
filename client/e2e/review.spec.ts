import { readFile } from "node:fs/promises";
import { expect, test, unique } from "./support/fixtures";

test("answer logs combine triage and search and export the filtered data", async ({
  adminPage: page,
}) => {
  await page.goto("/admin/chat-logs");
  await expect(page.getByRole("heading", { name: "Answer logs" })).toBeVisible();
  const triage = page.getByRole("button", { name: "Needs review" });
  await triage.click();
  await expect(triage).toHaveAttribute("aria-pressed", "true");
  const search = page.getByRole("searchbox", { name: "Search responses" });
  const needle = unique("no-answer-log");
  const response = page.waitForResponse((r) => {
    const url = new URL(r.url());
    return (
      url.pathname === "/api/admin/chat-logs" && [...url.searchParams.values()].includes(needle)
    );
  });
  await search.fill(needle);
  expect((await response).ok()).toBeTruthy();
  await expect(page.getByRole("heading", { name: "No matching responses" })).toBeVisible();
  await page.getByText(/Advanced filters/).click();
  const sorted = page.waitForResponse((r) => {
    const url = new URL(r.url());
    return url.pathname === "/api/admin/chat-logs" && url.searchParams.get("sort_by") === "latency";
  });
  await page.getByRole("combobox", { name: "Sort by" }).selectOption("latency");
  expect((await sorted).ok()).toBeTruthy();
  const ordered = page.waitForResponse((r) => {
    const url = new URL(r.url());
    return url.pathname === "/api/admin/chat-logs" && url.searchParams.get("sort_order") === "asc";
  });
  await page.getByRole("combobox", { name: "Order" }).selectOption("asc");
  expect((await ordered).ok()).toBeTruthy();
  const finished = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON", exact: true }).click();
  const download = await finished;
  expect(download.suggestedFilename()).toMatch(/\.json$/);
  expect(JSON.parse(await readFile(await download.path(), "utf8"))).toEqual([]);
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(search).toHaveValue("");
});

for (const [slug, heading] of [
  ["knowledge", "Improve knowledge"],
  ["benchmark-question", "Create a benchmark question"],
]) {
  test(`answer log workspace handles a missing log in ${slug}`, async ({ adminPage: page }) => {
    await page.goto(`/admin/chat-logs/2147483647/${slug}`);
    await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    await expect(page.getByRole("alert")).toHaveText(/Could not load this answer/);
  });
}

for (const [tab, slug] of [
  ["Questions", "questions"],
  ["Sweep Builder", "sweep-builder"],
  ["Sweeps", "progress"],
  ["Leaderboard", "leaderboard"],
  ["Trends", "trends"],
  ["Acceptance", "acceptance"],
]) {
  test(`benchmark opens and restores the ${tab} tab`, async ({ adminPage: page }) => {
    await page.goto(tab === "Questions" ? "/admin/benchmark?tab=trends" : "/admin/benchmark");
    await expect(page.getByRole("heading", { name: "Benchmark Lab" })).toBeVisible();
    const trigger = page.getByRole("tab", { name: tab, exact: true });
    await trigger.click();
    await expect(trigger).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("tabpanel", { name: tab, exact: true })).toBeVisible();
    await expect(page).toHaveURL(new RegExp(`[?&]tab=${slug}(?:&|$)`));
    await page.reload();
    await expect(trigger).toHaveAttribute("aria-selected", "true");
  });
}

test("benchmark creates, edits, searches and deletes a question", async ({
  adminPage: page,
  adminApi,
}) => {
  const prefix = unique("question");
  const question = `${prefix} original?`;
  const edit = `${prefix} edited?`;
  await page.goto("/admin/benchmark");
  const search = page.getByPlaceholder("Search questions...");
  await search.fill(prefix);
  await expect(page.getByText(/No questions yet/)).toBeVisible();
  try {
    await page.getByRole("button", { name: "Add Question" }).click();
    const form = page.getByRole("dialog", { name: "Add question" });
    await form.getByRole("textbox", { name: /^Question/ }).fill(question);
    await form
      .getByLabel("Expected answer", { exact: true })
      .fill("Reference answer for Playwright.");
    await form.getByRole("button", { name: "Create question" }).click();
    const item = page
      .getByRole("article")
      .filter({ has: page.getByRole("heading", { name: prefix, exact: false }) });
    await expect(item.getByRole("heading", { name: question, exact: true })).toBeVisible();
    await item.getByRole("button", { name: "Edit" }).click();
    const editor = page.getByRole("dialog", { name: "Edit question" });
    await editor.getByRole("textbox", { name: /^Question/ }).fill(edit);
    await editor.getByRole("button", { name: "Save changes" }).click();
    await expect(item.getByRole("heading", { name: edit, exact: true })).toBeVisible();
    page.once("dialog", (confirmation) => confirmation.dismiss());
    await item.getByRole("button", { name: /Delete question/ }).click();
    await expect(item).toBeVisible();
    page.once("dialog", (confirmation) => confirmation.accept());
    await item.getByRole("button", { name: /Delete question/ }).click();
    await expect(page.getByText(/No questions yet/)).toBeVisible();
  } finally {
    const response = await adminApi.get("/admin/benchmark/questions", {
      params: { search: prefix },
    });
    await expect(response).toBeOK();
    for (const record of (await response.json()).questions) {
      if ([question, edit].includes(record.question)) {
        await expect(await adminApi.delete(`/admin/benchmark/questions/${record.id}`)).toBeOK();
      }
    }
  }
});

test("benchmark rejects invalid import JSON and paginates eleven imported questions", async ({
  adminPage: page,
  adminApi,
}) => {
  const prefix = unique("import");
  const questions = Array.from({ length: 11 }, (_, i) => ({
    question: `${prefix} example ${i + 1}?`,
    expected_answer: "Synthetic answer",
    dataset: "belarus_law-main",
  }));
  await page.goto("/admin/benchmark");
  await page.getByPlaceholder("Search questions...").fill(prefix);
  await expect(page.getByText(/No questions yet/)).toBeVisible();
  await page.getByRole("button", { name: "Import", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Import Questions" });
  const input = dialog.getByRole("textbox");
  const submit = dialog.getByRole("button", { name: "Import", exact: true });
  await expect(submit).toBeDisabled();
  await input.fill("{");
  await submit.click();
  await expect(page.getByText("Invalid JSON", { exact: true })).toBeVisible();
  try {
    await input.fill(JSON.stringify(questions));
    await submit.click();
    await expect(dialog).toBeHidden();
    const pagination = page.getByRole("navigation", { name: "Questions pagination" }).first();
    await expect(pagination).toContainText("1–10 of 11 questions");
    await expect(pagination.getByRole("button", { name: "Previous" })).toBeDisabled();
    const matches = page.getByRole("article").filter({ hasText: prefix });
    await expect(matches).toHaveCount(10);
    await pagination.getByRole("button", { name: "Next" }).click();
    await expect(pagination).toContainText("11–11 of 11 questions");
    await expect(matches).toHaveCount(1);
    await expect(pagination.getByRole("button", { name: "Next" })).toBeDisabled();
    await pagination.getByRole("button", { name: "Previous" }).click();
    await expect(matches).toHaveCount(10);
  } finally {
    const response = await adminApi.get("/admin/benchmark/questions", {
      params: { search: prefix, limit: 20 },
    });
    await expect(response).toBeOK();
    for (const record of (await response.json()).questions) {
      if (questions.some(({ question }) => question === record.question)) {
        await expect(await adminApi.delete(`/admin/benchmark/questions/${record.id}`)).toBeOK();
      }
    }
  }
});
