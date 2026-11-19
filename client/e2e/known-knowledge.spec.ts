import { expect, test } from "./support/fixtures";

test.use({ screenshot: "on", trace: "on" });

test("act versions has valid HTML after the count loads", async ({ adminPage: page }) => {
  const loaded = page.waitForResponse((r) => r.url().includes("/api/admin/act-versions"));
  await page.goto("/admin/act-versions");
  const result = await loaded;
  expect(result.ok()).toBeTruthy();
  const { total } = await result.json();
  const summary = page.getByText(/^All versions/);
  await expect(summary).toHaveText(new RegExp(`All versions\\s*${total}$`));
  test.fail(true, "Known defect: Act Versions renders a block badge directly inside a paragraph.");
  await expect(summary.and(page.locator("p")).locator(":scope > div")).toHaveCount(0, {
    timeout: 1_000,
  });
});
