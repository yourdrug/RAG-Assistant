import { expect, test } from "./support/fixtures";

test.use({ screenshot: "on", trace: "on" });

for (const [url, heading] of [
  ["/chat", "Chat"],
  ["/admin/logs", /^(Actions|Logs)$/],
] as const) {
  test(`mobile hamburger has an accessible name on ${url}`, async ({ adminPage: page }) => {
    await page.goto(url);
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
    const hamburger = page.getByRole("banner").getByRole("button", { name: "", exact: true });
    test.fail(
      true,
      "Known accessibility defect: the icon-only sidebar trigger has no accessible name.",
    );
    await expect(hamburger).toHaveCount(0, { timeout: 1_000 });
  });
}
