import { expect, test } from "./support/fixtures";

for (const [url, heading] of [
  ["/chat", "Chat"],
  ["/admin/logs", /^(Actions|Logs)$/],
] as const) {
  test(`named mobile navigation button opens the sidebar on ${url}`, async ({
    adminPage: page,
  }) => {
    await page.goto(url);
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
    const menu = page.getByRole("banner").getByRole("button", { name: "Open navigation" });
    const sidebar = page.getByRole("complementary");
    await expect(menu).toHaveAttribute("aria-expanded", "false");
    await expect(sidebar).not.toBeInViewport();
    await menu.click();
    await expect(menu).toHaveAttribute("aria-expanded", "true");
    await expect(sidebar).toBeInViewport();
  });
}
