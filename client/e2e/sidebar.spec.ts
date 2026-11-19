import { expect, test } from "./support/fixtures";

test.use({ viewport: { width: 920, height: 960 } });

for (const [route, heading] of [
  ["/documents", "Documents"],
  ["/admin", "Admin Dashboard"],
] as const) {
  for (const colorScheme of ["light", "dark"] as const) {
    test(`sidebar stays opaque below the desktop breakpoint on ${route} in ${colorScheme} mode`, async ({
      adminPage: page,
    }) => {
      await page.emulateMedia({ colorScheme, reducedMotion: "reduce" });
      await page.goto(route);
      await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
      await expect(page.locator("html")).toHaveClass(colorScheme);

      const sidebar = page.getByRole("complementary");
      await expect(sidebar).not.toBeInViewport();
      await page.getByRole("banner").getByRole("button", { name: "Open navigation" }).click();
      await expect(sidebar).toBeInViewport();

      // A canvas normalizes the alpha channel from both legacy RGB and modern OKLCH colors.
      const alpha = await sidebar.evaluate((element) => {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 1;
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Could not create a canvas to check the sidebar color");
        context.fillStyle = getComputedStyle(element).backgroundColor;
        context.fillRect(0, 0, 1, 1);
        return context.getImageData(0, 0, 1, 1).data[3];
      });
      expect(alpha).toBe(255);
    });
  }
}
